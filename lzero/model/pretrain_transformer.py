import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os
from tqdm import tqdm
import wandb
from dataclasses import dataclass
import selfies as sf
import csv

from lzero.model.muzero_transformer import MassSelfiesED, SelfiesTokenizer
from lzero.model.muzero_transformer import get_actions_list


@dataclass
class PretrainConfig:
    """预训练配置"""
    # 模型参数
    vocab_size: int = 1000
    max_len: int = 100  
    d_model: int = 128  
    n_enc: int = 4      
    n_dec: int = 6      
    n_head: int = 8     
    dropout: float = 0.25 
    
    # 训练参数 
    batch_size: int = 16  
    learning_rate: float = 5e-5  
    num_epochs: int = 100
    warmup_steps: int = 2000  
    gradient_clip: float = 0.5  
    weight_decay: float = 0.05 
    
    # Early stopping
    early_stopping_patience: int = 10  
    early_stopping_min_delta: float = 0.005  
    save_best_model: bool = True
    
    # 数据参数 (fingerprint )
    spectrum_dim: int = 4096
    
    # 保存参数
    save_dir: str = "./pretrained_models"
    log_interval: int = 100
    save_interval: int = 1000
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据集参数
    train_data_file: str = "/hy-tmp/MCTS/MassEnv/DataLoader/train_spectrum_embeds_msg.pt"
    val_data_file: str = "/hy-tmp/MCTS/MassEnv/DataLoader/val_spectrum_embeds_msg.pt"


class RealSpectrumSelfiesDataset(Dataset):
    
    def __init__(self, data_file: str, tokenizer: SelfiesTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_file = data_file
        
        print(f"Loading real spectrum data from: {data_file}")
        self.data = torch.load(data_file, map_location='cpu')
        
        # 验证数据格式
        required_keys = ['embeds', 'smiles', 'formulas']
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"Missing key '{key}' in dataset file: {data_file}")
        
        # 转换SMILES到SELFIES
        print("Converting SMILES to SELFIES...")
        self.selfies_list = []
        valid_indices = []
        
        for i, smiles in enumerate(tqdm(self.data['smiles'], desc="Converting SMILES")):
            try:
                selfies = sf.encoder(smiles)
                if selfies and len(selfies) > 0:  # 确保SELFIES有效
                    self.selfies_list.append(selfies)
                    valid_indices.append(i)
            except Exception as e:
                # 跳过无效的SMILES
                continue
        
        # 过滤有效数据
        self.embeds = self.data['embeds'][valid_indices]
        self.formulas = [self.data['formulas'][i] for i in valid_indices]
        
        print(f"Loaded {len(self.selfies_list)} valid samples from {len(self.data['smiles'])} total samples")
        print(f"Spectrum embeddings shape: {self.embeds.shape}")
        
    def __len__(self):
        return len(self.selfies_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 获取spectrum embedding (已经是4096维)
        spectrum = self.embeds[idx].float()  # [4096]
        
        # 获取SELFIES
        selfies = self.selfies_list[idx]
        
        # 将SELFIES编码为token IDs
        tokens = [self.tokenizer.sos_token_id]
        try:
            selfies_tokens = self.tokenizer.encode_selfies(selfies, add_special_tokens=False)
            tokens.extend(selfies_tokens)
        except Exception as e:
            # 如果编码失败，使用空序列
            print(f"Warning: Failed to encode SELFIES '{selfies}': {e}")
            tokens = [self.tokenizer.sos_token_id, self.tokenizer.eos_token_id]
        
        # 截断或padding到max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_len - len(tokens)))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            'spectrum': spectrum,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'formula': self.formulas[idx],  # 用于调试
            'selfies': selfies  # 用于调试
        }


class PretrainLoss(nn.Module):
    """预训练损失函数"""
    
    def __init__(self, vocab_size: int, pad_token_id: int, label_smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        return self.loss_fn(logits, targets)


class WarmupLinearSchedule:
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = self.base_lr * min_lr_ratio
        
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup阶段：线性增长
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # 衰减阶段：线性衰减到最小学习率
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr - (self.base_lr - self.min_lr) * min(progress, 1.0)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


def create_model(config: PretrainConfig) -> MassSelfiesED:
    """创建模型"""
    return MassSelfiesED(
        vocab_size=config.vocab_size,
        max_len=config.max_len,
        d_model=config.d_model,
        n_enc=config.n_enc,
        n_dec=config.n_dec,
        n_head=config.n_head,
        dropout=config.dropout,
        device=config.device
    )


def pretrain_step(model: MassSelfiesED, batch: Dict[str, torch.Tensor], 
                 loss_fn: PretrainLoss, device: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    """单步预训练"""

    spectrum = batch['spectrum'].to(device)
    input_ids = batch['input_ids'].to(device)
    target_ids = batch['target_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    logits = model.forward_pretrain(spectrum, input_ids, attention_mask)  # (B, T, vocab_size)
    
    loss = loss_fn(logits, target_ids)
    

    predictions = torch.argmax(logits, dim=-1)  # (B, T)
    mask = (target_ids != loss_fn.pad_token_id)
    correct = (predictions == target_ids) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    

    perplexity = torch.exp(loss)
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'perplexity': perplexity.item()
    }
    
    return loss, metrics


def print_data_sample(dataset, tokenizer, sample_idx=0):
    print(f"\n{'='*60}")
    print(f"📊 DATA SAMPLE #{sample_idx}")
    print(f"{'='*60}")
    
    sample = dataset[sample_idx]
    
    print(f"🧬 Spectrum embedding shape: {sample['spectrum'].shape}")
    print(f"🧬 Spectrum embedding (first 10 values): {sample['spectrum'][:10].tolist()}")
    
    if 'selfies' in sample:
        print(f"🧪 Original SELFIES: {sample['selfies']}")
    if 'formula' in sample:
        print(f"🧪 Molecular Formula: {sample['formula']}")
    
    print(f"🔤 Input IDs shape: {sample['input_ids'].shape}")
    print(f"🔤 Target IDs shape: {sample['target_ids'].shape}")
    print(f"🔤 Attention mask shape: {sample['attention_mask'].shape}")
    
    # 解码tokens看看结果
    input_tokens = sample['input_ids'].tolist()
    target_tokens = sample['target_ids'].tolist()
    
    print(f"🔤 Input tokens (first 20): {input_tokens[:20]}")
    print(f"🔤 Target tokens (first 20): {target_tokens[:20]}")
    
    # 尝试解码
    try:
        decoded_input = tokenizer.decode_to_selfies(input_tokens, skip_special_tokens=True)
        decoded_target = tokenizer.decode_to_selfies(target_tokens, skip_special_tokens=True)
        print(f"🔤 Decoded input: {decoded_input}")
        print(f"🔤 Decoded target: {decoded_target}")
    except Exception as e:
        print(f"🔤 Decoding failed: {e}")
    
    print(f"{'='*60}\n")


def pretrain_transformer(config: PretrainConfig):
    """主预训练函数"""
    import time
    from datetime import datetime
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    tokenizer = SelfiesTokenizer(max_len=config.max_len)
    
    config.vocab_size = len(tokenizer.get_vocab())
    print(f"📚 Vocabulary size: {config.vocab_size}")
    
    print("🔬 Using REAL spectrum data from .pt files")
    
    train_dataset = RealSpectrumSelfiesDataset(
        config.train_data_file, 
        tokenizer, 
        config.max_len
    )
    
    val_dataset = None
    if os.path.exists(config.val_data_file):
        val_dataset = RealSpectrumSelfiesDataset(
            config.val_data_file,
            tokenizer,
            config.max_len
        )
    
    print_data_sample(train_dataset, tokenizer, sample_idx=0)
    if len(train_dataset) > 1:
        print_data_sample(train_dataset, tokenizer, sample_idx=1)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,  # 避免多进程问题
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,  # 避免多进程问题
            pin_memory=True if config.device == "cuda" else False
        )
    
    # 创建模型
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = WarmupLinearSchedule(optimizer, config.warmup_steps, total_steps)
    
    # 创建损失函数
    loss_fn = PretrainLoss(config.vocab_size, tokenizer.pad_token_id)
    
    # 训练循环变量
    model.train()
    global_step = 0
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练统计
    training_stats = {
        'val_losses': [],
        'train_losses': [],
        'start_time': time.time(),
        'early_stopped': False
    }
    
    # 设置日志记录功能
    log_training_step_fn = None
    if hasattr(config, 'enable_detailed_logging') and config.enable_detailed_logging:
        if hasattr(config, 'log_files') and config.log_files:
            from run_pretrain import log_training_step
            log_training_step_fn = lambda **kwargs: log_training_step(config.log_files['train_log'], **kwargs)
            print("📊 Detailed logging enabled")
        else:
            print("⚠️ Detailed logging requested but log files not found")
    
    # 简单的CSV logging (保持向后兼容)
    csv_log_path = os.path.join(config.save_dir, "validation_results.csv")
    csv_header_written = False
    
    print(f"\n🚀 Starting training for {config.num_epochs} epochs...")
    print(f"📊 Training samples: {len(train_dataset):,}")
    if val_dataset:
        print(f"📊 Validation samples: {len(val_dataset):,}")
    print(f"🔄 Steps per epoch: {len(train_loader):,}")
    print(f"🔄 Total steps: {total_steps:,}")
    
    for epoch in range(config.num_epochs):
        epoch_losses = []
        epoch_metrics = {'loss': [], 'accuracy': [], 'perplexity': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # 前向传播
            loss, metrics = pretrain_step(model, batch, loss_fn, config.device)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # 优化器步骤
            optimizer.step()
            scheduler.step()
            
            # 记录指标
            epoch_losses.append(loss.item())
            training_stats['train_losses'].append(loss.item())
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'ppl': f"{metrics['perplexity']:.2f}",
                'lr': f"{scheduler.get_lr():.2e}"
            })
            
            global_step += 1
            
            # 详细日志记录
            if global_step % config.log_interval == 0:
                current_lr = scheduler.get_lr()
                avg_recent_loss = np.mean(training_stats['train_losses'][-config.log_interval:])
                
                # 记录到详细日志
                if log_training_step_fn:
                    log_training_step_fn(
                        epoch=epoch + 1,
                        step=batch_idx + 1,
                        global_step=global_step,
                        train_loss=avg_recent_loss,
                        learning_rate=current_lr,
                        early_stopping_counter=patience_counter
                    )
                
                # 打印到控制台
                print(f"📈 Step {global_step}: loss={avg_recent_loss:.4f}, lr={current_lr:.2e}")
            
            # 保存检查点
            if global_step % config.save_interval == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'global_step': global_step,
                    'tokenizer_vocab': tokenizer.get_vocab(),
                    'training_stats': training_stats
                }
                torch.save(checkpoint, os.path.join(config.save_dir, f"checkpoint_step_{global_step}.pt"))
                print(f"💾 Saved checkpoint at step {global_step}")
        
        # Epoch结束后的验证
        current_val_loss = None
        if val_loader:
            model.eval()
            val_losses = []
            val_metrics = {'loss': [], 'accuracy': [], 'perplexity': []}
            
            print(f"🔍 Running validation after epoch {epoch+1}...")
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    loss, metrics = pretrain_step(model, batch, loss_fn, config.device)
                    val_losses.append(loss.item())
                    for key, value in metrics.items():
                        val_metrics[key].append(value)
            
            avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
            current_val_loss = avg_val_metrics['loss']
            val_perplexity = avg_val_metrics['perplexity']
            
            # 记录验证结果
            training_stats['val_losses'].append(current_val_loss)
            
            print(f"📊 Validation Results - Epoch {epoch+1}:")
            print(f"    Loss: {current_val_loss:.4f}")
            print(f"    Accuracy: {avg_val_metrics['accuracy']:.4f}")
            print(f"    Perplexity: {val_perplexity:.2f}")
            
            # 记录到详细日志
            is_best_model = False
            if log_training_step_fn:
                is_best_model = current_val_loss + config.early_stopping_min_delta < best_val_loss
                log_training_step_fn(
                    epoch=epoch + 1,
                    step=len(train_loader),
                    global_step=global_step,
                    train_loss=np.mean(epoch_losses),
                    learning_rate=scheduler.get_lr(),
                    val_loss=current_val_loss,
                    val_perplexity=val_perplexity,
                    best_val_loss=min(best_val_loss, current_val_loss),
                    is_best_model=is_best_model,
                    early_stopping_counter=patience_counter
                )
            
            # 简单CSV logging (向后兼容)
            if not csv_header_written:
                with open(csv_log_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['epoch', 'loss', 'accuracy', 'perplexity'])
                csv_header_written = True
            
            with open(csv_log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    epoch + 1,
                    current_val_loss,
                    avg_val_metrics['accuracy'],
                    avg_val_metrics['perplexity']
                ])
            
            # Early stopping logic
            if current_val_loss + config.early_stopping_min_delta < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                
                # Save best model
                if config.save_best_model:
                    best_checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch + 1,
                        'val_loss': current_val_loss,
                        'tokenizer_vocab': tokenizer.get_vocab(),
                        'training_stats': training_stats
                    }
                    torch.save(best_checkpoint, os.path.join(config.save_dir, "best_model.pt"))
                    print(f"🌟 Saved new best model with val_loss: {current_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Early stopping patience: {patience_counter}/{config.early_stopping_patience}")
            
            # Early stopping check
            if patience_counter >= config.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                training_stats['early_stopped'] = True
                break
            
            model.train()  # 回到训练模式
        
        # 即使没有验证集，也记录训练信息
        elif log_training_step_fn:
            avg_train_loss = np.mean(epoch_losses)
            log_training_step_fn(
                epoch=epoch + 1,
                step=len(train_loader),
                global_step=global_step,
                train_loss=avg_train_loss,
                learning_rate=scheduler.get_lr(),
                early_stopping_counter=patience_counter
            )
    
    # 训练结束
    training_time = time.time() - training_stats['start_time']
    training_stats['training_time'] = f"{training_time:.2f}s ({training_time/60:.1f}m)"
    training_stats['total_epochs'] = epoch + 1
    training_stats['total_steps'] = global_step
    training_stats['final_train_loss'] = training_stats['train_losses'][-1] if training_stats['train_losses'] else None
    training_stats['best_val_loss'] = best_val_loss if current_val_loss is not None else None
    training_stats['best_epoch'] = None
    
    # 找到最佳epoch
    if training_stats['val_losses']:
        best_epoch_idx = np.argmin(training_stats['val_losses'])
        training_stats['best_epoch'] = best_epoch_idx + 1
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️ Total time: {training_stats['training_time']}")
    print(f"📊 Total epochs: {training_stats['total_epochs']}")
    print(f"🔄 Total steps: {training_stats['total_steps']}")
    if training_stats['best_val_loss'] is not None:
        print(f"🌟 Best validation loss: {training_stats['best_val_loss']:.4f} (epoch {training_stats['best_epoch']})")
    if training_stats['early_stopped']:
        print(f"🛑 Training stopped early due to no improvement")
    
    # 保存最终模型
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'final_epoch': epoch + 1,
        'tokenizer_vocab': tokenizer.get_vocab(),
        'training_stats': training_stats
    }
    torch.save(final_checkpoint, os.path.join(config.save_dir, "final_model.pt"))
    print(f"💾 Final model saved to: {os.path.join(config.save_dir, 'final_model.pt')}")
    
    return training_stats


def create_dataloader_from_pt(data_file: str, batch_size: int = 32, max_len: int = 120, shuffle: bool = False):
    """从.pt文件创建dataloader，用于测试集评估"""
    tokenizer = SelfiesTokenizer(max_len=max_len)
    dataset = RealSpectrumSelfiesDataset(data_file, tokenizer, max_len)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    ), tokenizer


def load_pretrained_model(checkpoint_path: str, device: str = "cuda") -> Tuple[MassSelfiesED, PretrainConfig]:
    """加载预训练模型"""
    print(f"Loading model from {checkpoint_path}")
    
    # Import required classes and modules
    import torch.serialization
    import sys
    from lzero.model.muzero_transformer import MassSelfiesED, SelfiesTokenizer
    from lzero.model.pretrain_transformer import PretrainConfig
    
    # Add a temporary module alias to handle the module path issue
    import lzero.model.pretrain_transformer as pretrain_transformer_module
    if 'pretrain_transformer' not in sys.modules:
        sys.modules['pretrain_transformer'] = pretrain_transformer_module
    
    try:
        # Use safe_globals context manager to allow PretrainConfig to be unpickled
        # Try both possible module paths since the checkpoint might have been saved with different paths
        with torch.serialization.safe_globals([PretrainConfig, 'lzero.model.pretrain_transformer.PretrainConfig', 'pretrain_transformer.PretrainConfig']):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Safe loading with context manager failed: {e}")
        try:
            # Alternative: Add safe globals permanently and then load
            torch.serialization.add_safe_globals([PretrainConfig])
            # Also try to add the string paths that might be in the checkpoint
            try:
                torch.serialization.add_safe_globals(['lzero.model.pretrain_transformer.PretrainConfig'])
                torch.serialization.add_safe_globals(['pretrain_transformer.PretrainConfig'])
            except:
                pass  # These might fail but that's ok
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e2:
            print(f"Safe loading with add_safe_globals failed: {e2}")
            try:
                # Last resort: use the old unsafe method with explicit weights_only=False
                print("Using unsafe loading method as last resort...")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except Exception as e3:
                print(f"All loading methods failed. Final error: {e3}")
                raise RuntimeError(f"Cannot load checkpoint from {checkpoint_path}. All loading methods failed.") from e3
    finally:
        # Clean up the temporary module alias
        if 'pretrain_transformer' in sys.modules and sys.modules['pretrain_transformer'] is pretrain_transformer_module:
            del sys.modules['pretrain_transformer']
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint file does not contain config")
    
    config = checkpoint['config']
    
    # 重新创建tokenizer以获取vocab_size
    tokenizer = SelfiesTokenizer(max_len=config.max_len)
    config.vocab_size = len(tokenizer.get_vocab())
    
    # 创建模型
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


if __name__ == "__main__":
    # 配置参数
    config = PretrainConfig(
        batch_size=16,
        learning_rate=5e-5,
        num_epochs=100,
        max_len=100,
        d_model=128,
        save_dir="./pretrained_selfies_transformer"
    )
    
    # 开始预训练
    pretrain_transformer(config)
    print("Pretraining completed!") 