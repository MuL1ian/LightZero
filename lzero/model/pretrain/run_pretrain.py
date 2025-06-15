import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.serialization
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
import os
from tqdm import tqdm
import wandb
from dataclasses import dataclass
import selfies as sf
import csv
import argparse
import time

from lzero.model.muzero_transformer import MassSelfiesED, SelfiesTokenizer
from lzero.model.muzero_transformer import get_actions_list
from config import PretrainConfig


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
        tokenizer_failed = 0
        
        for i, smiles in enumerate(tqdm(self.data['smiles'], desc="Converting SMILES")):
            try:
                selfies = sf.encoder(smiles)
                if selfies and len(selfies) > 0: 
                    # extra check: validate if tokenizer can handle this SELFIES
                    try:
                        # test encoding
                        encoded_tokens = self.tokenizer.encode_selfies(selfies, add_special_tokens=False)
                        # test decoding
                        decoded_selfies = self.tokenizer.decode_to_selfies(encoded_tokens, skip_special_tokens=True)
                        
                        if decoded_selfies.replace(" ", "") == selfies.replace(" ", ""):
                            self.selfies_list.append(selfies)
                            valid_indices.append(i)
                        else:
                            tokenizer_failed += 1
                            if tokenizer_failed <= 5: 
                                if len(decoded_selfies) < 120:
                                    print(f"[WARN] Tokenizer round-trip failed for SELFIES: '{selfies}' -> '{decoded_selfies}'")
                    except Exception as tokenizer_error:
                        tokenizer_failed += 1
                        if tokenizer_failed <= 5:  
                            print(f"[WARN] Tokenizer failed for SELFIES '{selfies}': {tokenizer_error}")
            except Exception as e:
                continue
        
        if tokenizer_failed > 0:
            print(f"[INFO] Tokenizer compatibility check: {tokenizer_failed} SELFIES failed validation (over long SELFIES)")
        
        # filter valid data
        self.embeds = self.data['embeds'][valid_indices]
        self.formulas = [self.data['formulas'][i] for i in valid_indices]
        
        print(f"Loaded {len(self.selfies_list)} valid samples from {len(self.data['smiles'])} total samples")
        print(f"Spectrum embeddings shape: {self.embeds.shape}")
        
    def __len__(self):
        return len(self.selfies_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spectrum = self.embeds[idx].float()  # [4096]
        
        selfies = self.selfies_list[idx]
        
        # encode selfies to token ids
        tokens = [self.tokenizer.sos_token_id]
        try:
            # 🔧 关键修复：获取真实的SELFIES token，不要填充的
            import selfies as sf
            selfies_token_list = list(sf.split_selfies(selfies))
            
            # 手动编码每个SELFIES token
            selfies_tokens = []
            vocab = self.tokenizer.get_vocab()
            for token in selfies_token_list:
                if token in vocab:
                    selfies_tokens.append(vocab[token])
                else:
                    # 如果token不在词汇表中，跳过或使用UNK
                    print(f"Warning: Unknown SELFIES token: {token}")
                    continue
            
            # 预留空间给EOS token，确保不被截断
            max_selfies_len = self.max_len - 2  # 预留SOS和EOS的空间
            if len(selfies_tokens) > max_selfies_len:
                selfies_tokens = selfies_tokens[:max_selfies_len]
            
            tokens.extend(selfies_tokens)
            tokens.append(self.tokenizer.eos_token_id)
        except Exception as e:
            # if encoding failed, use empty sequence
            print(f"Warning: Failed to encode SELFIES '{selfies}': {e}")
            tokens = [self.tokenizer.sos_token_id, self.tokenizer.eos_token_id]
        
        # 🔧 关键修复：不要填充！保持原始长度让EOS位置多样化
        # tokens现在是: [SOS, selfies_tokens..., EOS] (长度可变)
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)  # [SOS, selfies_tokens...]
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)   # [selfies_tokens..., EOS]
        
        # 填充input_ids和target_ids到max_len-1 (因为模型期望固定长度)
        max_seq_len = self.max_len - 1  # 119
        
        if len(input_ids) < max_seq_len:
            pad_len = max_seq_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
        
        if len(target_ids) < max_seq_len:
            pad_len = max_seq_len - len(target_ids)
            target_ids = torch.cat([target_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            'spectrum': spectrum,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'formula': self.formulas[idx],  # for debugging
            'selfies': selfies  # for debugging
        }


class PretrainLoss(nn.Module):
    """loss function for pretrain"""
    
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
            # warmup phase: linear growth
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # decay phase: linear decay to min lr
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr - (self.base_lr - self.min_lr) * min(progress, 1.0)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


def create_model(config: PretrainConfig) -> MassSelfiesED:
    """create model"""
    return MassSelfiesED(
        vocab_size=config.vocab_size,
        max_len=config.max_len,
        d_model=config.d_model,  # Use d_model to match both config and MassSelfiesED class
        n_dec=config.n_dec,
        n_head=config.n_head,
        spectrum_chunk_size=config.spectrum_chunk_size,
        spectrum_attention_heads=config.spectrum_attention_heads,
        dropout=config.dropout,
        device=config.device
    )


def pretrain_step(model: MassSelfiesED, batch: Dict[str, torch.Tensor], 
                 loss_fn: PretrainLoss, device: str, tokenizer: SelfiesTokenizer = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """single step pretrain"""

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
    
    eos_stats = {}
    if tokenizer is not None:
        # count eos in target
        target_has_eos = (target_ids == tokenizer.eos_token_id).any(dim=1)
        target_eos_count = target_has_eos.sum().item()
        
        # count eos in prediction
        pred_has_eos = (predictions == tokenizer.eos_token_id).any(dim=1)
        pred_eos_count = pred_has_eos.sum().item()
        
        batch_size = target_ids.size(0)
        eos_stats = {
            'target_eos_rate': target_eos_count / batch_size * 100,  
            'pred_eos_rate': pred_eos_count / batch_size * 100,     
            'eos_accuracy': (target_has_eos == pred_has_eos).sum().item() / batch_size * 100  
        }

    perplexity = torch.exp(loss)
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'perplexity': perplexity.item(),
        **eos_stats  
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
    
    # decode tokens to see the result
    input_tokens = sample['input_ids'].tolist()
    target_tokens = sample['target_ids'].tolist()
    
    print(f"🔤 Input tokens (first 20): {input_tokens[:20]}")
    print(f"🔤 Target tokens (first 20): {target_tokens[:20]}")
    
    # try decoding
    try:
        decoded_input = tokenizer.decode_to_selfies(input_tokens, skip_special_tokens=True)
        decoded_target = tokenizer.decode_to_selfies(target_tokens, skip_special_tokens=True)
        print(f"🔤 Decoded input: {decoded_input}")
        print(f"🔤 Decoded target: {decoded_target}")
    except Exception as e:
        print(f"🔤 Decoding failed: {e}")
    
    print(f"{'='*60}\n")


def pretrain_transformer(config: PretrainConfig):
    """main pretrain function"""
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
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if config.device == "cuda" else False
        )
    
    # create model
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = WarmupLinearSchedule(optimizer, config.warmup_steps, total_steps)
    
    # create loss function with label smoothing
    loss_fn = PretrainLoss(config.vocab_size, tokenizer.pad_token_id, label_smoothing=0.05)
    
    print(f"🔍 Loss function setup:")
    print(f"  PAD token ID (ignored): {tokenizer.pad_token_id}")
    print(f"  EOS token ID (counted): {tokenizer.eos_token_id}")
    print(f"  SOS token ID (counted): {tokenizer.sos_token_id}")
    
    # training loop variables
    model.train()
    global_step = 0
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # training statistics
    training_stats = {
        'val_losses': [],
        'train_losses': [],
        'start_time': time.time(),
        'early_stopped': False
    }
    
    # set logging function
    log_training_step_fn = None
    if hasattr(config, 'enable_detailed_logging') and config.enable_detailed_logging:
        if hasattr(config, 'log_files') and config.log_files:
            from run_pretrain import log_training_step
            log_training_step_fn = lambda **kwargs: log_training_step(config.log_files['train_log'], **kwargs)
            print("📊 Detailed logging enabled")
        else:
            print("⚠️ Detailed logging requested but log files not found")
    
    # simple CSV logging
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
        epoch_metrics = {'loss': [], 'accuracy': [], 'perplexity': [], 'target_eos_rate': [], 'pred_eos_rate': [], 'eos_accuracy': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # forward pass
            loss, metrics = pretrain_step(model, batch, loss_fn, config.device, tokenizer)
            
            # backward pass
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # optimizer step
            optimizer.step()
            scheduler.step()
            
            # record metrics
            epoch_losses.append(loss.item())
            training_stats['train_losses'].append(loss.item())
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # update progress bar
            postfix_dict = {
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'ppl': f"{metrics['perplexity']:.2f}",
                'lr': f"{scheduler.get_lr():.2e}"
            }
            # add eos stats to progress bar
            if 'pred_eos_rate' in metrics:
                postfix_dict['eos%'] = f"{metrics['pred_eos_rate']:.1f}"
            pbar.set_postfix(postfix_dict)
            
            global_step += 1
            
            if global_step % config.log_interval == 0:
                current_lr = scheduler.get_lr()
                avg_recent_loss = np.mean(training_stats['train_losses'][-config.log_interval:])
                
                recent_metrics = []
                for i in range(max(0, len(epoch_metrics['accuracy']) - config.log_interval), len(epoch_metrics['accuracy'])):
                    if i < len(epoch_metrics.get('pred_eos_rate', [])):
                        recent_metrics.append({
                            'pred_eos_rate': epoch_metrics['pred_eos_rate'][i],
                            'target_eos_rate': epoch_metrics['target_eos_rate'][i],
                            'eos_accuracy': epoch_metrics['eos_accuracy'][i]
                        })
                
                eos_log_str = ""
                if recent_metrics:
                    avg_pred_eos = np.mean([m['pred_eos_rate'] for m in recent_metrics])
                    avg_target_eos = np.mean([m['target_eos_rate'] for m in recent_metrics])
                    avg_eos_acc = np.mean([m['eos_accuracy'] for m in recent_metrics])
                    eos_log_str = f", pred_eos={avg_pred_eos:.1f}%, target_eos={avg_target_eos:.1f}%, eos_acc={avg_eos_acc:.1f}%"
                
                if log_training_step_fn:
                    log_training_step_fn(
                        epoch=epoch + 1,
                        step=batch_idx + 1,
                        global_step=global_step,
                        train_loss=avg_recent_loss,
                        learning_rate=current_lr,
                        early_stopping_counter=patience_counter
                    )
                
                print(f"📈 Step {global_step}: loss={avg_recent_loss:.4f}, lr={current_lr:.2e}{eos_log_str}")
            
            # save checkpoint and validation
            if global_step % config.save_interval == 0:
                # Run validation before saving checkpoint
                current_val_loss = None
                if val_loader:
                    model.eval()
                    val_losses = []
                    val_metrics = {'loss': [], 'accuracy': [], 'perplexity': [], 'target_eos_rate': [], 'pred_eos_rate': [], 'eos_accuracy': []}
                    
                    print(f"🔍 Running validation at step {global_step}...")
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc="Validation", leave=False):
                            loss, metrics = pretrain_step(model, batch, loss_fn, config.device, tokenizer)
                            val_losses.append(loss.item())
                            for key, value in metrics.items():
                                val_metrics[key].append(value)
                    
                    avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
                    current_val_loss = avg_val_metrics['loss']
                    val_perplexity = avg_val_metrics['perplexity']
                    
                    training_stats['val_losses'].append(current_val_loss)
                    
                    print(f"📊 Validation Results - Step {global_step}:")
                    print(f"    Loss: {current_val_loss:.4f}")
                    print(f"    Accuracy: {avg_val_metrics['accuracy']:.4f}")
                    print(f"    Perplexity: {val_perplexity:.2f}")
                    if 'pred_eos_rate' in avg_val_metrics:
                        print(f"    Target EOS Rate: {avg_val_metrics['target_eos_rate']:.1f}%")
                        print(f"    Predicted EOS Rate: {avg_val_metrics['pred_eos_rate']:.1f}%")
                        print(f"    EOS Accuracy: {avg_val_metrics['eos_accuracy']:.1f}%")

                    # CSV logging
                    if not csv_header_written:
                        with open(csv_log_path, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(['step', 'epoch', 'loss', 'accuracy', 'perplexity'])
                        csv_header_written = True
                    
                    with open(csv_log_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            global_step,
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
                                'global_step': global_step,
                                'val_loss': current_val_loss,
                                'tokenizer_vocab': tokenizer.get_vocab(),
                                'training_stats': training_stats
                            }
                            torch.save(best_checkpoint, os.path.join(config.save_dir, "best_model.pt"))
                            print(f"🌟 Saved new best model with val_loss: {current_val_loss:.4f}")
                    else:
                        patience_counter += 1
                        print(f"⏳ No improvement. Early stopping patience: {patience_counter}/{config.early_stopping_patience}")
                    
                    model.train()  # Switch back to training mode
                
                # Save regular checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'global_step': global_step,
                    'epoch': epoch + 1,
                    'val_loss': current_val_loss,
                    'tokenizer_vocab': tokenizer.get_vocab(),
                    'training_stats': training_stats
                }
                torch.save(checkpoint, os.path.join(config.save_dir, f"checkpoint_step_{global_step}.pt"))
                print(f"💾 Saved checkpoint at step {global_step}")
                
                # Check early stopping
                if val_loader and patience_counter >= config.early_stopping_patience:
                    print(f"🛑 Early stopping triggered at step {global_step}")
                    training_stats['early_stopped'] = True
                    break
        
        # Check if early stopping was triggered during the epoch
        if training_stats.get('early_stopped', False):
            break
    
    # training finished
    training_time = time.time() - training_stats['start_time']
    training_stats['training_time'] = f"{training_time:.2f}s ({training_time/60:.1f}m)"
    training_stats['total_epochs'] = epoch + 1
    training_stats['total_steps'] = global_step
    training_stats['final_train_loss'] = training_stats['train_losses'][-1] if training_stats['train_losses'] else None
    training_stats['best_val_loss'] = best_val_loss if current_val_loss is not None else None
    training_stats['best_epoch'] = None
    
    # find best epoch
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
    
    # save final model
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
    """create dataloader from .pt file, for test set evaluation"""
    tokenizer = SelfiesTokenizer(max_len=max_len)
    dataset = RealSpectrumSelfiesDataset(data_file, tokenizer, max_len)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    ), tokenizer


def load_pretrained_model(checkpoint_path: str, device: torch.device) -> Tuple[MassSelfiesED, PretrainConfig]:
    """load pretrained model"""
    
    print(f"Loading model from {checkpoint_path}")
    
    if isinstance(device, str):
        device = torch.device(device)
    
    print(f"🔧 Target device: {device}")
    
    try:
        with torch.serialization.safe_globals([PretrainConfig]):
            checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Safe loading failed, trying with weights_only=False: {e}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e2:
            print(f"Both loading methods failed. Error: {e2}")
            raise e2
    
    config = checkpoint['config']
    
    print(f"\n📋 Config 内容:")
    print(f"{'='*50}")
    if hasattr(config, '__dict__'):
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
    else:
        # 如果是dataclass，也可以这样打印
        import dataclasses
        if dataclasses.is_dataclass(config):
            for field in dataclasses.fields(config):
                value = getattr(config, field.name)
                print(f"  {field.name}: {value}")
        else:
            print(f"  Config type: {type(config)}")
            print(f"  Config: {config}")
    print(f"{'='*50}\n")
    
    config.device = str(device)
    
    tokenizer = SelfiesTokenizer(max_len=config.max_len)
    config.vocab_size = len(tokenizer.get_vocab())
    
    print(f"🔧 Creating model...")
    model = create_model(config)
    
    print(f"🔧 Loading model state dict...")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"🔧 Moving model to device: {device}")
    model = model.to(device)
    model.eval()
    
    # 验证模型确实在正确的设备上
    model_device = next(model.parameters()).device
    print(f"🔍 Model actual device after loading: {model_device}")
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain SELFIES Transformer")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint to load")
    parser.add_argument("--eval_on_test", action="store_true", help="Evaluate on test set")
    parser.add_argument("--test_generation", action="store_true", help="Test generation capabilities")
    parser.add_argument("--test_data_path", type=str, default="/hy-tmp/MassEnv/DataLoader/test_spectrum_embeds_msg.pt", 
                        help="Path to test data file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation (default: 32)")
    
    args = parser.parse_args()
    
    config = PretrainConfig(
        save_dir="./pretrained_selfies_transformer"
    )
    print(f"USING DEVICE: {config.device}")
        
    # start pretrain
    pretrain_transformer(config)
    print("Pretraining completed!") 