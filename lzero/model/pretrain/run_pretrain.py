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
import selfies as sf
import csv
import argparse
from lzero.model.muzero_transformer import MassSelfiesED,SelfiesTokenizer
from config import PretrainConfig
# import selfies as sf


class RealSpectrumSelfiesDataset(Dataset):
    
    def __init__(self, data_file: str, tokenizer: SelfiesTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_file = data_file
        
        print(f"Loading real spectrum data from: {data_file}")
        self.data = torch.load(data_file, map_location='cpu')
        
        required_keys = ['embeds', 'smiles', 'formulas','selfies']
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"Missing key '{key}' in dataset file: {data_file}")
        
        self.selfies_list = []
        valid_indices = []
        tokenizer_failed = 0
        
        # Use pre-computed SELFIES directly, only validate tokenizer compatibility
        for i, selfies in enumerate(tqdm(self.data['selfies'], desc="Validating SELFIES")):
            try:
                if selfies and len(selfies) > 0: 
                    # validate if tokenizer can handle this SELFIES
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
                tokenizer_failed += 1
                if tokenizer_failed <= 5:
                    print(f"[WARN] SELFIES processing error: {e}")
                continue
        
        if tokenizer_failed > 0:
            print(f"[INFO] Tokenizer compatibility check: {tokenizer_failed} SELFIES failed validation")
        
        # filter valid data
        self.embeds = self.data['embeds'][valid_indices]
        self.formulas = [self.data['formulas'][i] for i in valid_indices]
        
        # Optional: store token counts if available for potential future use
        if 'token_counts' in self.data:
            self.token_counts = [self.data['token_counts'][i] for i in valid_indices]
        else:
            self.token_counts = None
        
        print(f"Loaded {len(self.selfies_list)} valid samples from {len(self.data['smiles'])} total samples")
        print(f"Spectrum embeddings shape: {self.embeds.shape}")
        
    def __len__(self):
        return len(self.selfies_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spectrum = self.embeds[idx].float()  # [4096]
        selfies = self.selfies_list[idx]
        
        try:
            encoding = self.tokenizer.encode(
                        list(sf.split_selfies(selfies)),
                        is_pretokenized=True,
                        add_special_tokens=True
                    )
                    
            full_token_ids = torch.tensor(encoding.ids, dtype=torch.long)
            attention_mask = torch.tensor(encoding.attention_mask, dtype=torch.float)

        except Exception as e:
            print(f"Warning: Failed to encode SELFIES '{selfies}': {e}")
            full_token_ids = torch.full((self.max_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(self.max_len, dtype=torch.float)
        

        input_ids = full_token_ids[:-1]
        target_ids = full_token_ids[1:]
        attention_mask = attention_mask[:-1]

        if len(input_ids) != self.max_len - 1:
            print(f"Error: length mismatch. Expected {self.max_len - 1}, got {len(input_ids)}")
            # 进行填充
            pad_len = (self.max_len - 1) - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.float)])
            
        
        return {
            'spectrum': spectrum,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'formula': self.formulas[idx],
            'selfies': selfies
        }


class PretrainLoss(nn.Module):
    """Combined loss function for pretrain: action prediction + value prediction"""
    
    def __init__(self, vocab_size: int, pad_token_id: int, label_smoothing: float = 0.1,
                 value_loss_weight: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.value_loss_weight = value_loss_weight
        
        # Language modeling loss
        self.lm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing
        )
        
        # Value prediction loss (binary classification)
        self.value_loss_fn = nn.BCEWithLogitsLoss(reduction='none') 
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                values: torch.Tensor = None, value_labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size) - action logits
            targets: (batch_size, seq_len) - target token ids
            values: (batch_size, seq_len, 1) - value predictions (optional)
            value_labels: (batch_size, seq_len) - binary value labels (optional)
            
        Returns:
            Dict containing 'total_loss', 'lm_loss', 'value_loss'
        """
        # Language modeling loss
        lm_logits = logits.view(-1, self.vocab_size)
        lm_targets = targets.view(-1)
        lm_loss = self.lm_loss_fn(lm_logits, lm_targets)
        
        losses = {
            'lm_loss': lm_loss,
            'total_loss': lm_loss
        }
        
        # not check 
        if values is not None and value_labels is not None:
            value_mask = (targets != self.pad_token_id).float()  # (B, T)
            
            values_flat = values.squeeze(-1)  # (B, T)
            value_loss_per_token = self.value_loss_fn(values_flat, value_labels)  # (B, T)
            
            masked_value_loss = value_loss_per_token * value_mask
            
            # Safe division to avoid division by zero
            total_valid_value_tokens = value_mask.sum()
            if total_valid_value_tokens > 0:
                value_loss = masked_value_loss.sum() / total_valid_value_tokens
            else:
                value_loss = torch.tensor(0.0, device=masked_value_loss.device)

            # total loss
            total_loss = lm_loss + self.value_loss_weight * value_loss
            
            losses.update({
                'value_loss': value_loss,
                'total_loss': total_loss
            })
        
        return losses


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
        d_model=config.d_model, 
        n_dec=config.n_dec,
        n_head=config.n_head,
        spectrum_chunk_size=config.spectrum_chunk_size,
        spectrum_attention_heads=config.spectrum_attention_heads,
        dropout=config.dropout,
        device=config.device,
        enable_spectrum_encoder=config.enable_spectrum_encoder,
        spectrum_encoder_checkpoint=config.spectrum_encoder_checkpoint
    )

# not check 
def generate_teacher_forcing_value_labels(
    model: MassSelfiesED,
    spectrum: torch.Tensor,
    input_ids: torch.Tensor, 
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    add_noise: bool = False,
    noise_prob: float = 0.1
) -> torch.Tensor:
    """
    Generate value labels based on model's teacher forcing accuracy.
    Value labels indicate whether the model can correctly predict subsequent tokens.
    
    Args:
        model: The transformer model
        spectrum: (B, spectrum_dim) spectrum embeddings
        input_ids: (B, T) input sequence [SOS, token1, token2, ...]
        target_ids: (B, T) target sequence [token1, token2, ..., EOS]  
        attention_mask: (B, T) attention mask
        tokenizer: SELFIES tokenizer
        add_noise: whether to add noise to value labels for robustness
        noise_prob: probability of flipping correct predictions to incorrect
        
    Returns:
        value_labels: (B, T) - 1 if can generate correctly from this position, 0 otherwise
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Get model predictions for the GT sequence
    with torch.no_grad():
        logits = model.forward_pretrain(spectrum, input_ids, attention_mask, return_value=False)
        if isinstance(logits, tuple):
            logits = logits[0]
    
    # Get predicted tokens
    predicted_tokens = torch.argmax(logits, dim=-1)  # (B, T)
    
    # Initialize value labels
    value_labels = torch.zeros(batch_size, seq_len, dtype=torch.float, device=device)
    pad_mask = (target_ids != tokenizer.pad_token_id).float()
    
    # For each sequence, compute cumulative correctness from right to left
    for b in range(batch_size):
        # Find the last non-padding position
        valid_positions = torch.where(target_ids[b] != tokenizer.pad_token_id)[0]
        if len(valid_positions) == 0:
            continue
            
        # Start from the right (end of sequence) and work backwards
        cumulative_correct = True
        for pos in range(len(valid_positions) - 1, -1, -1):
            actual_pos = valid_positions[pos].item()
            
            # Check if prediction at this position is correct
            if predicted_tokens[b, actual_pos] == target_ids[b, actual_pos]:
                # If we're still cumulative correct, this position gets 1
                if cumulative_correct:
                    value_labels[b, actual_pos] = 1.0
                # else it stays 0 (already set)
            else:
                # Prediction is wrong, so cumulative correctness breaks
                cumulative_correct = False
                # This position and all previous positions get 0 (already set)
    
    # Add noise for robustness (randomly flip some correct predictions to incorrect)
    if add_noise and noise_prob > 0:
        noise_mask = torch.rand_like(value_labels) < noise_prob
        # Only flip 1s to 0s (correct to incorrect), not the other way around
        flip_mask = noise_mask & (value_labels == 1.0) & (pad_mask == 1.0)
        value_labels[flip_mask] = 0.0
    
    # Apply padding mask
    value_labels = value_labels * pad_mask
    
    return value_labels

def pretrain_step(model: MassSelfiesED, batch: Dict[str, torch.Tensor], 
                 loss_fn: PretrainLoss, device: str, tokenizer: SelfiesTokenizer = None,
                 train_value_head: bool = True, global_step: int = 0,
                 add_noise_to_value_labels: bool = False, value_noise_prob: float = 0.1,
                 value_warmup_steps: int = 3000, enable_value_warmup: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:


    spectrum = batch['spectrum'].to(device)
    input_ids = batch['input_ids'].to(device)
    target_ids = batch['target_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Value warmup logic: train only policy for first value_warmup_steps
    should_train_value = train_value_head
    if enable_value_warmup and global_step < value_warmup_steps:
        should_train_value = False
    
    # Single forward pass with GT sequences for both policy and value training
    if should_train_value:
        # Forward pass that returns both policy logits and value predictions
        policy_logits, value_predictions = model.forward_pretrain(
            spectrum, input_ids, attention_mask, return_value=True
        )
        
        # Generate value labels based on model's teacher forcing accuracy
        token_value_labels = generate_teacher_forcing_value_labels(
            model, spectrum, input_ids, target_ids, attention_mask, tokenizer,
            add_noise=add_noise_to_value_labels, noise_prob=value_noise_prob
        )
    else:
        # Only policy training
        policy_logits = model.forward_pretrain(spectrum, input_ids, attention_mask, return_value=False)
        value_predictions = None
        token_value_labels = None
    
    # Both policy and value use GT targets for consistency
    policy_target_ids = target_ids
    value_target_ids = target_ids
    
    # 1. Calculate Policy Loss (always on GT targets)
    policy_loss = loss_fn.lm_loss_fn(
        policy_logits.view(-1, policy_logits.size(-1)), 
        policy_target_ids.view(-1)
    )
    
    # Policy accuracy (based on GT targets)
    policy_predictions = torch.argmax(policy_logits, dim=-1)  # (B, T)
    policy_mask = (policy_target_ids != loss_fn.pad_token_id)
    policy_correct = (policy_predictions == policy_target_ids) & policy_mask
    
    # Safe division to avoid division by zero
    total_valid_tokens = policy_mask.sum().float()
    if total_valid_tokens > 0:
        policy_accuracy = policy_correct.sum().float() / total_valid_tokens
    else:
        policy_accuracy = torch.tensor(0.0, device=device)
    
    # 2. Calculate Value Loss (if training value head)
    value_loss = torch.tensor(0.0, device=device)
    value_accuracy = 0.0
    avg_value_pred = 0.0
    
    if should_train_value and value_predictions is not None:
        # Use the unified loss function for value loss calculation
        value_loss_dict = loss_fn(
            logits=torch.zeros_like(policy_logits),  # dummy logits, only compute value loss
            targets=value_target_ids,
            values=value_predictions,
            value_labels=token_value_labels
        )
        value_loss = value_loss_dict.get('value_loss', torch.tensor(0.0))
        
        # Value metrics with safe division
        with torch.no_grad():
            value_preds = torch.sigmoid(value_predictions.squeeze(-1))  # (B, T)
            value_mask = (value_target_ids != loss_fn.pad_token_id).float()
            
            # Calculate accuracy at token level with safe division
            value_pred_binary = (value_preds > 0.5).float()
            value_correct = (value_pred_binary == token_value_labels) * value_mask
            
            total_value_tokens = value_mask.sum()
            if total_value_tokens > 0:
                value_accuracy = (value_correct.sum() / total_value_tokens).item()
                avg_value_pred = (value_preds * value_mask).sum().item() / total_value_tokens.item()
            else:
                value_accuracy = 0.0
                avg_value_pred = 0.0
    
    # 3. Combine losses for single optimization step
    total_loss = policy_loss + loss_fn.value_loss_weight * value_loss
    
    # EOS statistics (based on Policy predictions on GT targets)
    eos_stats = {}
    if tokenizer is not None:
        target_has_eos = (policy_target_ids == tokenizer.eos_token_id).any(dim=1)
        target_eos_count = target_has_eos.sum().item()
        
        pred_has_eos = (policy_predictions == tokenizer.eos_token_id).any(dim=1)
        pred_eos_count = pred_has_eos.sum().item()
        
        batch_size = policy_target_ids.size(0)
        eos_stats = {
            'target_eos_rate': target_eos_count / batch_size * 100,  
            'pred_eos_rate': pred_eos_count / batch_size * 100,     
            'eos_accuracy': (target_has_eos == pred_has_eos).sum().item() / batch_size * 100  
        }

    perplexity = torch.exp(policy_loss)
    
    metrics = {
        'loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'accuracy': policy_accuracy.item(),
        'value_accuracy': value_accuracy,
        'avg_value_pred': avg_value_pred,
        'perplexity': perplexity.item(),
        **eos_stats  
    }
    
    return total_loss, metrics


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
    if config.train_value_head:
        print(f"🎯 Value head training enabled - using real generation evaluation")
    
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
    
    # create loss function with label smoothing and value loss
    loss_fn = PretrainLoss(
        config.vocab_size, 
        tokenizer.pad_token_id, 
        label_smoothing=0.05,
        value_loss_weight=config.value_loss_weight if config.train_value_head else 0.0
    )
    
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
    current_val_loss = None  
    
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
    
    print(f"\n📚 Training mode: Combined Policy + Value Training")
    print(f"    Policy Head: GT sequences → Action prediction") 
    print(f"    Value Head: GT sequences → Teacher forcing accuracy prediction")
    print(f"    Single optimization step with combined loss")
    
    if config.train_value_head:
        print(f"🎯 Value head training enabled (weight: {config.value_loss_weight})")
        print(f"🎯 Value training strategy: {config.value_training_strategy}")
        if config.add_noise_to_value_labels:
            print(f"🎲 Value label noise enabled: prob={config.value_noise_prob}")
        print(f"💡 Value predicts: 'Can generate correctly from this position'")
        
        # Value warmup information
        if config.enable_value_warmup:
            print(f"🔥 Value warmup enabled: {config.value_warmup_steps} steps")
            print(f"    First {config.value_warmup_steps} steps: Policy only")
            print(f"    After step {config.value_warmup_steps}: Policy + Value")
        else:
            print(f"🔥 Value warmup disabled: Training value from step 0")
    else:
        print(f"📝 Language modeling only")
    
    for epoch in range(config.num_epochs):
        epoch_losses = []
        epoch_metrics = {
            'loss': [], 'policy_loss': [], 'value_loss': [], 'accuracy': [], 'perplexity': [], 
            'value_accuracy': [], 'avg_value_pred': [],
            'target_eos_rate': [], 'pred_eos_rate': [], 'eos_accuracy': []
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            loss, metrics = pretrain_step(model, batch, loss_fn, config.device, tokenizer, 
                                        train_value_head=config.train_value_head, global_step=global_step,
                                        add_noise_to_value_labels=config.add_noise_to_value_labels, 
                                        value_noise_prob=config.value_noise_prob,
                                        value_warmup_steps=config.value_warmup_steps,
                                        enable_value_warmup=config.enable_value_warmup)
            stage_info = "TEACHER_FORCE"
            
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
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
            
            # update progress bar
            # Determine training mode based on warmup status
            is_in_warmup = config.enable_value_warmup and global_step < config.value_warmup_steps
            training_mode = 'P_WARMUP' if is_in_warmup else 'P+V_TRAIN'
            
            postfix_dict = {
                'mode': training_mode,
                'loss': f"{loss.item():.4f}",
                'p_loss': f"{metrics['policy_loss']:.4f}",  # policy loss
                'acc': f"{metrics['accuracy']:.4f}",
                'ppl': f"{metrics['perplexity']:.2f}",
                'lr': f"{scheduler.get_lr():.2e}"
            }
            
            # add value metrics if available (only when not in warmup)
            if not is_in_warmup and 'value_loss' in metrics and metrics['value_loss'] > 0:
                postfix_dict['v_loss'] = f"{metrics['value_loss']:.4f}"
                postfix_dict['v_acc'] = f"{metrics.get('value_accuracy', 0):.3f}"
            
            # add eos stats to progress bar
            if 'pred_eos_rate' in metrics:
                postfix_dict['eos%'] = f"{metrics['pred_eos_rate']:.1f}"
                
            pbar.set_postfix(postfix_dict)
            
            global_step += 1
            
            # Check if we just finished warmup
            if config.enable_value_warmup and global_step == config.value_warmup_steps:
                print(f"\n🎯 Value warmup completed at step {global_step}! Starting value head training...")
            
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
                
                # Add value warmup progress info
                warmup_info = ""
                if config.enable_value_warmup and config.train_value_head:
                    if global_step < config.value_warmup_steps:
                        progress = (global_step / config.value_warmup_steps) * 100
                        remaining_steps = config.value_warmup_steps - global_step
                        warmup_info = f" [WARMUP: {progress:.1f}%, {remaining_steps} steps remain]"
                    else:
                        warmup_info = " [VALUE_TRAINING_ACTIVE]"
                
                if log_training_step_fn:
                    log_training_step_fn(
                        epoch=epoch + 1,
                        step=batch_idx + 1,
                        global_step=global_step,
                        train_loss=avg_recent_loss,
                        learning_rate=current_lr,
                        early_stopping_counter=patience_counter
                    )
                
                print(f"📈 Step {global_step}: loss={avg_recent_loss:.4f}, lr={current_lr:.2e}{eos_log_str}{warmup_info}")
            
            # save checkpoint and validation
            if global_step % config.save_interval == 0:
                # Run validation before saving checkpoint
                if val_loader:
                    model.eval()
                    val_losses = []
                    val_metrics = {
                        'loss': [], 'policy_loss': [], 'value_loss': [], 'accuracy': [], 'perplexity': [], 
                        'value_accuracy': [], 'avg_value_pred': [],
                        'target_eos_rate': [], 'pred_eos_rate': [], 'eos_accuracy': []
                    }
                    
                    print(f"🔍 Running validation at step {global_step}...")
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc="Validation", leave=False):
                            loss, metrics = pretrain_step(model, batch, loss_fn, config.device, tokenizer,
                                                         train_value_head=config.train_value_head, global_step=global_step,
                                                         add_noise_to_value_labels=config.add_noise_to_value_labels, 
                                                         value_noise_prob=config.value_noise_prob,
                                                         value_warmup_steps=config.value_warmup_steps,
                                                         enable_value_warmup=config.enable_value_warmup)
                            val_losses.append(loss.item())
                            for key, value in metrics.items():
                                if key in val_metrics:
                                    val_metrics[key].append(value)
                    
                    avg_val_metrics = {}
                    for k, v in val_metrics.items():
                        if len(v) > 0:
                            v_clean = []
                            for val in v:
                                if isinstance(val, torch.Tensor):
                                    v_clean.append(val.item())
                                else:
                                    v_clean.append(val)
                            avg_val_metrics[k] = np.mean(v_clean)
                        else:
                            avg_val_metrics[k] = 0.0
                    current_val_loss = avg_val_metrics['loss']
                    val_perplexity = avg_val_metrics['perplexity']
                    
                    training_stats['val_losses'].append(current_val_loss)
                    
                    # Determine warmup status for validation display
                    val_warmup_status = ""
                    if config.enable_value_warmup and config.train_value_head:
                        if global_step < config.value_warmup_steps:
                            progress = (global_step / config.value_warmup_steps) * 100
                            val_warmup_status = f" [Policy-only warmup: {progress:.1f}%]"
                        else:
                            val_warmup_status = " [Policy + Value training]"
                    
                    print(f"📊 Validation Results - Step {global_step}{val_warmup_status}:")
                    print(f"    Loss: {current_val_loss:.4f}")
                    print(f"    Accuracy: {avg_val_metrics['accuracy']:.4f}")
                    print(f"    Perplexity: {val_perplexity:.2f}")
                    
                    # Only show value metrics when value training is active
                    if not (config.enable_value_warmup and global_step < config.value_warmup_steps):
                        if 'value_loss' in avg_val_metrics and avg_val_metrics['value_loss'] > 0:
                            print(f"    Value Loss: {avg_val_metrics['value_loss']:.4f}")
                            print(f"    Value Accuracy: {avg_val_metrics['value_accuracy']:.4f}")
                    
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
                                'training_stats': training_stats,
                                'value_warmup_completed': global_step >= config.value_warmup_steps if config.enable_value_warmup else True
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