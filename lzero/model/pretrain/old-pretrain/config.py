"""
Centralized configuration file for pretraining

This module contains all configuration classes to avoid duplication across files.
"""

from dataclasses import dataclass
import torch


@dataclass
class PretrainConfig:
    """Pretraining configuration"""
    # Model parameters
    vocab_size: int = 100
    max_len: int = 120  
    d_model: int = 512             # Model dimension
    n_dec: int = 8                  # Number of decoder layers (decoder-only architecture)
    n_head: int = 16               # Number of attention heads
    spectrum_chunk_size: int = 256   # Each spectrum chunk size
    spectrum_attention_heads: int = 4 # attention heads for spectrum decomposer  
    #atten dim of spectrum decomposer = spectrum_chunk_size / spectrum_attention_heads suggest  > 64
    dropout: float = 0.15  
    
    # Training parameters 
    batch_size: int = 128        
    learning_rate: float = 4e-5  
    num_epochs: int = 12
    warmup_steps: int = 1500      
    gradient_clip: float = 1.0    
    weight_decay: float = 0.02    
    
    # Value head training parameters
    # i dont train value for this config
    train_value_head: bool = False              # whether to train value head
    value_loss_weight: float = 0.5             # weight for value loss
    
    # Sequence corruption parameters for value head training
    corruption_prob: float = 0.5               # corruption probability for each sequence
    corruption_ratio: float = 0.2              # corruption ratio for each sequence
    
    # Early stopping
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.01 
    save_best_model: bool = True
    
    # Data parameters (spectrum fingerprint)
    spectrum_dim: int = 4096
    
    # Save parameters
    save_dir: str = "./pretrained_model-onlyaction-head"
    log_interval: int = 200
    save_interval: int = 1000
    
    # Device
    device: str = "cuda"
    
    # Dataset parameters
    train_data_file: str = "/hy-tmp/MassEnv/DataLoader/train_spectrum_embeds_msg.pt"
    val_data_file: str = "/hy-tmp/MassEnv/DataLoader/val_spectrum_embeds_msg.pt"

