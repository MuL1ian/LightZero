"""
Centralized configuration file for pretraining

This module contains all configuration classes to avoid duplication across files.
"""

from dataclasses import dataclass
from typing import Optional
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
    
    
    enable_spectrum_encoder: bool = False  # False = 纯baseline模式，直接使用预处理的4096维embeddings
    spectrum_encoder_checkpoint: Optional[str] = None  # spectrum encoder checkpoint路径（当enable_spectrum_encoder=True时使用）
    
    # Training parameters 
    batch_size: int = 64         #
    learning_rate: float = 1e-4   
    num_epochs: int = 70
    warmup_steps: int = 330    
    gradient_clip: float = 1.0    
    weight_decay: float = 0.01    #
    
    # Value head training parameters
    train_value_head: bool = False              # 改为False - 只训练auto-regressive baseline
    value_loss_weight: float = 0.0             # 设为0.0 - 不使用value loss
    
    # Value warmup parameters (当train_value_head=False时这些参数无效)
    enable_value_warmup: bool = False           # 禁用value warmup
    value_warmup_steps: int = 2000             # number of steps to warmup value training (policy trains alone first)
    
    # Value training strategy parameters (当train_value_head=False时这些参数无效)
    value_training_strategy: str = "teacher_forcing"  # "teacher_forcing" or "corrupted_sequences"
    add_noise_to_value_labels: bool = False      
    value_noise_prob: float = 0.0              
    
    # Early stopping - 
    early_stopping_patience: int = 15          
    early_stopping_min_delta: float = 0.005    
    save_best_model: bool = True
    
    # Data parameters (spectrum fingerprint)
    spectrum_dim: int = 4096
    
    # Save parameters - 
    save_dir: str = "./pretrained_selfies_transformer"
    log_interval: int = 100       
    save_interval: int = 350    
    
    # Device
    device: str = "cuda"
    
    # Dataset parameters
    train_data_file: str = "/hy-tmp/MassEnv/DataLoader/train_spectrum_embeds_msg.pt"
    val_data_file: str = "/hy-tmp/MassEnv/DataLoader/val_spectrum_embeds_msg.pt"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    batch_size: int = 32
    max_len: int = 120
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    k_predictions: int = 1
    temperature: float = 0.0
    test_data_file: str = "/hy-tmp/MassEnv/DataLoader/test_spectrum_embeds_msg.pt"
    results_dir: str = "./evaluation_results" 
