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
    d_model: int = 512               # Model dimension
    n_dec: int = 6                  # Number of decoder layers (decoder-only architecture)
    n_head: int = 16                 # Number of attention heads
    num_projectors: int = 16         # Number of linear decomposers for spectrum
    spectrum_chunk_size: int = 512   # Each spectrum chunk size
    dropout: float = 0.25 
    
    # Training parameters 
    batch_size: int = 64
    learning_rate: float = 1e-4  
    num_epochs: int = 30
    warmup_steps: int = 1000  
    gradient_clip: float = 0.5  
    weight_decay: float = 0.05 
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    save_best_model: bool = True
    
    # Data parameters (spectrum fingerprint)
    spectrum_dim: int = 4096
    
    # Save parameters
    save_dir: str = "./pretrained_models"
    log_interval: int = 100
    save_interval: int = 1000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset parameters
    use_random_prefix: bool = False  # 使用完整序列+causal mask已经包含所有prefix-suffix组合
    train_data_file: str = "/hy-tmp/MCTS/MassEnv/DataLoader/train_spectrum_embeds_msg.pt"
    val_data_file: str = "/hy-tmp/MCTS/MassEnv/DataLoader/val_spectrum_embeds_msg.pt"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    batch_size: int = 32
    max_len: int = 120
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    k_predictions: int = 1
    temperature: float = 0.0
    test_data_file: str = "/hy-tmp/MCTS/MassEnv/DataLoader/test_spectrum_embeds_msg.pt"
    results_dir: str = "./evaluation_results" 