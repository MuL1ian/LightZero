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
    vocab_size: int = 1000
    max_len: int = 120  
    d_model: int = 512  
    n_enc: int = 4      
    n_dec: int = 6      
    n_head: int = 8 
    n_spectrum_heads: int = 32
    dropout: float = 0.25 
    
    # Training parameters 
    batch_size: int = 32
    learning_rate: float = 5e-5  
    num_epochs: int = 30
    warmup_steps: int = 50  
    gradient_clip: float = 0.5  
    weight_decay: float = 0.05 
    
    # Early stopping
    early_stopping_patience: int = 10  
    early_stopping_min_delta: float = 0.005  
    save_best_model: bool = True
    
    # Data parameters (spectrum fingerprint)
    spectrum_dim: int = 4096
    
    # Save parameters
    save_dir: str = "./pretrained_models"
    log_interval: int = 100
    save_interval: int = 2500
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset parameters
    train_data_file: str = "/home/zirui/MassEnv/DataLoader/train_spectrum_embeds_msg.pt"
    val_data_file: str = "/home/zirui/MassEnv/DataLoader/val_spectrum_embeds_msg.pt"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    batch_size: int = 32
    max_len: int = 120
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    k_predictions: int = 1
    temperature: float = 0.0
    test_data_file: str = "/home/zirui/MassEnv/DataLoader/test_spectrum_embeds_msg.pt"
    results_dir: str = "./evaluation_results" 