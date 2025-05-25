#!/usr/bin/env python3
"""
Test script to verify the enhanced MuZero transformer with BERT formula tokenization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from lzero.model.muzero_transformer import MuZeroSelfiesTransformerEnhanced
from zoo.masspecgym.envs.massgym_wrapper import MassGymLightZeroEnv

def test_enhanced_transformer():
    """Test the enhanced transformer with formula tokenization."""
    
    print("Testing Enhanced MuZero Transformer with Formula Tokenization:")
    print("=" * 60)
    
    # Create environment to get real observations
    cfg = {
        'env_id': 'mass_spec_env',
        'max_len': 100,
        'formula_max_len': 50,
        'debug': True,
        'formula_masking': True,
    }
    
    env = MassGymLightZeroEnv(cfg)
    
    # Create enhanced transformer model
    model_config = {
        'observation_shape': 4246,
        'max_len': 100,  # Match environment max_len
        'formula_max_len': 50,
        'd_model': 256,  # Smaller for testing
        'n_enc': 2,
        'n_dec': 2,
        'n_head': 4,
        'dropout': 0.1,
        'device': 'cpu',  # Use CPU for testing
        'target_formula': None,
    }
    
    model = MuZeroSelfiesTransformerEnhanced(**model_config)
    
    print(f"Model created with observation shape: {model_config['observation_shape']}")
    print(f"Spectrum dimension: {model.spectrum_dim}")
    print(f"SELFIES start index: {model.selfies_start_idx}")
    print(f"Formula start index: {model.formula_start_idx}")
    print(f"Formula max length: {model.formula_max_len}")
    
    # Test with real environment observation
    try:
        obs = env.reset()
        observation = obs['observation']
        
        print(f"\nEnvironment observation shape: {observation.shape}")
        print(f"Expected dimension: {env.expected_obs_dim}")
        
        # Test initial inference
        print("\nTesting initial inference...")
        initial_output = model.initial_inference(observation)
        
        print(f"Initial inference output:")
        print(f"  Value shape: {initial_output.value.shape}")
        print(f"  Policy logits shape: {initial_output.policy_logits.shape}")
        print(f"  Latent state shape: {initial_output.latent_state.shape}")
        print(f"  Reward: {initial_output.reward}")
        
        # Test SELFIES extraction
        print("\nTesting SELFIES extraction...")
        selfies_list = model._extract_selfies_from_latent_state(initial_output.latent_state)
        print(f"Extracted SELFIES: {selfies_list}")
        
        # Test formula extraction
        print("\nTesting formula extraction...")
        formula_list = model._extract_formula_from_latent_state(initial_output.latent_state)
        print(f"Extracted formula: {formula_list}")
        
        # Test recurrent inference
        print("\nTesting recurrent inference...")
        action = torch.tensor([0])  # First action
        recurrent_output = model.recurrent_inference(initial_output.latent_state, action)
        
        print(f"Recurrent inference output:")
        print(f"  Value shape: {recurrent_output.value.shape}")
        print(f"  Policy logits shape: {recurrent_output.policy_logits.shape}")
        print(f"  Latent state shape: {recurrent_output.latent_state.shape}")
        print(f"  Reward shape: {recurrent_output.reward.shape}")
        
        # Test SELFIES extraction after action
        print("\nTesting SELFIES extraction after action...")
        selfies_after = model._extract_selfies_from_latent_state(recurrent_output.latent_state)
        print(f"SELFIES after action: {selfies_after}")
        
        # Test multiple steps
        print("\nTesting multiple steps...")
        current_state = recurrent_output.latent_state
        for step in range(3):
            action = torch.tensor([step + 1])
            output = model.recurrent_inference(current_state, action)
            selfies = model._extract_selfies_from_latent_state(output.latent_state)
            print(f"  Step {step + 1}: Action={action.item()}, SELFIES='{selfies[0]}'")
            current_state = output.latent_state
        
        print("\n✅ Enhanced transformer test passed!")
        
    except Exception as e:
        print(f"\n❌ Enhanced transformer test failed: {e}")
        import traceback
        traceback.print_exc()

def test_dimension_consistency():
    """Test that dimensions are consistent throughout the pipeline."""
    
    print("\nTesting dimension consistency:")
    print("=" * 40)
    
    # Test with synthetic data
    batch_size = 2
    spectrum_dim = 4096
    selfies_len = 100
    formula_len = 50
    total_dim = spectrum_dim + selfies_len + formula_len
    
    # Create synthetic observation with valid structure
    # Spectrum part: random values
    spectrum_part = torch.randn(batch_size, spectrum_dim)
    
    # SELFIES part: valid token IDs (pad tokens mostly)
    selfies_part = torch.full((batch_size, selfies_len), 72, dtype=torch.long)  # 72 is pad token ID
    
    # Formula part: valid BERT token IDs (pad tokens)
    formula_part = torch.full((batch_size, formula_len), 0, dtype=torch.long)  # 0 is BERT pad token
    
    # Combine all parts
    synthetic_obs = torch.cat([
        spectrum_part, 
        selfies_part.float(), 
        formula_part.float()
    ], dim=-1)
    
    model = MuZeroSelfiesTransformerEnhanced(
        observation_shape=total_dim,
        max_len=selfies_len,
        formula_max_len=formula_len,
        device='cpu'
    )
    
    print(f"Synthetic observation shape: {synthetic_obs.shape}")
    
    # Test initial inference
    initial_out = model.initial_inference(synthetic_obs)
    print(f"Initial output latent state shape: {initial_out.latent_state.shape}")
    
    # Test recurrent inference
    actions = torch.tensor([0, 1])  # Batch of actions
    recurrent_out = model.recurrent_inference(initial_out.latent_state, actions)
    print(f"Recurrent output latent state shape: {recurrent_out.latent_state.shape}")
    
    # Verify dimensions are preserved
    assert initial_out.latent_state.shape == synthetic_obs.shape, "Initial latent state shape mismatch"
    assert recurrent_out.latent_state.shape == synthetic_obs.shape, "Recurrent latent state shape mismatch"
    
    print("✅ Dimension consistency test passed!")

if __name__ == "__main__":
    test_enhanced_transformer()
    test_dimension_consistency() 