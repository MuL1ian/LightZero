#!/usr/bin/env python3
"""
Test script to verify BERT formula tokenization in MassGymEnv.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from transformers import BertTokenizer
from zoo.masspecgym.envs.massgymenv import MassGymEnv

def test_formula_tokenization():
    """Test the formula tokenization functionality."""
    
    # Create environment configuration
    cfg = {
        'env_id': 'mass_spec_env',
        'max_len': 100,
        'formula_max_len': 50,
        'debug': True,
        'formula_masking': True,
    }
    
    # Create environment
    env = MassGymEnv(cfg)
    
    # Test different formulas
    test_formulas = [
        "C6H12O6",  # glucose
        "H2O",      # water
        "C2H6O",    # ethanol
        "C8H10N4O2", # caffeine
        "C21H30O2",  # very long formula
        "",          # empty formula
    ]
    
    print("Testing BERT formula tokenization:")
    print("=" * 50)
    
    for formula in test_formulas:
        print(f"\nFormula: '{formula}'")
        
        # Test the _encode_formula method
        try:
            encoded = env._encode_formula(formula)
            print(f"  Encoded shape: {encoded.shape}")
            print(f"  Encoded type: {type(encoded)}")
            print(f"  First 10 tokens: {encoded[:10].tolist()}")
            
            # Decode back to check
            decoded = env.formula_tokenizer.decode(encoded, skip_special_tokens=False)
            print(f"  Decoded: '{decoded}'")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
    print("Testing environment reset with formula tokenization:")
    
    # Test environment reset
    try:
        obs_timestep = env.reset()
        obs = obs_timestep.obs
        
        print(f"Observation shape: {obs['observation'].shape}")
        print(f"Expected dimension: {4096 + env.max_len + env.formula_max_len}")
        print(f"Action mask shape: {obs['action_mask'].shape}")
        print(f"Current formula: '{env.target_spectrum['formulas']}'")
        
        # Test a step
        action = 0  # First action
        step_timestep = env.step(action)
        step_obs = step_timestep.obs
        
        print(f"After step - Observation shape: {step_obs['observation'].shape}")
        
        print("\nTest passed! Formula tokenization is working correctly.")
        
    except Exception as e:
        print(f"Error during environment test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_formula_tokenization() 