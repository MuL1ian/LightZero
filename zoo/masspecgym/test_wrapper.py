#!/usr/bin/env python3
"""
Test script to verify the MassGym wrapper with BERT formula tokenization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from zoo.masspecgym.envs.massgym_wrapper import MassGymLightZeroEnv

def test_wrapper():
    """Test the wrapper functionality with formula tokenization."""
    
    # Create environment configuration
    cfg = {
        'env_id': 'mass_spec_env',
        'max_len': 100,
        'formula_max_len': 50,
        'debug': True,
        'formula_masking': True,
    }
    
    # Create wrapped environment
    env = MassGymLightZeroEnv(cfg)
    
    print("Testing MassGym LightZero wrapper:")
    print("=" * 50)
    print(f"Expected observation dimension: {env.expected_obs_dim}")
    print(f"Action space size: {env.action_size}")
    print(f"Max SELFIES length: {env.max_len}")
    print(f"Max formula length: {env.formula_max_len}")
    
    # Test environment reset
    try:
        obs = env.reset()
        
        print(f"\nReset observation shape: {obs['observation'].shape}")
        print(f"Action mask shape: {obs['action_mask'].shape}")
        print(f"To play: {obs['to_play']}")
        print(f"Chance: {obs['chance']}")
        print(f"Timestep: {obs['timestep']}")
        
        # Test a step
        action = 0  # First action
        timestep = env.step(action)
        step_obs = timestep.obs
        
        print(f"\nAfter step observation shape: {step_obs['observation'].shape}")
        print(f"Reward: {timestep.reward}")
        print(f"Done: {timestep.done}")
        
        print("\nWrapper test passed! Formula tokenization is working correctly.")
        
    except Exception as e:
        print(f"Error during wrapper test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wrapper() 