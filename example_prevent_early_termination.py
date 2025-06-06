#!/usr/bin/env python3
"""
Example script demonstrating the intelligent END token masking feature 
to prevent early termination in MassGymEnv.

This feature prevents the agent from selecting the END token too early 
by masking it out until the formula completion ratio reaches a threshold
or a maximum number of steps is reached.
"""

import sys
import os

# Add the necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from zoo.masspecgym.envs.massgymenv import MassGymEnv
from easydict import EasyDict
import numpy as np


def example_with_early_termination_prevention():
    """Example showing how to configure MassGymEnv to prevent early termination."""
    
    print("=" * 70)
    print("Example: MassGymEnv with Intelligent END Token Masking")
    print("=" * 70)
    
    # Configuration with early termination prevention enabled
    config = {
        'env_id': 'massgym',
        'debug': True,
        'max_episode_steps': 50,
        
        # Intelligent END token masking configuration
        'prevent_early_termination': True,      # Enable the feature
        'min_formula_completion': 0.8,          # Require 80% completion before END is allowed
        'allow_early_end_after_steps': 25,      # Allow END after 25 steps even if incomplete
        
        # Other configuration
        'formula_masking': True,
        'reward_normalize': False,
        'use_filter': True,
        'filter_len': 30,
    }
    
    print(f"Configuration:")
    print(f"  - Prevent early termination: {config['prevent_early_termination']}")
    print(f"  - Minimum formula completion: {config['min_formula_completion']}")
    print(f"  - Allow early end after steps: {config['allow_early_end_after_steps']}")
    print()
    
    # Create environment
    env = MassGymEnv(config)
    
    # Reset environment to get a target formula
    obs = env.reset()
    target_formula = env.target_spectrum['formulas']
    print(f"Target formula: {target_formula}")
    
    # Simulate an episode
    step = 0
    done = False
    
    print("\nSimulating episode steps:")
    print("-" * 50)
    
    while not done and step < 15:  # Limit to 15 steps for demonstration
        # Get valid actions
        action_mask = env.get_valid_actions()
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        
        # Check if END token is available
        end_token_idx = env.actions_list.index('<END>')
        end_available = action_mask[end_token_idx]
        
        # Show current state
        print(f"Step {step + 1}:")
        print(f"  Current SELFIES: '{env.current_selfies}'")
        print(f"  Used elements: {env.used_element_counts}")
        
        # Calculate completion
        from zoo.masspecgym.envs.utils import calculate_formula_completion_reward
        completion = calculate_formula_completion_reward(target_formula, env.used_element_counts)
        print(f"  Formula completion: {completion:.2f}")
        
        print(f"  END token available: {end_available}")
        print(f"  Valid actions: {len(valid_actions)} out of {len(env.actions_list)}")
        
        # Select a random valid action (not END if available)
        if len(valid_actions) > 1 and end_available:
            # Prefer non-END actions to see the masking in effect
            non_end_actions = [a for a in valid_actions if a != end_token_idx]
            if non_end_actions:
                action = np.random.choice(non_end_actions)
            else:
                action = np.random.choice(valid_actions)
        elif len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            print("  No valid actions available!")
            break
        
        action_name = env.actions_list[action]
        print(f"  Selected action: {action_name}")
        print()
        
        # Take the action
        timestep = env.step(action)
        done = timestep.done
        step += 1
    
    print("Episode completed!" if done else "Episode truncated for demonstration.")
    print(f"Final SELFIES: '{env.current_selfies}'")
    print(f"Final used elements: {env.used_element_counts}")
    
    # Calculate final completion
    final_completion = calculate_formula_completion_reward(target_formula, env.used_element_counts)
    print(f"Final completion ratio: {final_completion:.2f}")


def example_without_early_termination_prevention():
    """Example showing standard behavior without early termination prevention."""
    
    print("\n" + "=" * 70)
    print("Comparison: MassGymEnv WITHOUT Intelligent END Token Masking")
    print("=" * 70)
    
    # Configuration with early termination prevention disabled
    config = {
        'env_id': 'massgym',
        'debug': True,
        'max_episode_steps': 50,
        
        # Disable intelligent END token masking
        'prevent_early_termination': False,     # Disable the feature
        
        # Other configuration
        'formula_masking': True,
        'reward_normalize': False,
        'use_filter': True,
        'filter_len': 30,
    }
    
    print(f"Configuration:")
    print(f"  - Prevent early termination: {config['prevent_early_termination']}")
    print("  - END token is always available (original behavior)")
    print()
    
    # Create environment
    env = MassGymEnv(config)
    
    # Reset environment to get a target formula
    obs = env.reset()
    target_formula = env.target_spectrum['formulas']
    print(f"Target formula: {target_formula}")
    
    # Show that END token is always available
    step = 0
    for _ in range(3):  # Just show a few steps
        action_mask = env.get_valid_actions()
        end_token_idx = env.actions_list.index('<END>')
        end_available = action_mask[end_token_idx]
        
        print(f"Step {step + 1}:")
        print(f"  Current SELFIES: '{env.current_selfies}'")
        print(f"  Used elements: {env.used_element_counts}")
        print(f"  END token available: {end_available} (always True without prevention)")
        
        # Take a non-END action to continue
        valid_actions = [i for i, valid in enumerate(action_mask) if valid and i != end_token_idx]
        if valid_actions:
            action = np.random.choice(valid_actions)
            action_name = env.actions_list[action]
            print(f"  Selected action: {action_name}")
            timestep = env.step(action)
            step += 1
        else:
            print("  No non-END actions available!")
            break
        print()


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    print("Intelligent END Token Masking Demo")
    print("This feature prevents early episode termination by masking the END token")
    print("until the agent has used a sufficient portion of the target formula atoms.\n")
    
    try:
        # Run the main example
        example_with_early_termination_prevention()
        
        # Run the comparison example
        example_without_early_termination_prevention()
        
        print("\n" + "=" * 70)
        print("Configuration Parameters:")
        print("=" * 70)
        print("prevent_early_termination: Enable/disable the feature")
        print("min_formula_completion: Minimum completion ratio (0.0-1.0)")
        print("allow_early_end_after_steps: Safety limit to prevent infinite episodes")
        print("\nTip: Adjust these parameters based on your training requirements.")
        print("Higher completion ratios encourage more thorough formula exploration.")
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc() 