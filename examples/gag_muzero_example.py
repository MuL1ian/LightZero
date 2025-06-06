#!/usr/bin/env python3
"""
GAG MuZero Example Usage

This script demonstrates how to use the simplified GAG MuZero system that automatically
extracts generated vs ground-truth SELFIES pairs from MassGymEnv episodes for adversarial training.

The new approach is much cleaner than complex metadata tracking and leverages the natural
structure of MassGymEnv episodes to get perfect positive/negative pairs.
"""

import torch
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.policy import create_policy
from ding.worker import create_serial_collector, create_learner

from lzero.worker.gag_muzero_collector import GAGMuZeroCollector
from lzero.policy.gag_muzero import GAGMuZeroPolicy


def main():
    """
    Example of setting up and using GAG MuZero with the new simplified approach.
    """
    
    # ===== Configuration =====
    config = dict(
        env=dict(
            type='mass_gym_env',  # Your MassGymEnv
            env_id='MolGenEnv-v0',
            # ... other env config
        ),
        
        policy=dict(
            type='gag_muzero',
            enable_adversarial_training=True,
            use_global_reward_network=True,
            adversarial_loss_weight=0.1,
            preference_loss_weight=0.05,
            # ... other policy config
        ),
        
        collector=dict(
            type='episode_gag_muzero',  # Use the new GAG collector
            n_episode=8,
            # ... other collector config  
        ),
        
        learner=dict(
            type='serial',
            # ... learner config
        )
    )
    
    config = compile_config(config)
    
    # ===== Create Components =====
    
    # 1. Create environment
    env = create_env_manager(config.env, seed=42)
    
    # 2. Create GAG MuZero policy
    policy = create_policy(config.policy)
    
    # 3. Create GAG collector  
    collector = GAGMuZeroCollector(
        env=env,
        policy=policy,
        **config.collector
    )
    
    # 4. Connect policy and collector (IMPORTANT!)
    policy.set_collector(collector)
    
    # 5. Create learner
    learner = create_learner(config.learner, policy=policy)
    
    print("=== GAG MuZero Setup Complete ===")
    print(f"Policy: {type(policy).__name__}")
    print(f"Collector: {type(collector).__name__}")
    print(f"GAG Statistics: {policy.get_gag_statistics()}")
    
    # ===== Training Loop =====
    for iteration in range(100):
        
        # 1. Collect data - GAG collector automatically extracts pairs at episode completion
        print(f"\n--- Iteration {iteration} ---")
        print("Collecting episodes...")
        
        collect_data = collector.collect(
            n_episode=config.collector.n_episode,
            train_iter=iteration,
            policy_kwargs={'temperature': 1.0, 'epsilon': 0.1}
        )
        
        # 2. Get collected GAG pairs
        gag_pairs = collector.get_collected_pairs()
        print(f"Collected {len(gag_pairs)} generated/ground-truth pairs")
        
        if gag_pairs:
            # Print example pair
            example_pair = gag_pairs[0]
            print(f"Example pair:")
            print(f"  Generated: {example_pair['generated_selfies'][:50]}...")
            print(f"  Ground-truth: {example_pair['ground_truth_selfies'][:50]}...")
            print(f"  Reward: {example_pair['episode_reward']:.3f}")
        
        # 3. Learn - GAG policy automatically uses pairs for adversarial training
        print("Learning with adversarial training...")
        learn_info = learner.train(collect_data, iteration)
        
        # 4. Print learning statistics
        if 'adversarial_loss' in learn_info:
            print(f"Adversarial Loss: {learn_info['adversarial_loss']:.4f}")
            print(f"Preference Loss: {learn_info['preference_loss']:.4f}")
            print(f"Reward Accuracy: {learn_info.get('reward_accuracy', 0.0):.3f}")
        
        # 5. Clear pairs buffer to avoid memory buildup
        collector.clear_collected_pairs()
        
        # 6. Print GAG statistics every 10 iterations
        if iteration % 10 == 0:
            stats = policy.get_gag_statistics()
            print(f"GAG Stats: {stats}")


def demo_gag_pair_extraction():
    """
    Demonstrate how GAG pairs are automatically extracted.
    """
    print("\n=== GAG Pair Extraction Demo ===")
    
    # This happens automatically in the GAG collector:
    # 
    # 1. Episode runs in MassGymEnv
    # 2. Agent generates SELFIES sequence through actions
    # 3. Episode ends with episode_timestep.info containing:
    #    - 'generated_selfies': What the agent produced
    #    - 'target_selfies': The ground-truth target
    #    - 'spectrum_embed': The spectrum embedding
    # 4. GAG collector extracts this info and stores pairs
    # 5. GAG policy uses pairs for adversarial training:
    #    - Computes reward(generated_selfies, spectrum_embed) -> negative instance
    #    - Computes reward(target_selfies, spectrum_embed) -> positive instance
    #    - Performs pairwise preference learning to prefer ground-truth
    
    example_episode_info = {
        'generated_selfies': '[C][C][O][C][=O]',  # What agent produced
        'target_selfies': '[C][C][O][C][=O][OH]',  # Ground-truth target
        'spectrum_embed': torch.randn(4096),  # Target spectrum
        'eval_episode_return': 0.75  # Episode reward
    }
    
    print("Example episode completion info:")
    for key, value in example_episode_info.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print("\nGAG collector automatically:")
    print("1. Extracts generated vs ground-truth SELFIES")
    print("2. Stores pairs with spectrum embeddings")
    print("3. Provides to GAG policy for adversarial training")
    print("4. No complex metadata tracking needed!")


if __name__ == '__main__':
    print("GAG MuZero - Simplified Adversarial Training")
    print("=" * 50)
    
    # Run demonstration
    demo_gag_pair_extraction()
    
    # Run main example (commented out to avoid env dependencies)
    # main()
    
    print("\n" + "=" * 50)
    print("Example complete! See comments in main() for actual usage.") 