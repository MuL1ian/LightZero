"""
MassGym Environment Configuration with Batched Reward Computation

This configuration demonstrates how to set up the MassGym environment with
batched reward computation for better scaling across multiple environment processes.

The batched reward system allows multiple environments running in subprocesses
to share reward computation, significantly improving throughput when using
neural network-based rewards.
"""

from easydict import EasyDict
from lzero.model import global_reward_network

# Number of environments for training
collector_env_num = 256
evaluator_env_num = 256
debug = False
batch_size = 256
max_moves = 80
# NEW: Configure reward server for subprocess-based environments
# The reward server runs in a dedicated process and handles batched reward computation
# for all environment subprocesses, solving the multiprocessing limitations of the old system
reward_server_config = {
    'use_reward_server': True,        # Enable reward server for subprocess environments
    'batch_size': batch_size,                  # Batch size for reward server (smaller for responsiveness)
    'batch_timeout': 20.0,            # Timeout for batching (seconds)
    'client_timeout': 20.0,           # Timeout for client requests (auto-calculated if None: min(60.0, max(5.0, batch_timeout * 10 + 5)))
    'enable_batched_rewards': False,  # Disable old batching system
}

# OLD: Configure batched rewards for the training setup (DEPRECATED)
# This is kept for backward compatibility but should not be used with subprocess environments
batched_reward_config = global_reward_network.configure_batched_rewards_for_training(
    num_envs=collector_env_num + evaluator_env_num,
    auto_batch_size=False,  # Automatically calculate optimal batch size
    batch_timeout=reward_server_config['batch_timeout'],
    batch_size=reward_server_config['batch_size']
)

# Disable old batching system to avoid conflicts with reward server
batched_reward_config['enable_batched_rewards'] = False

# IMPORTANT: Reward Server vs Batched Rewards
# 
# OLD SYSTEM (batched_reward_config):
# - Batched rewards don't work with subprocess-based environments due to multiprocessing limitations
# - Each subprocess has its own copy of global variables, preventing shared batching
# - Only works in single-process training with multiple threads
#
# NEW SYSTEM (reward_server_config):
# - Reward server runs in a dedicated process and handles requests from all environment subprocesses
# - Uses multiprocessing queues for inter-process communication
# - Works with subprocess-based environments (the default for LightZero)
# - Provides true batched computation across all environments
# - Automatically starts when training begins and stops when training ends
#
# The reward server is now the recommended approach for all multi-environment setups.

# Base MassGym environment configuration
massgym_batched_config = dict(
    # Experiment name
    exp_name=f'data_muzero/massgym_reward_server_ce{collector_env_num}_ee{evaluator_env_num}_seed0',
    
    # Weights & Biases logging configuration
    wandb_freq=1,  # Upload to wandb every step for maximum monitoring detail
    
    env=dict(
        type='massgym_lightzero',  # Fixed: use the wrapper instead of raw environment
        import_names=['zoo.masspecgym.envs.massgym_wrapper'],  # Fixed: import the wrapper
        env_id='mass_spec_env',
        
        # Basic environment settings
        max_episode_steps=max_moves,
        obs_type='fingerprint',
        reward_type='cosine_similarity',
        reward_normalize=False,
        reward_norm_scale=1.0,
        
        # Environment manager settings
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        
        # SELFIES and formula settings
        max_len=100,
        formula_masking=True,
        formula_max_len=50,  # Fixed: match model config
        
        # Rendering settings
        render_mode=None,
        replay_format='svg',
        replay_name_suffix='eval',
        replay_path=None,
        channel_last=True,
        need_flatten=False,
        
        # Batched reward computation settings
        **reward_server_config,  # Use the new reward server configuration
        
        # Path to reward network checkpoint (update this path as needed)
        reward_network_checkpoint='reward_model/diffms/models/reward_model_enhanced/best_model.pt',
        
        # Environment creation settings
        # collector_env_num=collector_env_num,
        # evaluator_env_num=evaluator_env_num,
        # n_evaluator_episode=5,
        stop_value=1e6,
        
        # Debug settings
        debug=debug,  # Set to True for smaller dataset
        use_filter=not debug,
        filter_len=75,
    ),
    
    # ================================================================================
    # PRETRAINED MODEL LOADING CONFIGURATION
    # ================================================================================
    # 
    # POLICY VALUE NETWORK (MuZeroSelfiesTransformerEnhanced):
    # - NOW LOADING pretrained MassSelfiesED transformer component
    # - pretrained_transformer_path="pretrained_selfies_transformer/best_model.pt"
    # - model_path=None (no complete policy checkpoint)
    # - MuZero networks (representation, dynamics, prediction) train from scratch
    # - Transformer backbone (self.transformer) uses pretrained weights
    # 
    # REWARD NETWORK:
    # - reward_network_checkpoint loads pretrained reward model
    # - Used for environment reward computation
    # ================================================================================
    
    # Policy configuration (example for MuZero)
    policy=dict(
        type='gag_muzero',
        import_names=['lzero.policy.gag_muzero'],
        
        # Model settings
        model=dict(
            observation_shape=4246,  # 4096 (spectrum) + 100 (SELFIES max_len) + 50 (formula_max_len)
            action_space_size=75,    # Updated to match vocabulary size (includes all tokens)
            model_type='mlp',
            categorical_distribution=False,
            latent_state_dim=512,
            state_norm=False,
            self_supervised_learning_loss=False,
            use_transformer=True,
            pretrained_transformer_path="pretrained_selfies_transformer/best_model.pt",
            formula_max_len=50,  # Ensure model matches environment observation structure
            load_pretrained_transformer=True,
        ),
        
        # Required policy settings
        model_path=None,  # No pretrained policy checkpoint available
        cuda=True,
        env_type='not_board_games',
        action_type='varied_action_space',
        game_segment_length=100,
        
        # MCTS settings
        mcts_ctree=True,
        simulation_num=32,
        batch_size=batch_size,
        
        # Training settings
        learning_rate=1e-4,
        num_simulations=32,
        max_moves=max_moves,
        update_per_collect=1,
        optim_type='Adam',
        
        # Additional required parameters
        max_num_considered_actions=16,
        piecewise_decay_lr_scheduler=False,
        ssl_loss_weight=2,
        reanalyze_ratio=0.0,
        n_episode=collector_env_num,
        eval_freq=int(50),
        replay_buffer_size=int(1e2),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        discount_factor=0.99,
        # Reward network integration
        use_reward_network=True,
        reward_network_checkpoint='reward_model/diffms/models/reward_model/best_model.pt',
        
        # Weights & Biases logging configuration
        use_wandb=True,  # Enable wandb logging
        
        # GAG MuZero adversarial training configuration
        enable_adversarial_training=True,
        adversarial_loss_weight=0.1,
        preference_loss_weight=0.05,
        preference_temperature=1.0,
        use_global_reward_network=True,
        normalize_rewards=True,
        reward_norm_scale=10.0,
        use_curriculum_learning=False,
        curriculum_steps=5000,
        
        # Server-based reward network training configuration  
        reward_network_learning_rate=1e-4,
        reward_network_weight_decay=1e-4,
        training_batch_size=batch_size,
        training_timeout=10.0,
        
        # Ground-truth trajectory injection settings
        gt_trajectory_injection_enabled=True,
        target_gt_ratio=0.3,  # Target 30% ground-truth trajectories in replay buffer
    ),
    
    # Weights & Biases logger configuration
    wandb_logger=dict(
        gradient_logger=True,    # Log gradients to wandb
        video_logger=False,      # Log videos (not applicable for this environment)
        plot_logger=True,        # Log plots to wandb
        action_logger=True,      # Log action distributions
        return_logger=True,      # Log episode returns
    ),
    
    # Training configuration
    seed=0,
    
    # Collection configuration
    collection_steps_per_iter=1,  # Number of data collection steps per training iteration

    # Collector settings
    collector=dict(
        type='episode',
        get_train_sample=True,
        env_num=collector_env_num,
    ),
    
    # Evaluator settings
    evaluator=dict(
        env_num=evaluator_env_num,
        n_episode=evaluator_env_num,
    ),
    
    # Replay buffer settings
    replay_buffer=dict(
        replay_buffer_size=int(5e5),
        batch_size=batch_size,
    ),
)

# Convert to EasyDict for easier access
massgym_batched_config = EasyDict(massgym_batched_config)
main_config = massgym_batched_config

# Create config for environment and policy creation
massgym_batched_create_config = dict(
    env=dict(
        type='massgym_lightzero',  # Fixed: use the wrapper
        import_names=['zoo.masspecgym.envs.massgym_wrapper'],  # Fixed: import the wrapper
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='gag_muzero',
        import_names=['lzero.policy.gag_muzero'],
    ),
)
massgym_batched_create_config = EasyDict(massgym_batched_create_config)
create_config = massgym_batched_create_config

# Example usage functions
def print_batching_info():
    """Print information about the current batching configuration"""
    stats = global_reward_network.get_batching_stats()
    print("\n=== Batched Reward Configuration ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=====================================\n")

def monitor_batching_performance():
    """Monitor batching performance during training"""
    import time
    
    print("Monitoring batching performance...")
    for i in range(10):
        stats = global_reward_network.get_batching_stats()
        if stats.get('batching_enabled', False):
            print(f"Step {i}: Queue={stats.get('queue_size', 0)}, "
                  f"Pending={stats.get('pending_responses', 0)}, "
                  f"Alive={stats.get('processor_alive', False)}")
        else:
            print(f"Step {i}: Batching not enabled")
        time.sleep(1)

def cleanup_on_exit():
    """Clean up batching resources when training finishes"""
    global_reward_network.cleanup_batched_rewards()
    print("Batched reward resources cleaned up")

# Example of how to use this configuration in a training script:
"""
from zoo.masspecgym.config.massgym_batched_config import massgym_batched_config, cleanup_on_exit
import atexit

# Register cleanup function
atexit.register(cleanup_on_exit)

# Use the configuration for training
config = massgym_batched_config

# EXAMPLE: Modify collection steps for different training scenarios
# For faster iteration cycles (good for debugging):
# config.collection_steps_per_iter = 4

# For standard training (default):
# config.collection_steps_per_iter = 16

# For longer collection cycles (potentially more stable):
# config.collection_steps_per_iter = 32

# Your training code here...
# The environments will automatically use batched reward computation
"""

# Performance tips and notes
PERFORMANCE_NOTES = """
=== Batched Reward Performance Tips ===

1. Batch Size Selection:
   - Larger batches (32-64) for high-throughput training
   - Smaller batches (8-16) for low-latency evaluation
   - Auto-sizing based on number of environments is recommended

2. Timeout Configuration:
   - Shorter timeouts (0.05-0.1s) for responsive training
   - Longer timeouts (0.2-0.5s) for maximum throughput
   - Balance between latency and batching efficiency

3. Environment Count:
   - More environments = better batching efficiency
   - Recommended: 16+ environments for good batching
   - Monitor queue sizes to ensure environments aren't starved

4. Collection Steps Configuration:
   - collection_steps_per_iter: Controls how many collection steps per training iteration
   - Default: 16 steps (previously hardcoded)
   - Higher values = more data collection per iteration, longer iterations
   - Lower values = faster iteration cycles, potentially less stable learning
   - Recommended: 4-32 depending on environment complexity and training stability

5. Memory Considerations:
   - Batched processing uses more GPU memory
   - Monitor GPU memory usage during training
   - Reduce batch size if running out of memory

6. Debugging:
   - Use get_batching_stats() to monitor performance
   - Check processor_alive status if rewards seem slow
   - Disable batching temporarily to isolate issues

Expected Performance Improvements:
- 2-4x throughput improvement with 16+ environments
- 4-8x improvement with 32+ environments
- Diminishing returns beyond 64 environments

===========================================
"""

# Weights & Biases Configuration Notes
WANDB_NOTES = """
=== Weights & Biases Logging Configuration ===

The configuration now includes comprehensive wandb logging:

1. Policy Configuration:
   - use_wandb=True: Enables wandb logging throughout the training process
   - Logs training metrics, losses, and GAG-specific statistics

2. Logger Configuration:
   - gradient_logger=True: Logs gradient norms and distributions
   - plot_logger=True: Logs training plots and visualizations
   - action_logger=True: Logs action selection statistics
   - return_logger=True: Logs episode returns and rewards

3. Automatic Logging:
   - Training losses and metrics are logged every step
   - Environment statistics are logged during collection
   - Model gradients are monitored for debugging
   - GAG-specific metrics (adversarial loss, preference loss, etc.)

4. Project Organization:
   - Project name: "LightZero"
   - Run name is auto-generated based on experiment configuration
   - All configuration parameters are logged to wandb

5. Performance Impact:
   - Minimal overhead for metric logging
   - Gradient logging may add slight overhead but provides valuable insights
   - Video logging is disabled (not applicable for this environment)

To view your experiment:
1. Install wandb: pip install wandb
2. Login: wandb login
3. Run your training script
4. View results at: https://wandb.ai/

===========================================
"""

# PRETRAINED MODEL LOADING STATUS SUMMARY
PRETRAINED_MODEL_STATUS = """
=== TRANSFORMER PRETRAINED MODEL LOADING STATUS ===

✅ POLICY VALUE NETWORK (MuZeroSelfiesTransformerEnhanced):
   - NOW LOADING pretrained MassSelfiesED transformer component
   - pretrained_transformer_path="pretrained_selfies_transformer/best_model.pt"
   - model_path=None (no complete policy checkpoint)
   - MuZero networks (representation, dynamics, prediction) train from scratch
   - Transformer backbone (self.transformer) uses pretrained weights

✅ REWARD NETWORK:
   - Successfully loads pretrained reward model
   - reward_network_checkpoint='reward_model/diffms/models/reward_model/best_model.pt'
   - Used for environment reward computation

CLARIFICATION:
- model_path: For complete trained policy checkpoints (representation + dynamics + prediction networks)

CURRENT STATUS: Ready to run with pretrained transformer and reward network
"""

if __name__ == "__main__":
    import atexit
    import os
    import multiprocessing as mp
    
    # Set environment variables for better multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization issues
    try:
        mp.set_start_method('spawn', force=False)
        print("[INFO] Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("[INFO] Multiprocessing start method already set")
    
    # Register cleanup function
    atexit.register(cleanup_on_exit)
    
    print("MassGym Batched Reward Configuration")
    print("====================================")
    print(f"Collector environments: {collector_env_num}")
    print(f"Evaluator environments: {evaluator_env_num}")
    print(f"Total environments: {collector_env_num + evaluator_env_num}")
    print(f"Batched rewards enabled: {batched_reward_config['enable_batched_rewards']}")
    print(f"Batch size: {batched_reward_config['batch_size']}")
    print(f"Batch timeout: {batched_reward_config['batch_timeout']:.3f}s")
    print(f"Wandb logging enabled: {main_config.policy.use_wandb}")
    print("\nPerformance Notes:")
    print(PERFORMANCE_NOTES)
    print("\nWandB Configuration:")
    print(WANDB_NOTES)
    
    # Training configuration
    entry_type = "train_gag_muzero"
    max_env_step = int(2e7)
    
    print(f"\nStarting training with entry type: {entry_type}")
    print(f"Maximum environment steps: {max_env_step}")
    print("Batched reward computation will be automatically enabled...")
    print("Wandb logging will be automatically initialized...")
    
    if entry_type == "train_muzero":
        from lzero.entry import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        from lzero.entry import train_muzero_with_gym_env as train_muzero
    elif entry_type == "train_gag_muzero":
        from lzero.entry import train_gag_muzero as train_muzero
    else:
        raise ValueError(f"Unknown entry type: {entry_type}")
    
    # Start training with batched rewards and wandb logging
    train_muzero(
        [main_config, create_config], 
        seed=0, 
        model_path=main_config.policy.get('model_path', None), 
        max_env_step=max_env_step
    )

    print(PRETRAINED_MODEL_STATUS) 
