import logging
import os
from functools import partial
from typing import Optional, Tuple

import torch
import wandb
from ding.config import compile_config
from ding.envs import create_env_manager
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed, get_rank
from ding.worker import BaseLearner
from tensorboardX import SummaryWriter

from lzero.entry.utils import log_buffer_memory_usage, log_buffer_run_time
from lzero.policy import visit_count_temperature
from lzero.policy.random_policy import LightZeroRandomPolicy
from lzero.worker import MuZeroCollector as Collector
from lzero.worker import MuZeroEvaluator as Evaluator
# Import GAG-specific components
from lzero.worker.gag_muzero_collector import GAGMuZeroCollector
try:
    from .utils import random_collect, calculate_update_per_collect
except ImportError:
    from utils import random_collect, calculate_update_per_collect


def log_to_wandb(data_dict, step=None, prefix=""):
    """Helper function to log data to wandb with optional prefix"""
    if wandb.run is not None:
        if prefix:
            data_dict = {f"{prefix}/{k}": v for k, v in data_dict.items()}
        if step is not None:
            wandb.log(data_dict, step=step)
        else:
            wandb.log(data_dict)


def train_gag_muzero(
        input_cfg: Tuple[dict, dict],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        The train entry for MCTS+RL algorithms, including MuZero, EfficientZero, Sampled EfficientZero, Gumbel Muzero, GAG Muzero.
    Arguments:
        - input_cfg (:obj:`Tuple[dict, dict]`): Config in dict type.
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - model_path (:obj:`Optional[str]`): The pretrained model path, which should
            point to the ckpt file of the pretrained model, and an absolute path is recommended.
            In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """

    cfg, create_cfg = input_cfg
    assert create_cfg.policy.type in ['efficientzero', 'muzero', 'muzero_context', 'muzero_rnn_full_obs', 'sampled_efficientzero', 'sampled_muzero', 'gumbel_muzero', 'stochastic_muzero', 'gag_muzero'], \
        "train_muzero entry now only support the following algo.: 'efficientzero', 'muzero', 'sampled_efficientzero', 'gumbel_muzero', 'stochastic_muzero', 'gag_muzero'"

    if create_cfg.policy.type in ['muzero', 'muzero_context', 'muzero_rnn_full_obs']:
        from lzero.mcts import MuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'efficientzero':
        from lzero.mcts import EfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_efficientzero':
        from lzero.mcts import SampledEfficientZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'sampled_muzero':
        from lzero.mcts import SampledMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gumbel_muzero':
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'stochastic_muzero':
        from lzero.mcts import StochasticMuZeroGameBuffer as GameBuffer
    elif create_cfg.policy.type == 'gag_muzero':
        # GAG MuZero uses the same buffer as Gumbel MuZero
        from lzero.mcts import GumbelMuZeroGameBuffer as GameBuffer

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = 'cuda'
    else:
        cfg.policy.device = 'cpu'

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    
    # Enhanced wandb configuration
    wandb_freq = cfg.get('wandb_freq', 1)  # Default to every step if not specified
    use_wandb = cfg.policy.get('use_wandb', False)
    
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    if cfg.policy.eval_offline:
        cfg.policy.learn.learner.hook.save_ckpt_after_iter = cfg.policy.eval_freq

    if use_wandb:
        # Enhanced wandb initialization with more detailed config
        wandb_config = {
            'algorithm': create_cfg.policy.type,
            'seed': seed,
            'batch_size': cfg.policy.batch_size,
            'learning_rate': cfg.policy.learning_rate,
            'collection_steps_per_iter': cfg.get('collection_steps_per_iter', 16),
            'wandb_upload_frequency': wandb_freq,
            'max_train_iter': max_train_iter,
            'max_env_step': max_env_step,
            'device': cfg.policy.device,
            'eval_freq': cfg.policy.eval_freq,
        }
        wandb_config.update(cfg)
        
        wandb.init(
            project="LightZero-GAG-MuZero",
            config=wandb_config,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
            name=f"{create_cfg.policy.type}_{cfg.exp_name}",
            tags=[create_cfg.policy.type, "molecular_generation", "mass_spectrometry"],
        )
        
        # Log initial configuration (don't specify step to avoid conflicts)
        if get_rank() == 0:
            wandb.config.update({
                'algorithm': create_cfg.policy.type,
                'wandb_freq': wandb_freq,
                'collection_steps': cfg.get('collection_steps_per_iter', 16),
                'batch_size': cfg.policy.batch_size,
                'learning_rate': cfg.policy.learning_rate,
            })

    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])

    # load pretrained model
    if model_path is not None:
        policy.learn_mode.load_state_dict(torch.load(model_path, map_location=cfg.policy.device))

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)

    # ==============================================================
    # MCTS+RL algorithms related core code
    # ==============================================================
    policy_config = cfg.policy
    batch_size = policy_config.batch_size
    # specific game buffer for MCTS+RL algorithms
    replay_buffer = GameBuffer(policy_config)
    
    # ==============================================================
    # GAG MuZero specific collector setup
    # ==============================================================
    if create_cfg.policy.type == 'gag_muzero':
        # Use GAG collector for automatic generated/ground-truth pair extraction
        collector = GAGMuZeroCollector(
            env=collector_env,
            policy=policy.collect_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            policy_config=policy_config,
        )
        
        # CRITICAL: Connect policy and collector for GAG adversarial training
        if hasattr(policy, 'set_collector'):
            policy.set_collector(collector)
        
        # Set up reward function and vocabulary for ground-truth trajectory generation
        if hasattr(policy, '_reward_function') and policy._reward_function is not None:
            collector.set_reward_function(policy._reward_function)
        
        # Set target ground-truth trajectory ratio (configurable)
        target_gt_ratio = cfg.get('target_gt_ratio', 0.3)  # Default to 30%
        if hasattr(collector, 'set_target_gt_ratio'):
            collector.set_target_gt_ratio(target_gt_ratio)
        
        # Try to get vocabulary from the policy or environment
        vocab_dict = None
        
        # Try multiple sources for vocabulary with debug logging
        if get_rank() == 0:
            logging.info("GAG MuZero: Searching for vocabulary...")
            
        # PRIORITY 1: Get vocabulary directly from environment tokenizer (matches actual env)
        if hasattr(collector_env, 'tokenizer') and hasattr(collector_env.tokenizer, 'get_vocab'):
            vocab_dict = collector_env.tokenizer.get_vocab()
            if get_rank() == 0:
                logging.info("GAG MuZero: Found vocabulary from collector_env.tokenizer.get_vocab()")
        # PRIORITY 2: Get from environment's vocab method
        elif hasattr(collector_env, 'get_vocab'):
            vocab_dict = collector_env.get_vocab()
            if get_rank() == 0:
                logging.info("GAG MuZero: Found vocabulary from collector_env.get_vocab()")
        # PRIORITY 3: Get from policy model tokenizer
        elif hasattr(policy, '_learn_model') and hasattr(policy._learn_model, 'selfies_tokenizer'):
            if hasattr(policy._learn_model.selfies_tokenizer, 'get_vocab'):
                vocab_dict = policy._learn_model.selfies_tokenizer.get_vocab()
                if get_rank() == 0:
                    logging.info("GAG MuZero: Found vocabulary from policy._learn_model.selfies_tokenizer.get_vocab()")
            elif hasattr(policy._learn_model.selfies_tokenizer, 'vocab'):
                vocab_dict = policy._learn_model.selfies_tokenizer.vocab
                if get_rank() == 0:
                    logging.info("GAG MuZero: Found vocabulary from policy._learn_model.selfies_tokenizer.vocab")
        elif hasattr(policy, '_learn_model') and hasattr(policy._learn_model, 'transformer') and hasattr(policy._learn_model.transformer, 'tokenizer'):
            if hasattr(policy._learn_model.transformer.tokenizer, 'get_vocab'):
                vocab_dict = policy._learn_model.transformer.tokenizer.get_vocab()
                if get_rank() == 0:
                    logging.info("GAG MuZero: Found vocabulary from policy._learn_model.transformer.tokenizer.get_vocab()")
        elif hasattr(policy, '_learn_model') and hasattr(policy._learn_model, 'tok'):
            if hasattr(policy._learn_model.tok, 'get_vocab'):
                vocab_dict = policy._learn_model.tok.get_vocab()
                if get_rank() == 0:
                    logging.info("GAG MuZero: Found vocabulary from policy._learn_model.tok.get_vocab()")
        elif hasattr(policy, 'vocab'):
            vocab_dict = policy.vocab
            if get_rank() == 0:
                logging.info("GAG MuZero: Found vocabulary from policy.vocab")
        
        # If we still don't have vocab, create extended SELFIES vocabulary
        if vocab_dict is None:
            if get_rank() == 0:
                logging.info("GAG MuZero: Creating extended SELFIES vocabulary for ground-truth trajectories")
            
            # Create full vocabulary with semantic robust alphabet
            import selfies as sf
            from main.LightZero.lzero.model.selfies_tokenizer import SelfiesTokenizer
            
            # Use the same tokenizer as in the model
            temp_tokenizer = SelfiesTokenizer(max_len=100)
            vocab_dict = temp_tokenizer.get_vocab().copy()
        
        if get_rank() == 0:
            original_size = len(vocab_dict)
            logging.info(f"GAG MuZero: Using actual environment tokenizer vocabulary with {original_size} tokens")
            
            # Check what tokens are actually in the vocabulary
            test_tokens = ['[C@H1]', '[C@@H1]', '[/C]', '[\\C]', '[nH]']
            present_tokens = [token for token in test_tokens if token in vocab_dict]
            missing_tokens = [token for token in test_tokens if token not in vocab_dict]
            
            if present_tokens:
                print(f"GAG MuZero: Environment tokenizer includes stereochemistry tokens: {len(present_tokens)}/{len(test_tokens)}")
                print(f"  Present: {present_tokens}")
            
            if missing_tokens:
                print(f"GAG MuZero: Environment tokenizer missing stereochemistry tokens: {len(missing_tokens)}/{len(test_tokens)}")
                print(f"  Missing: {missing_tokens}")
                print("GAG MuZero: Using actual tokenizer vocabulary - unknown tokens will be skipped during processing")
            
            # DO NOT extend vocabulary - use exactly what the environment has
            print(f"GAG MuZero: Final vocabulary size: {len(vocab_dict)} tokens (no extension applied)")

        collector.set_vocab_dict(vocab_dict)
            
        if get_rank() == 0:
            logging.info("GAG MuZero: Using GAG collector for automatic pair extraction")
            collection_steps = cfg.get('collection_steps_per_iter', 16)
            logging.info(f"Collection steps per iteration: {collection_steps}")
            if hasattr(policy, 'get_gag_statistics'):
                logging.info(f"GAG Statistics: {policy.get_gag_statistics()}")
            else:
                logging.info("GAG Statistics: get_gag_statistics method not available")
    else:
        # Use regular collector for other algorithms
        collector = Collector(
            env=collector_env,
            policy=policy.collect_mode,
            tb_logger=tb_logger,
            exp_name=cfg.exp_name,
            policy_config=policy_config,
        )
    
    evaluator = Evaluator(
        eval_freq=cfg.policy.eval_freq,
        n_evaluator_episode=cfg.env.n_evaluator_episode,
        stop_value=cfg.env.stop_value,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
        policy_config=policy_config
    )

    # ==============================================================
    # Main loop
    # ==============================================================
    # Learner's before_run hook.
    learner.call_hook('before_run')
    if use_wandb:
        policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

    if cfg.policy.update_per_collect is not None:
        update_per_collect = cfg.policy.update_per_collect

    # The purpose of collecting random data before training:
    # Exploration: Collecting random data helps the agent explore the environment and avoid getting stuck in a suboptimal policy prematurely.
    # Comparison: By observing the agent's performance during random action-taking, we can establish a baseline to evaluate the effectiveness of reinforcement learning algorithms.
    if cfg.policy.random_collect_episode_num > 0:
        random_collect(cfg.policy, policy, LightZeroRandomPolicy, collector, collector_env, replay_buffer)

    if cfg.policy.eval_offline:
        eval_train_iter_list = []
        eval_train_envstep_list = []

    # Evaluate the random agent
    stop, episode_info = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
    
    # Log initial evaluation results to wandb
    if use_wandb and get_rank() == 0:
        log_to_wandb({
            'evaluation/reward': episode_info.get('reward_mean', 0.0),
            'evaluation/episode': 0,
            'evaluation/is_random_agent': True,
        })

    while True:
        log_buffer_memory_usage(learner.train_iter, replay_buffer, tb_logger)
        log_buffer_run_time(learner.train_iter, replay_buffer, tb_logger)
        
        # Log buffer statistics to wandb
        if use_wandb and get_rank() == 0 and learner.train_iter % wandb_freq == 0:
            buffer_stats = {
                'buffer/size': replay_buffer.get_num_of_transitions(),
                'buffer/capacity': getattr(replay_buffer, 'capacity', 'N/A'),
                'buffer/memory_usage_mb': getattr(replay_buffer, 'memory_usage', 0) / (1024 * 1024) if hasattr(replay_buffer, 'memory_usage') else 0,
            }
            if hasattr(replay_buffer, 'get_buffer_stats'):
                buffer_stats.update(replay_buffer.get_buffer_stats())
            log_to_wandb(buffer_stats)
        
        collect_kwargs = {}
        # set temperature for visit count distributions according to the train_iter,
        # please refer to Appendix D in MuZero paper for details.
        collect_kwargs['temperature'] = visit_count_temperature(
            policy_config.manual_temperature_decay,
            policy_config.fixed_temperature_value,
            policy_config.threshold_training_steps_for_final_temperature,
            trained_steps=learner.train_iter
        )

        if policy_config.eps.eps_greedy_exploration_in_collect:
            epsilon_greedy_fn = get_epsilon_greedy_fn(
                start=policy_config.eps.start,
                end=policy_config.eps.end,
                decay=policy_config.eps.decay,
                type_=policy_config.eps.type
            )
            collect_kwargs['epsilon'] = epsilon_greedy_fn(collector.envstep)
        else:
            collect_kwargs['epsilon'] = 0.0

        # Log exploration parameters to wandb
        if use_wandb and get_rank() == 0 and learner.train_iter % wandb_freq == 0:
            log_to_wandb({
                'exploration/temperature': collect_kwargs['temperature'],
                'exploration/epsilon': collect_kwargs['epsilon'],
                'training/env_step': collector.envstep,
                'training/train_iter': learner.train_iter,
            })

        # Evaluate policy performance.
        if evaluator.should_eval(learner.train_iter):
            if cfg.policy.eval_offline:
                eval_train_iter_list.append(learner.train_iter)
                eval_train_envstep_list.append(collector.envstep)
            else:
                stop, episode_info = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                
                # Enhanced evaluation logging to wandb
                if use_wandb and get_rank() == 0:
                    eval_data = {
                        'evaluation/reward': episode_info.get('reward_mean', 0.0),
                        'evaluation/train_iter': learner.train_iter,
                        'evaluation/env_step': collector.envstep,
                        'evaluation/is_random_agent': False,
                    }
                    
                    # Add more evaluation metrics from episode_info
                    for key, value in episode_info.items():
                        if isinstance(value, (int, float)) and key not in ['train_iter', 'ckpt_name']:
                            eval_data[f'evaluation/{key}'] = value
                    
                    log_to_wandb(eval_data)
                    
                    # Log evaluation milestone
                    reward_mean = episode_info.get('reward_mean', 0.0)
                    logging.info(f"Evaluation at iter {learner.train_iter}: reward={reward_mean:.4f}, env_step={collector.envstep}")
                
                if stop:
                    break

        # Collect data by default config n_sample/n_episode.
        collection_steps = cfg.get('collection_steps_per_iter', 16)  # Default to 16 if not specified
        total_gag_pairs = 0
        collection_rewards = []
        
        for i in range(collection_steps):
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            
            # Extract collection statistics
            if new_data:
                episode_rewards = [episode['reward'] for episode in new_data if 'reward' in episode]
                if episode_rewards:
                    collection_rewards.extend(episode_rewards)

            # ==============================================================
            # GAG MuZero specific logging
            # ==============================================================
            if create_cfg.policy.type == 'gag_muzero' and hasattr(collector, 'get_collected_pairs'):
                # Log GAG-specific statistics
                gag_pairs = collector.get_collected_pairs()
                total_gag_pairs += len(gag_pairs)
                
                # Get ground-truth trajectories
                gt_trajectories = collector.get_ground_truth_trajectories() if hasattr(collector, 'get_ground_truth_trajectories') else []
                
                if get_rank() == 0 and len(gag_pairs) > 0:
                    avg_episode_reward = sum(p.get('episode_reward', 0.0) for p in gag_pairs) / len(gag_pairs)
                    avg_episode_length = sum(p.get('episode_length', 0) for p in gag_pairs) / len(gag_pairs)
                    
                    # Enhanced GAG logging
                    if use_wandb and learner.train_iter % wandb_freq == 0:
                        gag_data = {
                            'gag/collected_pairs_count': len(gag_pairs),
                            'gag/avg_episode_reward': avg_episode_reward,
                            'gag/avg_episode_length': avg_episode_length,
                            'gag/total_pairs_this_iter': total_gag_pairs,
                            'gag/collection_step': i + 1,
                            'gag/ground_truth_trajectories_count': len(gt_trajectories),
                        }
                        
                        # Add more detailed GAG statistics
                        if len(gag_pairs) > 0:
                            generated_lengths = [len(p.get('generated_selfies', '')) for p in gag_pairs]
                            ground_truth_lengths = [len(p.get('ground_truth_selfies', '')) for p in gag_pairs]
                            gag_data.update({
                                'gag/avg_generated_length': sum(generated_lengths) / len(generated_lengths) if generated_lengths else 0,
                                'gag/avg_ground_truth_length': sum(ground_truth_lengths) / len(ground_truth_lengths) if ground_truth_lengths else 0,
                                'gag/min_episode_reward': min(p.get('episode_reward', 0.0) for p in gag_pairs),
                                'gag/max_episode_reward': max(p.get('episode_reward', 0.0) for p in gag_pairs),
                            })
                        
                        # Add ground-truth trajectory statistics
                        if len(gt_trajectories) > 0:
                            avg_gt_reward = sum(t.get('final_reward', 0.0) for t in gt_trajectories) / len(gt_trajectories)
                            avg_gt_return = sum(t.get('discounted_return', 0.0) for t in gt_trajectories) / len(gt_trajectories)
                            avg_gt_length = sum(t.get('trajectory_length', 0) for t in gt_trajectories) / len(gt_trajectories)
                            
                            gag_data.update({
                                'gag/avg_gt_final_reward': avg_gt_reward,
                                'gag/avg_gt_discounted_return': avg_gt_return,
                                'gag/avg_gt_trajectory_length': avg_gt_length,
                                'gag/gt_trajectories_with_rewards': sum(1 for t in gt_trajectories if t.get('final_reward', 0) > 0),
                            })
                        
                        # Add replay buffer ratio statistics
                        if hasattr(collector, 'get_current_gt_ratio'):
                            current_gt_ratio = collector.get_current_gt_ratio()
                            target_gt_ratio = getattr(collector, '_target_gt_ratio', 0.3)
                            
                            gag_data.update({
                                'gag/buffer_gt_ratio': current_gt_ratio,
                                'gag/buffer_gt_ratio_target': target_gt_ratio,
                                'gag/buffer_gt_count': getattr(collector, '_gt_trajectories_in_buffer', 0),
                                'gag/buffer_total_count': getattr(collector, '_total_trajectories_in_buffer', 0),
                            })
                        
                        log_to_wandb(gag_data)
                    
                    if tb_logger:
                        tb_logger.add_scalar('gag_muzero/collected_pairs_count', len(gag_pairs), learner.train_iter)
                        tb_logger.add_scalar('gag_muzero/avg_episode_reward', avg_episode_reward, learner.train_iter)
                        tb_logger.add_scalar('gag_muzero/avg_episode_length', avg_episode_length, learner.train_iter)
                    
                    # Log example pair for debugging (reduced frequency)
                    if len(gag_pairs) > 0 and learner.train_iter % (wandb_freq * 10) == 0:
                        example_pair = gag_pairs[0]
                        logging.debug(f"GAG MuZero - Example pair: Generated='{example_pair['generated_selfies'][:30]}...', "
                                    f"Ground-truth='{example_pair['ground_truth_selfies'][:30]}...', "
                                    f"Reward={example_pair['episode_reward']:.3f}")

            # Determine updates per collection
            update_per_collect = calculate_update_per_collect(cfg, new_data)

            # save returned new_data collected by the collector
            replay_buffer.push_game_segments(new_data)
            # remove the oldest data if the replay buffer is full.
            replay_buffer.remove_oldest_data_to_fit()
            
            # ==============================================================
            # GAG MuZero: Inject ground-truth trajectories to maintain 30% ratio
            # ==============================================================
            if create_cfg.policy.type == 'gag_muzero' and hasattr(collector, 'inject_ground_truth_trajectories'):
                collector.inject_ground_truth_trajectories(replay_buffer)

        # Log collection summary to wandb
        if use_wandb and get_rank() == 0 and learner.train_iter % wandb_freq == 0:
            collection_summary = {
                'collection/steps_per_iter': collection_steps,
                'collection/total_gag_pairs': total_gag_pairs,
                'collection/env_step': collector.envstep,
            }
            
            if collection_rewards:
                collection_summary.update({
                    'collection/avg_reward': sum(collection_rewards) / len(collection_rewards),
                    'collection/min_reward': min(collection_rewards),
                    'collection/max_reward': max(collection_rewards),
                    'collection/num_episodes': len(collection_rewards),
                })
            
            # Add replay buffer information for GAG MuZero
            if create_cfg.policy.type == 'gag_muzero' and hasattr(collector, 'get_current_gt_ratio'):
                buffer_gt_ratio = collector.get_current_gt_ratio()
                collection_summary.update({
                    'collection/buffer_size': replay_buffer.get_num_of_transitions(),
                    'collection/buffer_gt_ratio': buffer_gt_ratio,
                    'collection/buffer_gt_target': getattr(collector, '_target_gt_ratio', 0.3),
                })
            
            log_to_wandb(collection_summary)

        # Learn policy from collected data.
        for i in range(update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            if replay_buffer.get_num_of_transitions() > batch_size:
                train_data = replay_buffer.sample(batch_size, policy)
            else:
                logging.warning(
                    f'The data in replay_buffer is not sufficient to sample a mini-batch: '
                    f'batch_size: {batch_size}, '
                    f'{replay_buffer} '
                    f'continue to collect now ....'
                )
                break

            if use_wandb:
                policy.set_train_iter_env_step(learner.train_iter, collector.envstep)

            # The core train steps for MCTS+RL algorithms.
            # For GAG MuZero, pass ground-truth trajectories if available
            if create_cfg.policy.type == 'gag_muzero' and hasattr(collector, 'get_ground_truth_trajectories'):
                gt_trajectories = collector.get_ground_truth_trajectories()
                if hasattr(learner, 'policy') and hasattr(learner.policy, 'learn'):
                    # Pass ground-truth trajectories to the policy's learn method
                    log_vars = learner.train(train_data, collector.envstep, ground_truth_trajectories=gt_trajectories)
                else:
                    log_vars = learner.train(train_data, collector.envstep)
            else:
                log_vars = learner.train(train_data, collector.envstep)

            # ==============================================================
            # Enhanced training loss logging to wandb
            # ==============================================================
            if get_rank() == 0 and use_wandb and learner.train_iter % wandb_freq == 0:
                if isinstance(log_vars, list) and len(log_vars) > 0:
                    train_info = log_vars[0]
                    
                    # Log core training losses
                    training_losses = {}
                    core_metrics = ['total_loss', 'policy_loss', 'value_loss', 'reward_loss', 'consistency_loss']
                    for metric in core_metrics:
                        if metric in train_info:
                            training_losses[f'loss/{metric}'] = train_info[metric]
                    
                    # Log learning rates and optimization metrics
                    if 'lr' in train_info:
                        training_losses['optimization/learning_rate'] = train_info['lr']
                    if 'grad_norm' in train_info:
                        training_losses['optimization/gradient_norm'] = train_info['grad_norm']
                    if 'entropy' in train_info:
                        training_losses['optimization/entropy'] = train_info['entropy']
                    
                    # Log MuZero specific metrics
                    muzero_metrics = ['value_priority_orig', 'value_target', 'reward_target', 'policy_target']
                    for metric in muzero_metrics:
                        if metric in train_info:
                            training_losses[f'muzero/{metric}'] = train_info[metric]
                    
                    # Log any additional metrics present in train_info
                    for key, value in train_info.items():
                        if key not in core_metrics + muzero_metrics + ['lr', 'grad_norm', 'entropy']:
                            if isinstance(value, (int, float)):
                                training_losses[f'training/{key}'] = value
                    
                    log_to_wandb(training_losses)

            # ==============================================================
            # GAG MuZero specific training logging
            # ==============================================================
            if create_cfg.policy.type == 'gag_muzero' and get_rank() == 0:
                # Log GAG-specific training metrics
                if isinstance(log_vars, list) and len(log_vars) > 0:
                    train_info = log_vars[0]
                    if 'adversarial_loss' in train_info:
                        # Enhanced GAG logging to wandb (every step)
                        if use_wandb and learner.train_iter % wandb_freq == 0:
                            gag_training_losses = {
                                'gag_loss/adversarial_loss': train_info['adversarial_loss'],
                                'gag_loss/preference_loss': train_info.get('preference_loss', 0.0),
                                'gag_loss/reward_accuracy': train_info.get('reward_accuracy', 0.0),
                                'gag_loss/adversarial_weight': train_info.get('adversarial_weight', 0.0),
                                'gag_loss/reward_network_loss': train_info.get('reward_network_loss', 0.0),
                                'gag_training/gag_pairs_collected': train_info.get('gag_pairs_collected', 0),
                                'gag_training/server_training_timeout': train_info.get('server_training_timeout', 0.0),
                            }
                            
                            # Add server training status as a categorical metric
                            server_status = train_info.get('server_training_status', 'unknown')
                            status_mapping = {
                                'success': 1.0,
                                'failed': 0.0,
                                'no_data': 0.5,
                                'disabled': -1.0,
                                'unknown': -0.5
                            }
                            gag_training_losses['gag_training/server_status_numeric'] = status_mapping.get(server_status, -0.5)
                            
                            # Add reward network training metrics if available
                            if 'generated_rewards_mean' in train_info:
                                gag_training_losses['gag_rewards/generated_mean'] = train_info['generated_rewards_mean']
                            if 'ground_truth_rewards_mean' in train_info:
                                gag_training_losses['gag_rewards/ground_truth_mean'] = train_info['ground_truth_rewards_mean']
                            if 'reward_gap' in train_info:
                                gag_training_losses['gag_rewards/reward_gap'] = train_info['reward_gap']
                            
                            log_to_wandb(gag_training_losses)
                        
                        if tb_logger is not None:
                            tb_logger.add_scalar('gag_muzero/adversarial_loss', train_info['adversarial_loss'], learner.train_iter)
                            tb_logger.add_scalar('gag_muzero/preference_loss', train_info.get('preference_loss', 0.0), learner.train_iter)
                            tb_logger.add_scalar('gag_muzero/reward_accuracy', train_info.get('reward_accuracy', 0.0), learner.train_iter)
                            tb_logger.add_scalar('gag_muzero/adversarial_weight', train_info.get('adversarial_weight', 0.0), learner.train_iter)
                            # Add server training metrics
                            tb_logger.add_scalar('gag_muzero/reward_network_loss', train_info.get('reward_network_loss', 0.0), learner.train_iter)
                            tb_logger.add_scalar('gag_muzero/gag_pairs_collected', train_info.get('gag_pairs_collected', 0), learner.train_iter)
                            tb_logger.add_scalar('gag_muzero/server_training_timeout', train_info.get('server_training_timeout', 0.0), learner.train_iter)
                        
                        # Log detailed status every 100 iterations to avoid spam in logs (but still log to wandb every step)
                        if learner.train_iter % 100 == 0:
                            server_status = train_info.get('server_training_status', 'unknown')
                            gag_pairs = train_info.get('gag_pairs_collected', 0)
                            
                            if server_status == 'success':
                                server_msg = "✓ Server Training Success"
                            elif server_status == 'failed':
                                error_msg = train_info.get('server_training_error', 'Unknown error')
                                server_msg = f"✗ Server Training Failed: {error_msg}"
                            elif server_status == 'no_data':
                                server_msg = "○ No GAG Data"
                            elif server_status == 'disabled':
                                server_msg = "◊ Training Disabled"
                            else:
                                server_msg = f"? Unknown Status: {server_status}"
                            
                            logging.info(f"GAG MuZero Training - Adversarial Loss: {train_info['adversarial_loss']:.4f}, "
                                       f"Preference Loss: {train_info.get('preference_loss', 0.0):.4f}, "
                                       f"Reward Accuracy: {train_info.get('reward_accuracy', 0.0):.3f}, "
                                       f"GAG Pairs: {gag_pairs}, "
                                       f"Server: {server_msg}")

            if cfg.policy.use_priority:
                replay_buffer.update_priority(train_data, log_vars[0]['value_priority_orig'])

        # ==============================================================
        # GAG MuZero specific cleanup
        # ==============================================================
        if create_cfg.policy.type == 'gag_muzero' and hasattr(collector, 'clear_collected_pairs'):
            # Clear GAG pairs buffer to avoid memory buildup
            collector.clear_collected_pairs()
            
            # Clear ground-truth trajectories buffer
            if hasattr(collector, 'clear_ground_truth_trajectories'):
                collector.clear_ground_truth_trajectories()

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            if cfg.policy.eval_offline:
                logging.info(f'eval offline beginning...')
                ckpt_dirname = './{}/ckpt'.format(learner.exp_name)
                
                offline_eval_results = []
                # Evaluate the performance of the pretrained model.
                for eval_idx, (train_iter, collector_envstep) in enumerate(zip(eval_train_iter_list, eval_train_envstep_list)):
                    ckpt_name = 'iteration_{}.pth.tar'.format(train_iter)
                    ckpt_path = os.path.join(ckpt_dirname, ckpt_name)
                    # load the ckpt of pretrained model
                    policy.learn_mode.load_state_dict(torch.load(ckpt_path, map_location=cfg.policy.device))
                    stop, episode_info = evaluator.eval(learner.save_checkpoint, train_iter, collector_envstep)
                    
                    reward_mean = episode_info.get('reward_mean', 0.0)
                    
                    # Log offline evaluation results to wandb
                    if use_wandb and get_rank() == 0:
                        offline_eval_data = {
                            'offline_eval/reward': reward_mean,
                            'offline_eval/train_iter': train_iter,
                            'offline_eval/env_step': collector_envstep,
                            'offline_eval/checkpoint_index': eval_idx,
                        }
                        # Add other metrics from episode_info
                        for key, value in episode_info.items():
                            if isinstance(value, (int, float)) and key not in ['train_iter', 'ckpt_name']:
                                offline_eval_data[f'offline_eval/{key}'] = value
                        log_to_wandb(offline_eval_data)
                    
                    offline_eval_results.append({
                        'train_iter': train_iter,
                        'collector_envstep': collector_envstep,
                        'reward': reward_mean
                    })
                    
                    logging.info(
                        f'eval offline at train_iter: {train_iter}, collector_envstep: {collector_envstep}, reward: {reward_mean}')
                
                # Log offline evaluation summary
                if use_wandb and get_rank() == 0 and offline_eval_results:
                    rewards = [r['reward'] for r in offline_eval_results]
                    offline_summary = {
                        'offline_eval_summary/num_checkpoints': len(offline_eval_results),
                        'offline_eval_summary/best_reward': max(rewards),
                        'offline_eval_summary/final_reward': rewards[-1],
                        'offline_eval_summary/avg_reward': sum(rewards) / len(rewards),
                        'offline_eval_summary/reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
                    }
                    log_to_wandb(offline_summary)
                
                logging.info(f'eval offline finished!')
            break

    # ==============================================================
    # Enhanced final logging for GAG MuZero and wandb
    # ==============================================================
    if get_rank() == 0:
        # Log final training statistics to wandb
        if use_wandb:
            final_training_stats = {
                'final/total_train_iterations': learner.train_iter,
                'final/total_env_steps': collector.envstep,
                'final/final_buffer_size': replay_buffer.get_num_of_transitions(),
                'final/training_completed': 1.0,
            }
            
            # Add algorithm-specific final stats
            if create_cfg.policy.type == 'gag_muzero':
                if hasattr(policy, 'get_gag_statistics'):
                    gag_final_stats = policy.get_gag_statistics()
                    if isinstance(gag_final_stats, dict):
                        for key, value in gag_final_stats.items():
                            if isinstance(value, (int, float)):
                                final_training_stats[f'final_gag/{key}'] = value
                
                # Log collector-specific final stats
                if hasattr(collector, 'get_final_statistics'):
                    collector_stats = collector.get_final_statistics()
                    if isinstance(collector_stats, dict):
                        for key, value in collector_stats.items():
                            if isinstance(value, (int, float)):
                                final_training_stats[f'final_collector/{key}'] = value
            
            log_to_wandb(final_training_stats)
            
            # Log training completion milestone
            logging.info(f"Training completed: {learner.train_iter} iterations, {collector.envstep} env steps")
        
        # GAG MuZero specific final logging
        if create_cfg.policy.type == 'gag_muzero':
            if hasattr(policy, 'get_gag_statistics'):
                final_stats = policy.get_gag_statistics()
                logging.info(f"GAG MuZero Training Complete - Final Statistics: {final_stats}")
                
                # Log detailed final GAG statistics
                if use_wandb and isinstance(final_stats, dict):
                    detailed_final_stats = {}
                    for key, value in final_stats.items():
                        if isinstance(value, (int, float)):
                            detailed_final_stats[f'final_detailed_gag/{key}'] = value
                        elif isinstance(value, str):
                            # For string values, log as text
                            logging.info(f"Final GAG {key}: {value}")
                    
                    if detailed_final_stats:
                        log_to_wandb(detailed_final_stats)
            else:
                logging.info("GAG MuZero Training Complete - Statistics method not available")

    # Learner's after_run hook.
    learner.call_hook('after_run')
    
    # Enhanced wandb finalization
    if use_wandb:
        # Log experiment metadata
        if get_rank() == 0:
            experiment_metadata = {
                'experiment/algorithm': create_cfg.policy.type,
                'experiment/total_duration_steps': learner.train_iter,
                'experiment/wandb_freq_used': wandb_freq,
                'experiment/success': 1.0,
            }
            log_to_wandb(experiment_metadata)
        
        wandb.finish()
    
    return policy
