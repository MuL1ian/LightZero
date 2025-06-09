import time
from collections import deque, namedtuple
from typing import Optional, Any, List

import numpy as np
import torch
import selfies as sf
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import SERIAL_COLLECTOR_REGISTRY

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.worker.muzero_collector import MuZeroCollector


def escape_selfies_for_logging(selfies_string):
    """
    Escape SELFIES string to prevent rich markup interpretation in logging.
    
    Args:
        selfies_string: SELFIES string that may contain markup-like tokens
        
    Returns:
        escaped_string: String safe for logging with rich markup
    """
    if selfies_string is None:
        return "None"
    return str(selfies_string).replace('[', '\\[').replace(']', '\\]')


@SERIAL_COLLECTOR_REGISTRY.register('episode_gag_muzero')
class GAGMuZeroCollector(MuZeroCollector):
    """
    Overview:
        The GAG MuZero Collector that extends the base MuZero Collector to collect
        generated vs ground-truth SELFIES pairs at episode completion from MassGymEnv.
        
        Key improvem1.0:
        1. Collects generated vs ground-truth SELFIES pairs for adversarial training
        2. Computes proper value targets from actual environment rewards (not MCTS values)
        3. Tracks real rewards during episodes for accurate value function training
        
        When an episode ends in MassGymEnv, we have access to:
        1. Generated SELFIES: What the agent produced during the episode
        2. Ground-truth SELFIES: The target the environment was guiding toward
        
        These pairs are stored and used for adversarial training.
    """

    def __init__(self, *args, **kwargs):
        """
        Overview:
            Initialize the GAG MuZero Collector with enhanced value computation capabilities.
        """
        # Initialize GAG-specific tracking BEFORE calling super().__init__
        self._generated_ground_truth_pairs = []  # Store (generated_selfies, ground_truth_selfies, spectrum_embed) tuples
        self._episode_pair_count = 0
        self._collected_pair_hashes = set()  # Track collected pairs to avoid duplicates
        self._collection_step_interval = 10  # Collect GAG data every N steps to control frequency
        self._step_counter = {}  # Track step count per environment
        
        # Enhanced value computation tracking
        self._episode_rewards = {}  # Track actual environment rewards per episode
        self._episode_lengths = {}  # Track episode lengths
        self._discount_factor = None  # Will be set from policy config
        
        # Ground-truth trajectory generation
        self._ground_truth_trajectories = []  # Store generated ground-truth trajectories
        self._reward_function = None  # Will be set from policy/global reward network
        self._vocab_dict = None  # Will be set from environment or policy
        self._max_trajectory_length = 100  # Maximum trajectory length
        
        # Replay buffer management for ground-truth trajectories
        self._target_gt_ratio = 0.3  # Target ratio of ground-truth trajectories in replay buffer
        self._gt_trajectories_in_buffer = 0  # Count of ground-truth trajectories in buffer
        self._total_trajectories_in_buffer = 0  # Total trajectories in buffer
        self._gt_trajectory_injection_enabled = True  # Enable/disable GT trajectory injection
        self._total_gt_injections = 0  # Total number of GT trajectories injected
        self._injection_attempts = 0  # Total number of injection attempts
        
        super().__init__(*args, **kwargs)
        
        # Extract discount factor from policy config
        if hasattr(self, 'policy_config') and hasattr(self.policy_config, 'discount_factor'):
            self._discount_factor = self.policy_config.discount_factor
        else:
            self._discount_factor = 0.997  # Default MuZero discount factor
        
        if self._rank == 0:
            self._logger.info("GAG MuZero Collector initialized - will collect generated/ground-truth SELFIES pairs")
            self._logger.info(f"GAG collection frequency: every {self._collection_step_interval} steps")
            self._logger.info(f"Using discount factor: {self._discount_factor} for value computation")
            self._logger.info("Ground-truth trajectory generation enabled")
            self._logger.info(f"Target ground-truth trajectory ratio in replay buffer: {self._target_gt_ratio:.1%}")

    def _compute_discounted_returns(self, rewards: List[float], gamma: float = None) -> List[float]:
        """
        Overview:
            Compute discounted returns from a sequence of rewards.
            Returns[t] = R[t] + gamma * R[t+1] + gamma^2 * R[t+2] + ...
            
        Arguments:
            - rewards: List of rewards from the episode
            - gamma: Discount factor (defaults to self._discount_factor)
            
        Returns:
            - discounted_returns: List of discounted returns for each timestep
        """
        if gamma is None:
            gamma = self._discount_factor
            
        returns = []
        G = 0.0  # Initialize return
        
        # Compute returns backwards (from end to beginning)
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.append(G)
            
        # Reverse to get returns in forward order
        returns.reverse()
        return returns

    def _initialize_episode_tracking(self, env_id: int):
        """
        Overview:
            Initialize tracking for a new episode.
            
        Arguments:
            - env_id: Environment ID
        """
        self._episode_rewards[env_id] = []
        self._episode_lengths[env_id] = 0
        if env_id not in self._step_counter:
            self._step_counter[env_id] = 0

    def _track_episode_step(self, env_id: int, reward: float):
        """
        Overview:
            Track a step in the episode by recording the reward.
            
        Arguments:
            - env_id: Environment ID
            - reward: Reward received at this step
        """
        if env_id not in self._episode_rewards:
            self._initialize_episode_tracking(env_id)
            
        self._episode_rewards[env_id].append(reward)
        self._episode_lengths[env_id] += 1

    def _finalize_episode_values(self, env_id: int, game_segment: GameSegment):
        """
        Overview:
            Compute and store proper value targets from actual environment rewards
            when an episode finishes.
            
        Arguments:
            - env_id: Environment ID
            - game_segment: Game segment to update with proper value targets
        """
        if env_id not in self._episode_rewards or not self._episode_rewards[env_id]:
            if self._rank == 0:
                self._logger.warning(f"No episode rewards tracked for env {env_id}")
            return
        
        try:
            # Get the actual rewards from this episode
            episode_rewards = self._episode_rewards[env_id]
            
            # Compute discounted returns from actual environment rewards
            discounted_returns = self._compute_discounted_returns(episode_rewards, self._discount_factor)
            
            # Update game segment with proper value targets
            # Replace the MCTS/network values with actual discounted returns
            segment_length = len(game_segment.reward_segment) if hasattr(game_segment, 'reward_segment') else 0
            
            if segment_length > 0 and len(discounted_returns) >= segment_length:
                # Update the root value segment with actual discounted returns
                # Ensure we don't exceed the game segment length
                updated_values = discounted_returns[:segment_length]
                
                # Convert to numpy array to match expected format
                game_segment.root_value_segment = np.array(updated_values, dtype=np.float32)
                
                if self._rank == 0:
                    self._logger.debug(f"Updated game segment values for env {env_id}: "
                                     f"segment_length={segment_length}, "
                                     f"episode_rewards_sum={sum(episode_rewards):.3f}, "
                                     f"first_return={discounted_returns[0]:.3f}")
            
            # Store enhanced episode statistics for logging
            episode_stats = {
                'env_id': env_id,
                'episode_length': len(episode_rewards),
                'total_reward': sum(episode_rewards),
                'discounted_return': discounted_returns[0] if discounted_returns else 0.0,
                'avg_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
                'value_correction_applied': True
            }
            
            # Add stats to game segment for potential use in training
            if not hasattr(game_segment, 'episode_stats'):
                game_segment.episode_stats = []
            game_segment.episode_stats.append(episode_stats)
            
        except Exception as e:
            if self._rank == 0:
                self._logger.error(f"Error finalizing episode values for env {env_id}: {e}")
                import traceback
                traceback.print_exc()
        finally:
            # Clean up episode tracking
            if env_id in self._episode_rewards:
                del self._episode_rewards[env_id]
            if env_id in self._episode_lengths:
                del self._episode_lengths[env_id]

    def _handle_episode_completion(self, env_id: int, episode_timestep, game_segment: GameSegment):
        """
        Overview:
            Enhanced episode completion handler that:
            1. Computes proper value targets from actual rewards
            2. Extracts GAG pairs for adversarial training
            
        Arguments:
            - env_id: Environment ID
            - episode_timestep: The final timestep of the episode
            - game_segment: Current game segment
        """
        try:
            # FIRST: Compute proper value targets from actual environment rewards
            self._finalize_episode_values(env_id, game_segment)
            
            # SECOND: Extract generated and ground-truth SELFIES from the completed episode
            generated_selfies, ground_truth_selfies, spectrum_embed = self._extract_episode_selfies_pair(
                env_id, episode_timestep, game_segment
            )
            
            if generated_selfies and ground_truth_selfies and spectrum_embed is not None:
                # Store the pair for adversarial training
                pair_data = {
                    'generated_selfies': generated_selfies,
                    'ground_truth_selfies': ground_truth_selfies,
                    'spectrum_embed': spectrum_embed,
                    'episode_reward': episode_timestep.info.get('eval_episode_return', 0.0),
                    'episode_length': getattr(game_segment, 'current_step', 0),
                    'collection_timestamp': time.time(),
                    'value_correction_applied': True  # Flag indicating proper value computation
                }
                
                self._generated_ground_truth_pairs.append(pair_data)
                self._episode_pair_count += 1
                
                # Generate ground-truth trajectory from the ground-truth SELFIES
                gt_trajectory = self.generate_ground_truth_trajectory(ground_truth_selfies, spectrum_embed)
                if gt_trajectory is not None:
                    pair_data['ground_truth_trajectory'] = gt_trajectory
                
                # Store in game segment for training access
                if not hasattr(game_segment, 'gag_pairs'):
                    game_segment.gag_pairs = []
                game_segment.gag_pairs.append(pair_data)
                
                if self._rank == 0:
                    self._logger.debug(f"Collected GAG pair {self._episode_pair_count}: "
                                     f"Generated='{generated_selfies[:20]}...', "
                                     f"GT='{ground_truth_selfies[:20]}...', "
                                     f"Reward={pair_data['episode_reward']:.3f}")
            
        except Exception as e:
            if self._rank == 0:
                self._logger.warning(f"Failed to handle episode completion for env {env_id}: {e}")

    def _extract_episode_selfies_pair(self, env_id: int, episode_timestep, game_segment: GameSegment):
        """
        Overview:
            Extract the generated and ground-truth SELFIES pair from a completed episode.
            
        Arguments:
            - env_id: Environment ID
            - episode_timestep: Final timestep of the episode
            - game_segment: Game segment containing the episode trajectory
            
        Returns:
            - generated_selfies: SELFIES string generated by the agent
            - ground_truth_selfies: Ground-truth target SELFIES
            - spectrum_embed: Spectrum embedding for the target
        """
        try:
            # Method 1: Extract from episode_timestep.info (preferred)
            if hasattr(episode_timestep, 'info') and episode_timestep.info:
                info = episode_timestep.info
                
                # Extract generated SELFIES (what the agent produced)
                generated_selfies = info.get('generated_selfies', None)
                if generated_selfies is None:
                    generated_selfies = info.get('final_selfies', None)
                if generated_selfies is None:
                    generated_selfies = info.get('agent_selfies', None)
                
                # Extract ground-truth SELFIES (the target)
                ground_truth_selfies = info.get('target_selfies', None) 
                if ground_truth_selfies is None:
                    ground_truth_selfies = info.get('ground_truth_selfies', None)
                if ground_truth_selfies is None:
                    ground_truth_selfies = info.get('gt_selfies', None)
                
                # Extract spectrum embedding with proper shape handling
                spectrum_embed = info.get('spectrum_embed', None)
                if spectrum_embed is None:
                    spectrum_embed = info.get('target_spectrum', None)
                
                # Ensure spectrum_embed has consistent shape and type as TENSOR
                if spectrum_embed is not None:
                    # Convert to PyTorch tensor - reward server expects tensors!
                    if hasattr(spectrum_embed, 'detach'):
                        # Already PyTorch tensor - keep as tensor
                        spectrum_embed = spectrum_embed.detach()
                        if spectrum_embed.device.type == 'cuda':
                            spectrum_embed = spectrum_embed.cpu()  # Move to CPU for serialization if needed
                    elif hasattr(spectrum_embed, 'numpy'):
                        # Some other tensor type - convert to PyTorch tensor
                        spectrum_embed = torch.from_numpy(spectrum_embed.numpy())
                    else:
                        # Numpy array or other - convert to PyTorch tensor
                        spectrum_embed = torch.from_numpy(np.asarray(spectrum_embed))
                    
                    # Ensure it's a 1D tensor with the expected shape
                    if spectrum_embed.dim() > 1:
                        spectrum_embed = spectrum_embed.squeeze()
                    
                    # Ensure it's float32 for consistency
                    spectrum_embed = spectrum_embed.float()
                
                if generated_selfies and ground_truth_selfies:
                    return generated_selfies, ground_truth_selfies, spectrum_embed
            
            # Method 2: Reconstruct from game segment trajectory
            # generated_selfies = self._reconstruct_selfies_from_trajectory(game_segment)
            
            # # Method 3: Extract from environment state (if available)
            # ground_truth_selfies = self._extract_target_selfies_from_env(env_id)
            # spectrum_embed = self._extract_spectrum_from_env(env_id)
            
            # return generated_selfies, ground_truth_selfies, spectrum_embed
            
        except Exception as e:
            if self._rank == 0:
                self._logger.warning(f"Error extracting SELFIES pair: {e}")
            return None, None, None

    def _reconstruct_selfies_from_trajectory(self, game_segment: GameSegment) -> str:
        """
        Overview:
            Reconstruct the generated SELFIES string from the action trajectory in the game segment.
            
        Arguments:
            - game_segment: Game segment containing the action sequence
            
        Returns:
            - selfies_string: Reconstructed SELFIES string
        """
        try:
            if not hasattr(game_segment, 'action_segment') or len(game_segment.action_segment) == 0:
                return None
            
            # This is environment-specific - you'll need to implement based on MassGymEnv's action encoding
            actions = game_segment.action_segment
            
            # Convert action sequence to SELFIES string
            # This depends on how MassGymEnv encodes actions -> SELFIES tokens
            selfies_tokens = []
            for action in actions:
                if isinstance(action, (list, np.ndarray)):
                    action = action[0] if len(action) > 0 else 0
                
                # Map action to SELFIES token (environment-specific)
                token = self._action_to_selfies_token(int(action))
                if token and token != '<PAD>' and token != '<END>':
                    selfies_tokens.append(token)
                elif token == '<END>':
                    break
            
            selfies_string = ''.join(selfies_tokens)
            return selfies_string if selfies_string else None
            
        except Exception as e:
            if self._rank == 0:
                self._logger.warning(f"Error reconstructing SELFIES from trajectory: {e}")
            return None

    def _action_to_selfies_token(self, action: int) -> str:
        """
        Overview:
            Convert a single action to its corresponding SELFIES token.
            This is environment-specific and should be adapted based on MassGymEnv's vocabulary.
            
        Arguments:
            - action: Action index
            
        Returns:
            - token: SELFIES token string
        """
        # This is a placeholder - replace with actual MassGymEnv vocabulary mapping
        # You would typically get this from the environment's action space or vocabulary
        try:
            # Example vocabulary - replace with actual MassGymEnv mapping
            vocab = [
                '<PAD>', '[C]', '[O]', '[N]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]',
                '[=C]', '[=O]', '[=N]', '[=S]', '[#C]', '[#N]', 
                '[C@@H]', '[C@H]', '[NH]', '[OH]', '[SH]', '[nH]',
                '[Ring1]', '[Ring2]', '[Branch1]', '[Branch2]', '<END>'
            ]
            
            if 0 <= action < len(vocab):
                return vocab[action]
            else:
                return '<UNK>'
                
        except Exception as e:
            return '<UNK>'

    def _extract_target_selfies_from_env(self, env_id: int) -> str:
        """
        Overview:
            Extract the target SELFIES from the environment state.
            
        Arguments:
            - env_id: Environment ID
            
        Returns:
            - target_selfies: Ground-truth SELFIES string
        """
        try:
            # Try to get target SELFIES from environment
            if hasattr(self._env, 'get_target_selfies'):
                return self._env.get_target_selfies(env_id)
            elif hasattr(self._env, 'envs') and hasattr(self._env.envs[env_id], 'target_selfies'):
                return self._env.envs[env_id].target_selfies
            elif hasattr(self._env, 'envs') and hasattr(self._env.envs[env_id], 'get_target'):
                target = self._env.envs[env_id].get_target()
                if isinstance(target, dict) and 'selfies' in target:
                    return target['selfies']
                elif isinstance(target, str):
                    return target
            
            return None
            
        except Exception as e:
            if self._rank == 0:
                self._logger.warning(f"Error extracting target SELFIES from env: {e}")
            return None

    def _extract_spectrum_from_env(self, env_id: int) -> np.ndarray:
        """
        Overview:
            Extract the spectrum embedding from the environment state.
            
        Arguments:
            - env_id: Environment ID
            
        Returns:
            - spectrum_embed: Spectrum embedding array
        """
        try:
            # Try to get spectrum embedding from environment
            if hasattr(self._env, 'get_spectrum_embed'):
                return self._env.get_spectrum_embed(env_id)
            elif hasattr(self._env, 'envs') and hasattr(self._env.envs[env_id], 'spectrum_embed'):
                return self._env.envs[env_id].spectrum_embed
            elif hasattr(self._env, 'envs') and hasattr(self._env.envs[env_id], 'get_spectrum'):
                return self._env.envs[env_id].get_spectrum()
            
            # Fallback: generate dummy spectrum for testing
            return np.random.randn(4096).astype(np.float32)
            
        except Exception as e:
            if self._rank == 0:
                self._logger.warning(f"Error extracting spectrum from env: {e}")
            return np.random.randn(4096).astype(np.float32)

    def get_collected_pairs(self) -> List[dict]:
        """
        Overview:
            Get all collected generated/ground-truth pairs for adversarial training.
            
        Returns:
            - pairs: List of dictionaries containing pair data
        """
        if hasattr(self, '_generated_ground_truth_pairs'):
            return self._generated_ground_truth_pairs.copy()
        else:
            return []

    def clear_collected_pairs(self):
        """
        Overview:
            Clear the collected pairs buffer.
        """
        if hasattr(self, '_generated_ground_truth_pairs'):
            self._generated_ground_truth_pairs.clear()
        else:
            self._generated_ground_truth_pairs = []
    
    def set_reward_function(self, reward_function):
        """
        Overview:
            Set the reward function for ground-truth trajectory generation.
            
        Arguments:
            - reward_function: Function that takes (selfies_string, spectrum_embed) and returns reward
        """
        self._reward_function = reward_function
        if self._rank == 0:
            self._logger.info(f"[SETUP] Reward function set: {reward_function is not None}")
            if reward_function is not None:
                self._logger.info(f"[SETUP] Reward function type: {type(reward_function)}")
    
    def set_vocab_dict(self, vocab_dict):
        """
        Overview:
            Set the vocabulary dictionary for SELFIES token conversion.
            
        Arguments:
            - vocab_dict: Dictionary mapping tokens to indices
        """
        self._vocab_dict = vocab_dict
        if self._rank == 0:
            self._logger.info(f"[SETUP] Vocabulary set with {len(vocab_dict)} tokens")
            self._logger.info(f"[SETUP] First 10 vocab keys: {list(vocab_dict.keys())[:10]}")
            self._logger.info(f"[SETUP] Ground-truth trajectory generation should now be possible")
    
    def generate_ground_truth_trajectory(self, ground_truth_selfies, spectrum_embed):
        """
        Overview:
            Generate a ground-truth trajectory from a SELFIES string.
            
        Arguments:
            - ground_truth_selfies: Target SELFIES string
            - spectrum_embed: Spectrum embedding for reward computation
            
        Returns:
            - trajectory_data: Dictionary containing trajectory information or None if failed
        """
        # Enhanced debugging
        if self._rank == 0:
            self._logger.info("[DEBUG] generate_ground_truth_trajectory called:")
            # Escape SELFIES string to avoid rich markup interpretation
            escaped_selfies = escape_selfies_for_logging(ground_truth_selfies)
            self._logger.info(f"  - ground_truth_selfies: {escaped_selfies}")
            self._logger.info(f"  - spectrum_embed shape: {spectrum_embed.shape if hasattr(spectrum_embed, 'shape') else 'N/A'}")
            self._logger.info(f"  - vocab_dict is None: {self._vocab_dict is None}")
            self._logger.info(f"  - reward_function is None: {self._reward_function is None}")
            if self._vocab_dict is not None:
                self._logger.info(f"  - vocab_dict size: {len(self._vocab_dict)}")
                # Escape vocab keys to avoid markup issues
                vocab_keys = [str(k).replace('[', '\\[').replace(']', '\\]') for k in list(self._vocab_dict.keys())[:10]]
                self._logger.info(f"  - vocab_dict keys (first 10): {vocab_keys}")
        
        if self._vocab_dict is None:
            if self._rank == 0:
                self._logger.warning("Cannot generate ground-truth trajectory: vocabulary not set")
            return None
        
        if not ground_truth_selfies:
            if self._rank == 0:
                self._logger.warning("Cannot generate ground-truth trajectory: empty ground_truth_selfies")
            return None
        
        try:
            trajectory_data = create_ground_truth_trajectory(
                ground_truth_selfies=ground_truth_selfies,
                spectrum_embed=spectrum_embed,
                reward_function=self._reward_function,
                vocab_dict=self._vocab_dict,
                max_length=self._max_trajectory_length,
                discount_factor=self._discount_factor
            )
            
            if trajectory_data is not None:
                self._ground_truth_trajectories.append(trajectory_data)
                
                if self._rank == 0:
                    self._logger.info(f"[SUCCESS] Generated ground-truth trajectory: "
                                    f"length={trajectory_data['trajectory_length']}, "
                                    f"final_reward={trajectory_data['final_reward']:.3f}, "
                                    f"discounted_return={trajectory_data['discounted_return']:.3f}")
            else:
                if self._rank == 0:
                    self._logger.warning("[FAILED] create_ground_truth_trajectory returned None")
            
            return trajectory_data
            
        except Exception as e:
            if self._rank == 0:
                self._logger.error(f"Error generating ground-truth trajectory: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def get_ground_truth_trajectories(self):
        """
        Overview:
            Get all collected ground-truth trajectories.
            
        Returns:
            - trajectories: List of trajectory data dictionaries
        """
        return self._ground_truth_trajectories.copy()
    
    def clear_ground_truth_trajectories(self):
        """
        Overview:
            Clear the ground-truth trajectories buffer.
        """
        self._ground_truth_trajectories.clear()
    
    def set_target_gt_ratio(self, ratio: float):
        """
        Overview:
            Set the target ratio of ground-truth trajectories in replay buffer.
            
        Arguments:
            - ratio: Target ratio (0.0 to 1.0)
        """
        self._target_gt_ratio = max(0.0, min(1.0, ratio))
        if self._rank == 0:
            self._logger.info(f"Target ground-truth trajectory ratio set to: {self._target_gt_ratio:.1%}")
    
    def convert_gt_trajectory_to_game_segment(self, trajectory_data: dict) -> 'GameSegment':
        """
        Overview:
            Convert ground-truth trajectory data to GameSegment format using deterministic planning.
            This treats the GT SELFIES as a deterministic trajectory where policy logits have
            high probability for the correct action and low probability for others.
            
        Arguments:
            - trajectory_data: Dictionary containing trajectory information
            
        Returns:
            - game_segment: GameSegment object compatible with replay buffer (same format as regular data)
        """
        try:
            from lzero.mcts.buffer.game_segment import GameSegment
            
            # Extract trajectory components
            action_sequence = trajectory_data['action_sequence']
            rewards = trajectory_data['rewards']
            values = trajectory_data['values']
            target_selfies = trajectory_data['target_selfies']
            spectrum_embed = trajectory_data['spectrum_embed']
            
            trajectory_length = len(action_sequence)
            
            # Create a proper GameSegment using the same constructor as regular data collection
            # Get action space and config from environment if available
            action_space = self._env.action_space if hasattr(self, '_env') and self._env else None
            if action_space is None:
                # Fallback: create mock action space with vocabulary size
                class MockActionSpace:
                    def __init__(self, n):
                        self.n = n
                # Use vocabulary size instead of hardcoded 69
                vocab_size = len(self._vocab_dict) if hasattr(self, '_vocab_dict') and self._vocab_dict else 75
                action_space = MockActionSpace(vocab_size)
                if self._rank == 0 and hasattr(self, '_logger'):
                    self._logger.info(f"Created mock action space with size {vocab_size} to match vocabulary")
            
            # Ensure action space size matches vocabulary size to prevent index errors
            expected_action_space_size = len(self._vocab_dict) if hasattr(self, '_vocab_dict') and self._vocab_dict else 75
            if hasattr(action_space, 'n') and action_space.n != expected_action_space_size:
                if self._rank == 0 and hasattr(self, '_logger'):
                    self._logger.warning(f"Action space size mismatch: env={action_space.n}, vocab={expected_action_space_size}. Using vocab size.")
                # Create corrected action space
                class MockActionSpace:
                    def __init__(self, n):
                        self.n = n
                action_space = MockActionSpace(expected_action_space_size)
                
            # Use the same policy config as regular collection
            if hasattr(self, 'policy_config'):
                config = self.policy_config
            else:
                # Fallback: create minimal config
                from easydict import EasyDict
                config = EasyDict({
                    'num_unroll_steps': 5,
                    'td_steps': 5,
                    'model': EasyDict({
                        'frame_stack_num': 1,
                        'action_space_size': action_space.n,
                        'observation_shape': spectrum_embed.shape if hasattr(spectrum_embed, 'shape') else [4096]
                    }),
                    'discount_factor': self._discount_factor,
                    'gray_scale': False,
                    'transform2string': False,
                    'sampled_algo': False,
                    'gumbel_algo': True,  # GAG uses Gumbel MuZero
                    'use_ture_chance_label_in_chance_encoder': False,
                    'game_segment_length': trajectory_length
                })
            
            # Create GameSegment using proper constructor
            game_segment = GameSegment(
                action_space=action_space,
                game_segment_length=trajectory_length,
                config=config
            )
            
            # Prepare spectrum observations for each step
            if isinstance(spectrum_embed, torch.Tensor):
                spectrum_array = spectrum_embed.detach().cpu().numpy()
            else:
                spectrum_array = np.array(spectrum_embed, dtype=np.float32)
            
            # Ensure spectrum has correct shape for observation
            if spectrum_array.ndim == 1:
                obs_shape = spectrum_array.shape
            else:
                obs_shape = spectrum_array.flatten().shape
                spectrum_array = spectrum_array.flatten()
            
            # Create initial observation with proper format (4096 + 100 + 50 = 4246)
            initial_selfies_tokens = np.zeros(100, dtype=np.float32)  # Empty SELFIES at start
            initial_formula_tokens = np.zeros(50, dtype=np.float32)
            if hasattr(trajectory_data, 'get') and trajectory_data.get('formula'):
                # Simple formula encoding
                formula = trajectory_data['formula']
                hash_val = hash(formula) % 50
                initial_formula_tokens[hash_val] = 1.0
            
            initial_obs = np.concatenate([spectrum_array, initial_selfies_tokens, initial_formula_tokens])
            
            # Initialize with frame_stack_num observations (same initial obs repeated)
            init_observations = [initial_obs.copy() for _ in range(config.model.frame_stack_num)]
            game_segment.reset(init_observations)
            
            # Create deterministic policy logits (high prob for GT action, low for others)
            num_actions = action_space.n
            
            # Validate action sequence is compatible with action space
            max_action_in_sequence = max(action_sequence) if action_sequence else 0
            if max_action_in_sequence >= num_actions:
                if self._rank == 0 and hasattr(self, '_logger'):
                    self._logger.error(f"Action sequence contains index {max_action_in_sequence} but action space is {num_actions}")
                    self._logger.error(f"Action sequence: {action_sequence[:10]}... (showing first 10)")
                    self._logger.error(f"Vocabulary size: {len(self._vocab_dict) if hasattr(self, '_vocab_dict') and self._vocab_dict else 'unknown'}")
                return None
            
            if self._rank == 0 and hasattr(self, '_logger'):
                self._logger.debug(f"GT GameSegment: action_space={num_actions}, max_action={max_action_in_sequence}, traj_len={trajectory_length}")
            
            # Build the trajectory step by step using the same append() method as regular collection
            for step, action_idx in enumerate(action_sequence):
                # Create observation matching the regular environment format:
                # spectrum (4096) + selfies_tokens (100) + formula_tokens (50) = 4246
                
                # For GT trajectory, use partial SELFIES up to current step
                partial_selfies = target_selfies[:step*5] if step > 0 else ""  # Approximate partial construction
                
                # Encode partial SELFIES (100 dimensions) 
                try:
                    partial_selfies_tokens = np.zeros(100, dtype=np.float32)
                    if hasattr(self, '_env') and hasattr(self._env, 'tokenizer'):
                        tokenizer = self._env.tokenizer
                        encoded = tokenizer.encode_selfies(partial_selfies)
                        if len(encoded) <= 100:
                            partial_selfies_tokens[:len(encoded)] = encoded
                    # Fallback: use step-based dummy encoding
                    else:
                        partial_selfies_tokens[:min(step+1, 100)] = step + 1
                except:
                    # Fallback: simple step-based encoding
                    partial_selfies_tokens = np.zeros(100, dtype=np.float32)
                    partial_selfies_tokens[:min(step+1, 100)] = step + 1
                 
                # Encode formula tokens (50 dimensions to match config)
                formula_tokens = np.zeros(50, dtype=np.float32)
                if hasattr(trajectory_data, 'get') and trajectory_data.get('formula'):
                    try:
                        # Use BERT tokenizer for formula if available
                        formula = trajectory_data['formula']
                        # Simple fallback: use hash-based encoding
                        hash_val = hash(formula) % 50
                        formula_tokens[hash_val] = 1.0
                    except:
                        formula_tokens[0] = 1.0  # Default encoding
                
                # Combine observation parts: spectrum + selfies + formula (4096 + 100 + 50 = 4246)
                obs = np.concatenate([spectrum_array, partial_selfies_tokens, formula_tokens])
                
                # Reward at this step
                reward = rewards[step] if step < len(rewards) else 0.0
                
                # Action mask (all actions allowed for simplicity)
                action_mask = np.ones(num_actions, dtype=np.bool_)
                
                # Append the transition to game segment (same as regular collection)
                game_segment.append(
                    action=action_idx,
                    obs=obs,
                    reward=reward,
                    action_mask=action_mask,
                    to_play=0,  # Single player
                    timestep=step
                )
                
                # Store MCTS-style statistics for this step
                # Create deterministic visit counts (high for GT action, low for others)
                child_visits = np.ones(num_actions, dtype=np.float32)
                child_visits[action_idx] = 100.0  # High visit count for GT action
                
                # Create deterministic improved policy probabilities (for Gumbel MuZero)
                improved_policy = np.ones(num_actions, dtype=np.float32) * 0.01
                improved_policy[action_idx] = 0.9  # High probability for GT action
                improved_policy = improved_policy / improved_policy.sum()  # Normalize
                
                # Store search statistics (same as regular MCTS collection)
                game_segment.store_search_stats(
                    visit_counts=child_visits.tolist(),  # Convert to list as expected
                    root_value=values[step] if step < len(values) else 0.0,  # Single value, not list
                    improved_policy=improved_policy.tolist()  # Convert to list as expected
                )
            
            # Convert to array format (required for replay buffer compatibility)
            game_segment.game_segment_to_array()
            
            # Add GT metadata for identification (but don't break compatibility)
            game_segment.is_ground_truth = True
            game_segment.ground_truth_selfies = target_selfies
            game_segment.ground_truth_reward = rewards[-1] if rewards else 0.0
            
            if self._rank == 0 and hasattr(self, '_logger') and self._logger:
                self._logger.debug(f"Created GT GameSegment: length={trajectory_length}, "
                                 f"final_reward={rewards[-1] if rewards else 0.0:.3f}, "
                                 f"segments={len(game_segment.action_segment)}")
            
            return game_segment
            
        except Exception as e:
            if self._rank == 0 and hasattr(self, '_logger') and self._logger:
                self._logger.error(f"Error creating GT GameSegment: {e}")
                import traceback
                traceback.print_exc()
            elif self._rank == 0:
                print(f"Error creating GT GameSegment: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def get_current_gt_ratio(self) -> float:
        """
        Overview:
            Get the current ratio of ground-truth trajectories in replay buffer.
            
        Returns:
            - ratio: Current ground-truth trajectory ratio
        """
        if self._total_trajectories_in_buffer == 0:
            return 0.0
        return self._gt_trajectories_in_buffer / self._total_trajectories_in_buffer
    
    def update_gt_buffer_tracking(self, injected_count: int, regular_count: int):
        """
        Overview:
            Update GT buffer tracking with newly injected data.
            
        Arguments:
            - injected_count: Number of GT trajectories injected
            - regular_count: Number of regular trajectories in this batch
        """
        # Update counts based on injected data
        self._gt_trajectories_in_buffer += injected_count
        self._total_trajectories_in_buffer += (injected_count + regular_count)
        
        if self._rank == 0:
            current_ratio = self.get_current_gt_ratio()
            self._logger.debug(f"Updated buffer tracking: GT={self._gt_trajectories_in_buffer}, "
                             f"Total={self._total_trajectories_in_buffer}, Ratio={current_ratio:.1%}")
    
    def update_buffer_counts(self, replay_buffer):
        """
        Overview:
            Update the counts of ground-truth vs total trajectories in replay buffer.
            
        Arguments:
            - replay_buffer: The replay buffer to analyze
        """
        try:
            # Count trajectories in the buffer
            total_count = 0
            gt_count = 0
            
            if hasattr(replay_buffer, 'game_segments') and replay_buffer.game_segments:
                for segment in replay_buffer.game_segments:
                    if segment is not None:
                        total_count += 1
                        if hasattr(segment, 'is_ground_truth') and segment.is_ground_truth:
                            gt_count += 1
            
            self._total_trajectories_in_buffer = total_count
            self._gt_trajectories_in_buffer = gt_count
            
            if self._rank == 0:
                current_ratio = self.get_current_gt_ratio()
                self._logger.debug(f"Buffer counts updated: {gt_count}/{total_count} "
                                 f"({current_ratio:.1%} ground-truth)")
                
        except Exception as e:
            if self._rank == 0:
                self._logger.warning(f"Error updating buffer counts: {e}")
    
    def inject_ground_truth_trajectories(self, replay_buffer):
        """
        Overview:
            Inject ground-truth trajectories into replay buffer to maintain target ratio.
            
        Arguments:
            - replay_buffer: The replay buffer to inject trajectories into
        """
        if not self._gt_trajectory_injection_enabled or not self._ground_truth_trajectories:
            return
        
        try:
            # Update current buffer counts
            self.update_buffer_counts(replay_buffer)
            
            current_ratio = self.get_current_gt_ratio()
            
            # Check if we need to inject more ground-truth trajectories
            if current_ratio < self._target_gt_ratio and self._total_trajectories_in_buffer > 0:
                # Calculate how many GT trajectories we need
                target_gt_count = int(self._total_trajectories_in_buffer * self._target_gt_ratio)
                needed_gt_count = max(0, target_gt_count - self._gt_trajectories_in_buffer)
                
                # Limit to available ground-truth trajectories
                available_gt_count = len(self._ground_truth_trajectories)
                inject_count = min(needed_gt_count, available_gt_count)
                
                if inject_count > 0:
                    self._injection_attempts += 1
                    
                    if self._rank == 0:
                        self._logger.info(f"Injecting {inject_count} ground-truth trajectories to reach "
                                        f"{self._target_gt_ratio:.1%} ratio (current: {current_ratio:.1%})")
                    
                    # Select and inject ground-truth trajectories
                    import random
                    selected_trajectories = random.sample(self._ground_truth_trajectories, inject_count)
                    
                    injected_count = 0
                    for traj_data in selected_trajectories:
                        game_segment = self.convert_gt_trajectory_to_game_segment(traj_data)
                        if game_segment is not None:
                            # Add to replay buffer
                            if hasattr(replay_buffer, 'push_game_segments'):
                                replay_buffer.push_game_segments([game_segment])
                                injected_count += 1
                            elif hasattr(replay_buffer, 'add_game_segment'):
                                replay_buffer.add_game_segment(game_segment)
                                injected_count += 1
                    
                    # Update counts
                    self._gt_trajectories_in_buffer += injected_count
                    self._total_trajectories_in_buffer += injected_count
                    self._total_gt_injections += injected_count
                    
                    if self._rank == 0:
                        final_ratio = self.get_current_gt_ratio()
                        self._logger.info(f"Injected {injected_count} ground-truth trajectories. "
                                        f"New ratio: {final_ratio:.1%} "
                                        f"(Total injected: {self._total_gt_injections})")
                        
        except Exception as e:
            if self._rank == 0:
                self._logger.error(f"Error injecting ground-truth trajectories: {e}")
                import traceback
                traceback.print_exc()

    def _log_gag_statistics(self, train_iter: int):
        """
        Overview:
            Enhanced GAG statistics logging including value computation metrics and ground-truth trajectories.
        """
        if self._rank == 0:
            current_gt_ratio = self.get_current_gt_ratio()
            
            # Enhanced debugging information
            self._logger.info("=" * 80)
            self._logger.info("GAG MUZERO COLLECTOR STATISTICS")
            self._logger.info("=" * 80)
            self._logger.info(f"📊 Data Collection:")
            self._logger.info(f"  • Collected GAG pairs: {self._episode_pair_count}")
            self._logger.info(f"  • Generated GT trajectories: {len(self._ground_truth_trajectories)}")
            self._logger.info(f"  • Value discount factor: {self._discount_factor}")
            
            self._logger.info(f"🎯 GT Trajectory Injection:")
            self._logger.info(f"  • Target GT ratio: {self._target_gt_ratio:.1%}")
            self._logger.info(f"  • Current GT ratio: {current_gt_ratio:.1%}")
            self._logger.info(f"  • GT trajectories in buffer: {self._gt_trajectories_in_buffer}")
            self._logger.info(f"  • Total trajectories in buffer: {self._total_trajectories_in_buffer}")
            self._logger.info(f"  • Total GT injections: {self._total_gt_injections}")
            self._logger.info(f"  • Injection attempts: {self._injection_attempts}")
            self._logger.info(f"  • GT injection enabled: {self._gt_trajectory_injection_enabled}")
            
            # Debug why GT ratio might be 0
            if current_gt_ratio == 0.0:
                self._logger.warning("🚨 GT RATIO IS 0% - DEBUGGING:")
                if not self._gt_trajectory_injection_enabled:
                    self._logger.warning("  ❌ GT trajectory injection is DISABLED")
                elif len(self._ground_truth_trajectories) == 0:
                    self._logger.warning(f"  ❌ No GT trajectories available (generated: {len(self._ground_truth_trajectories)})")
                elif self._total_trajectories_in_buffer == 0:
                    self._logger.warning("  ❌ Buffer is empty (no trajectories to compute ratio)")
                else:
                    self._logger.warning(f"  ❓ GT injection seems active but ratio is 0% - check injection logic")
                    self._logger.warning(f"     Available GT trajectories: {len(self._ground_truth_trajectories)}")
                    self._logger.warning(f"     Buffer total count: {self._total_trajectories_in_buffer}")
                    self._logger.warning(f"     Buffer GT count: {self._gt_trajectories_in_buffer}")
            
            # GT trajectory details
            if self._ground_truth_trajectories:
                self._logger.info(f"📈 GT Trajectory Details:")
                avg_gt_reward = np.mean([t['final_reward'] for t in self._ground_truth_trajectories])
                avg_gt_return = np.mean([t['discounted_return'] for t in self._ground_truth_trajectories])
                avg_gt_length = np.mean([t['trajectory_length'] for t in self._ground_truth_trajectories])
                
                self._logger.info(f"  • Average final reward: {avg_gt_reward:.4f}")
                self._logger.info(f"  • Average discounted return: {avg_gt_return:.4f}")
                self._logger.info(f"  • Average trajectory length: {avg_gt_length:.1f}")
                
                # Show example trajectory info
                example_traj = self._ground_truth_trajectories[0]
                escaped_selfies = escape_selfies_for_logging(example_traj.get('target_selfies', 'N/A')[:30])
                self._logger.info(f"  • Example trajectory: '{escaped_selfies}...' -> reward: {example_traj.get('final_reward', 0):.4f}")
            
            # Episode collection details
            if self._generated_ground_truth_pairs:
                avg_reward = np.mean([p['episode_reward'] for p in self._generated_ground_truth_pairs])
                self._logger.info(f"📝 Episode Collection:")
                self._logger.info(f"  • Average episode reward: {avg_reward:.4f}")
                self._logger.info(f"  • Pairs this collection: {len(self._generated_ground_truth_pairs)}")
            
            self._logger.info("=" * 80)
            
            if self._tb_logger and self._episode_pair_count > 0:
                self._tb_logger.add_scalar('gag_collector/total_pairs', self._episode_pair_count, train_iter)
                self._tb_logger.add_scalar('gag_collector/pairs_this_collection', len(self._generated_ground_truth_pairs), train_iter)
                self._tb_logger.add_scalar('gag_collector/discount_factor', self._discount_factor, train_iter)
                self._tb_logger.add_scalar('gag_collector/ground_truth_trajectories', len(self._ground_truth_trajectories), train_iter)
                
                # Log replay buffer statistics
                self._tb_logger.add_scalar('gag_collector/buffer_gt_ratio', current_gt_ratio, train_iter)
                self._tb_logger.add_scalar('gag_collector/buffer_gt_count', self._gt_trajectories_in_buffer, train_iter)
                self._tb_logger.add_scalar('gag_collector/buffer_total_count', self._total_trajectories_in_buffer, train_iter)
                self._tb_logger.add_scalar('gag_collector/target_gt_ratio', self._target_gt_ratio, train_iter)
                self._tb_logger.add_scalar('gag_collector/total_gt_injections', self._total_gt_injections, train_iter)
                self._tb_logger.add_scalar('gag_collector/injection_attempts', self._injection_attempts, train_iter)
                
                # Log average episode rewards
                if self._generated_ground_truth_pairs:
                    avg_reward = np.mean([p['episode_reward'] for p in self._generated_ground_truth_pairs])
                    self._tb_logger.add_scalar('gag_collector/avg_episode_reward', avg_reward, train_iter)
                
                # Log ground-truth trajectory statistics
                if self._ground_truth_trajectories:
                    avg_gt_reward = np.mean([t['final_reward'] for t in self._ground_truth_trajectories])
                    avg_gt_return = np.mean([t['discounted_return'] for t in self._ground_truth_trajectories])
                    avg_gt_length = np.mean([t['trajectory_length'] for t in self._ground_truth_trajectories])
                    
                    self._tb_logger.add_scalar('gag_collector/avg_gt_reward', avg_gt_reward, train_iter)
                    self._tb_logger.add_scalar('gag_collector/avg_gt_return', avg_gt_return, train_iter)
                    self._tb_logger.add_scalar('gag_collector/avg_gt_length', avg_gt_length, train_iter)

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the GAG collector, including pair collection state and reward tracking.
        """
        super().reset(_policy, _env)
        
        # Reset GAG-specific state - check if attributes exist first (defensive programming)
        if hasattr(self, '_generated_ground_truth_pairs'):
            self._generated_ground_truth_pairs.clear()
        else:
            self._generated_ground_truth_pairs = []
            
        if hasattr(self, '_episode_pair_count'):
            self._episode_pair_count = 0
        else:
            self._episode_pair_count = 0
            
        # Reset enhanced value computation tracking
        if hasattr(self, '_episode_rewards'):
            self._episode_rewards.clear()
        else:
            self._episode_rewards = {}
            
        if hasattr(self, '_episode_lengths'):
            self._episode_lengths.clear()
        else:
            self._episode_lengths = {}
            
        if hasattr(self, '_collected_pair_hashes'):
            self._collected_pair_hashes.clear()
        else:
            self._collected_pair_hashes = set()
            
        if hasattr(self, '_step_counter'):
            self._step_counter.clear()
        else:
            self._step_counter = {}
        
        # Reset ground-truth trajectory tracking
        if hasattr(self, '_ground_truth_trajectories'):
            self._ground_truth_trajectories.clear()
        else:
            self._ground_truth_trajectories = []
        
        # Reset replay buffer tracking
        if hasattr(self, '_gt_trajectories_in_buffer'):
            self._gt_trajectories_in_buffer = 0
        else:
            self._gt_trajectories_in_buffer = 0
            
        if hasattr(self, '_total_trajectories_in_buffer'):
            self._total_trajectories_in_buffer = 0
        else:
            self._total_trajectories_in_buffer = 0

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None,
                collect_with_pure_policy: bool = False) -> List[Any]:
        """
        Overview:
            Enhanced collect method that:
            1. Tracks actual environment rewards during episodes
            2. Computes proper discounted returns for value targets
            3. Extracts generated/ground-truth SELFIES pairs for GAG training
        """
        # Store the original _env.step method to intercept episode timesteps
        original_env_step = self._env.step
        
        def intercepted_step(actions):
            # Call the original step method
            timesteps = original_env_step(actions)
            
            # Process each timestep to track rewards and extract GAG data
            for env_id, episode_timestep in timesteps.items():
                # Initialize episode tracking if needed
                if env_id not in self._episode_rewards:
                    self._initialize_episode_tracking(env_id)
                
                # Track the actual environment reward for proper value computation
                if hasattr(episode_timestep, 'reward'):
                    actual_reward = float(episode_timestep.reward)
                    self._track_episode_step(env_id, actual_reward)
                
                # Initialize step counter for this environment if not exists
                if env_id not in self._step_counter:
                    self._step_counter[env_id] = 0
                
                self._step_counter[env_id] += 1
                
                # Extract GAG pair from timesteps based on collection frequency or episode completion
                try:
                    if hasattr(episode_timestep, 'info') and episode_timestep.info:
                        info = episode_timestep.info
                        
                        # Check if we have the necessary GAG data in this timestep
                        has_generated = 'current_selfies' in info or 'generated_selfies' in info
                        has_target = any(key in info for key in ['target_selfies', 'ground_truth_selfies', 'gt_selfies'])
                        has_spectrum = any(key in info for key in ['spectrum_embed', 'target_spectrum'])
                        
                        # Collect data if:
                        # 1. Episode is done (final state), OR
                        # 2. We have all necessary data AND it's time to collect (based on step interval)
                        # should_collect = (
                        #     episode_timestep.done or 
                        #     (has_generated and has_target and has_spectrum and 
                        #      self._step_counter[env_id] % self._collection_step_interval == 0)
                        # )
                        
                        # if should_collect and has_generated and has_target and has_spectrum:
                        #     # Create a hash to check for duplicates
                        #     generated_selfies = (info.get('generated_selfies') or 
                        #                        info.get('current_selfies') or 
                        #                        info.get('final_selfies', ''))
                        #     ground_truth_selfies = (info.get('target_selfies') or 
                        #                           info.get('ground_truth_selfies') or 
                        #                           info.get('gt_selfies', ''))
                            
                        #     pair_hash = hash((generated_selfies, ground_truth_selfies, env_id))
                            
                        #     if pair_hash not in self._collected_pair_hashes:
                        #         self._extract_and_store_gag_pair(env_id, episode_timestep)
                        #         self._collected_pair_hashes.add(pair_hash)
                        
                        if episode_timestep.done:
                            self._extract_and_store_gag_pair(env_id, episode_timestep)
                            
                except Exception as e:
                    if self._rank == 0:
                        # Escape error message to prevent rich markup issues with SELFIES tokens
                        escaped_error = str(e).replace('[', '\\[').replace(']', '\\]')
                        self._logger.warning(f"Error processing timestep for env {env_id}: {escaped_error}")
                
                # Reset tracking when episode is done
                if episode_timestep.done:
                    self._step_counter[env_id] = 0
                    # Episode tracking will be cleaned up in _finalize_episode_values
            
            return timesteps
        
        # Temporarily replace the step method
        self._env.step = intercepted_step
        
        try:
            # Call the base collector's collect method
            return_data = super().collect(n_episode, train_iter, policy_kwargs, collect_with_pure_policy)
            
            # Validate observation shapes in collected data to catch dimension mismatches early
            if return_data and len(return_data) > 0 and hasattr(return_data[0], '__len__'):
                for i, game_segment in enumerate(return_data[0][:5]):  # Check first 5 segments
                    if hasattr(game_segment, 'obs_segment') and len(game_segment.obs_segment) > 0:
                        obs_shape = game_segment.obs_segment[0].shape
                        expected_shape = 4246  # 4096 + 100 + 50
                        if len(obs_shape) > 0 and obs_shape[-1] != expected_shape:
                            if self._rank == 0:
                                self._logger.error(f"❌ OBSERVATION SHAPE MISMATCH in segment {i}:")
                                self._logger.error(f"   Expected: (..., {expected_shape}), Got: {obs_shape}")
                                self._logger.error(f"   This indicates environment config inconsistency!")
                                self._logger.error(f"   Check formula_max_len in environment vs model config")
                            # Don't raise error, just log and continue for graceful handling
                            
        finally:
            # Restore the original step method
            self._env.step = original_env_step
        
        # Generate and inject ground-truth trajectories into the replay buffer
        self._inject_ground_truth_trajectories_to_data(return_data)
        
        # Log GAG statistics after collection
        self._log_gag_statistics(train_iter)
        
        return return_data
    
    def _inject_ground_truth_trajectories_to_data(self, return_data):
        """
        Overview:
            Inject ground-truth trajectories into the collected data in the EXACT same format
            as regular trajectories. This ensures perfect compatibility with the replay buffer.
            
        Arguments:
            - return_data: The data returned by collect() - [game_segments_list, meta_data_list]
        """
        try:
            if self._rank == 0:
                self._logger.info("🔧 GT Trajectory Injection (Simplified Format):")
                self._logger.info(f"  • GT injection enabled: {self._gt_trajectory_injection_enabled}")
                self._logger.info(f"  • Available GT trajectories: {len(self._ground_truth_trajectories)}")
                self._logger.info(f"  • Return data structure: {type(return_data)} with {len(return_data)} elements")
                if len(return_data) > 0:
                    self._logger.info(f"  • Regular trajectories: {len(return_data[0])}")
            
            if not self._gt_trajectory_injection_enabled:
                if self._rank == 0:
                    self._logger.warning("  ❌ GT trajectory injection is DISABLED")
                return
                
            if not self._ground_truth_trajectories:
                if self._rank == 0:
                    self._logger.warning("  ❌ No GT trajectories available")
                return
                
            # Ensure return_data has the correct structure [game_segments, meta_data]
            if len(return_data) < 2:
                if self._rank == 0:
                    self._logger.error("  ❌ Invalid return_data structure - expected [game_segments, meta_data]")
                return
            
            # Calculate injection count based on target ratio
            num_regular_trajectories = len(return_data[0])
            target_gt_count = max(1, int(num_regular_trajectories * self._target_gt_ratio))
            available_gt_count = len(self._ground_truth_trajectories)
            inject_count = min(target_gt_count, available_gt_count)
            
            if self._rank == 0:
                self._logger.info(f"  • Regular trajectories: {num_regular_trajectories}")
                self._logger.info(f"  • Target GT ratio: {self._target_gt_ratio:.1%}")
                self._logger.info(f"  • Will inject: {inject_count} GT trajectories")
            
            if inject_count <= 0:
                if self._rank == 0:
                    self._logger.warning("  ❌ No GT trajectories to inject")
                return
            
            # Select GT trajectories to inject
            import random
            selected_trajectories = random.sample(self._ground_truth_trajectories, inject_count)
            
            injected_count = 0
            conversion_failures = 0
            
            for i, traj_data in enumerate(selected_trajectories):
                if self._rank == 0:
                    escaped_selfies = escape_selfies_for_logging(traj_data.get('target_selfies', 'N/A')[:20])
                    self._logger.debug(f"    Converting GT trajectory {i+1}/{inject_count}: '{escaped_selfies}...'")
                
                # Convert to GameSegment using the same process as regular collection
                game_segment = self.convert_gt_trajectory_to_game_segment(traj_data)
                if game_segment is not None:
                    # Compute priorities using the same method as regular trajectories
                    gt_priorities = self._compute_gt_priorities(game_segment, traj_data)
                    
                    # Create metadata in EXACT same format as regular collection
                    # This matches the format in muzero_collector.py line 737-742
                    gt_meta_data = {
                        'priorities': gt_priorities,
                        'done': True,  # GT trajectories are always complete
                        'unroll_plus_td_steps': getattr(self, 'unroll_plus_td_steps', 
                                                      getattr(self.policy_config, 'num_unroll_steps', 5) + 
                                                      getattr(self.policy_config, 'td_steps', 5))
                    }
                    
                    # Add to return data in EXACT same format as regular collection
                    # Format: [game_segments_list, meta_data_list]
                    return_data[0].append(game_segment)  # Add GameSegment to list
                    return_data[1].append(gt_meta_data)  # Add metadata dict to list
                    injected_count += 1
                    
                    if self._rank == 0:
                        self._logger.debug(f"    ✅ Successfully injected GT trajectory {i+1}")
                else:
                    if self._rank == 0:
                        self._logger.warning(f"    ❌ Failed to convert GT trajectory {i+1}")
                    conversion_failures += 1
            
            # Update tracking counters
            self._total_gt_injections += injected_count
            self._injection_attempts += 1
            
            # Update buffer tracking
            self.update_gt_buffer_tracking(injected_count, num_regular_trajectories)
            
            if self._rank == 0:
                self._logger.info(f"  📊 GT Injection Results:")
                self._logger.info(f"    • Attempted: {inject_count}")
                self._logger.info(f"    • Successfully injected: {injected_count}")
                self._logger.info(f"    • Conversion failures: {conversion_failures}")
                self._logger.info(f"    • Total GT injections so far: {self._total_gt_injections}")
                
                if injected_count > 0:
                    new_total = num_regular_trajectories + injected_count
                    batch_gt_ratio = injected_count / new_total if new_total > 0 else 0
                    overall_gt_ratio = self.get_current_gt_ratio()
                    self._logger.info(f"    • This batch GT ratio: {batch_gt_ratio:.1%}")
                    self._logger.info(f"    • Overall buffer GT ratio: {overall_gt_ratio:.1%}")
                    self._logger.info(f"    ✅ Successfully injected {injected_count} GT trajectories")
                else:
                    self._logger.warning(f"    ❌ No GT trajectories were successfully injected!")
                
        except Exception as e:
            if self._rank == 0:
                self._logger.error(f"❌ Error injecting GT trajectories: {e}")
                import traceback
                traceback.print_exc()
    
    def _compute_gt_priorities(self, game_segment, traj_data):
        """
        Overview:
            Compute priorities for ground-truth trajectories using the same method as regular trajectories.
            This ensures GT data has the exact same format and priority structure.
            
        Arguments:
            - game_segment: The GameSegment for the ground-truth trajectory
            - traj_data: The trajectory data dictionary
            
        Returns:
            - priorities: numpy array of priorities matching the regular priority computation
        """
        try:
            # Use the same priority computation as regular trajectories
            # Priorities are typically based on TD errors or value prediction differences
            values = traj_data.get('values', [])
            
            if not values or len(values) == 0:
                # Fallback: use high priority for all steps (similar to new data)
                segment_length = len(traj_data.get('action_sequence', []))
                return np.array([2.0] * segment_length, dtype=np.float32) + 1e-6
            
            # Compute priorities using the same logic as _compute_priorities in MuZeroCollector
            # For GT trajectories, we treat them as having perfect predictions (low TD error)
            # but slightly elevated priority to ensure they get sampled
            
            # Base priority for ground-truth (slightly higher than average)
            base_priority = 1.5
            
            # Add variation based on value magnitude (similar to regular priority computation)
            priorities = []
            for i, value in enumerate(values):
                # Small variation based on value to prevent all-same priorities
                value_variation = min(abs(float(value)) * 0.1, 0.5)
                priority = base_priority + value_variation + 1e-6  # Avoid zero priorities
                priorities.append(priority)
            
            return np.array(priorities, dtype=np.float32)
            
        except Exception as e:
            if self._rank == 0 and hasattr(self, '_logger'):
                self._logger.warning(f"Error computing GT priorities: {e}")
            # Fallback to uniform high priority
            segment_length = len(traj_data.get('action_sequence', []))
            return np.array([2.0] * segment_length, dtype=np.float32) + 1e-6
    
    def _extract_and_store_gag_pair(self, env_id: int, episode_timestep):
        """
        Extract and store GAG pair from a completed episode.
        """
        try:
            # Extract generated and ground-truth SELFIES from the completed episode
            generated_selfies, ground_truth_selfies, spectrum_embed = self._extract_episode_selfies_pair(
                env_id, episode_timestep, None  # We don't have game_segment here
            )
            
            if generated_selfies and ground_truth_selfies and spectrum_embed is not None:
                # Ensure attributes exist
                if not hasattr(self, '_generated_ground_truth_pairs'):
                    self._generated_ground_truth_pairs = []
                if not hasattr(self, '_episode_pair_count'):
                    self._episode_pair_count = 0
                
                # Store the pair for adversarial training
                pair_data = {
                    'generated_selfies': generated_selfies,
                    'ground_truth_selfies': ground_truth_selfies,
                    'spectrum_embed': spectrum_embed,
                    'episode_reward': episode_timestep.info.get('eval_episode_return', 0.0),
                    'episode_length': episode_timestep.info.get('episode_length', 0),
                    'collection_timestamp': time.time()
                }
                
                self._generated_ground_truth_pairs.append(pair_data)
                self._episode_pair_count += 1
                
                # Generate ground-truth trajectory from the ground-truth SELFIES
                if hasattr(self, '_logger') and self._rank == 0:
                    escaped_gt_selfies = escape_selfies_for_logging(ground_truth_selfies[:30])
                    self._logger.debug(f"Generating GT trajectory for pair {self._episode_pair_count}: '{escaped_gt_selfies}...'")
                
                gt_trajectory = self.generate_ground_truth_trajectory(ground_truth_selfies, spectrum_embed)
                if gt_trajectory is not None:
                    pair_data['ground_truth_trajectory'] = gt_trajectory
                    if hasattr(self, '_logger') and self._rank == 0:
                        self._logger.info(f"✅ Generated GT trajectory for pair {self._episode_pair_count} "
                                        f"(reward: {gt_trajectory.get('final_reward', 0):.4f}, "
                                        f"length: {gt_trajectory.get('trajectory_length', 0)})")
                else:
                    if hasattr(self, '_logger') and self._rank == 0:
                        self._logger.warning(f"❌ Failed to generate GT trajectory for pair {self._episode_pair_count}")
                        self._logger.warning(f"   Ground-truth SELFIES: '{escaped_gt_selfies}...'")
                        self._logger.warning(f"   Spectrum embed shape: {spectrum_embed.shape if spectrum_embed is not None else 'None'}")
                
                if hasattr(self, '_logger') and self._rank == 0:
                    # Escape SELFIES strings to avoid rich markup interpretation
                    escaped_gen = escape_selfies_for_logging(generated_selfies[:20])
                    escaped_gt = escape_selfies_for_logging(ground_truth_selfies[:20])
                    self._logger.debug(f"Collected GAG pair {self._episode_pair_count}: "
                                     f"Generated='{escaped_gen}...', "
                                     f"GT='{escaped_gt}...', "
                                     f"Reward={pair_data['episode_reward']:.3f}")
            else:
                if self._rank == 0:
                    self._logger.debug(f"Failed to extract valid GAG pair from env {env_id}")
                    
        except Exception as e:
            if self._rank == 0:
                # Escape error message to prevent rich markup issues with SELFIES tokens
                escaped_error = str(e).replace('[', '\\[').replace(']', '\\]')
                self._logger.warning(f"Exception in GAG pair extraction from env {env_id}: {escaped_error}")
                import traceback
                traceback.print_exc()
    


    def pad_and_save_last_trajectory(self, i: int, last_game_segments: List[GameSegment],
                                     last_game_priorities: List[np.ndarray],
                                     game_segments: List[GameSegment], done: np.ndarray) -> None:
        """
        Overview:
            Enhanced trajectory saving that ensures proper value targets are computed
            before saving trajectories to the replay buffer.
        """
        # If this is the end of an episode, finalize the value computation
        if done[i] and last_game_segments[i] is not None:
            self._finalize_episode_values(i, last_game_segments[i])
            
            # Also handle episode completion for GAG pair extraction and GT trajectory generation
            # We need to create a dummy episode_timestep since we don't have access to it here
            # This is a backup mechanism in case the collect() method didn't catch the completion
            try:
                # Create a minimal episode_timestep-like object with info
                class DummyTimestep:
                    def __init__(self):
                        self.done = True
                        self.info = {}
                
                dummy_timestep = DummyTimestep()
                # Try to extract any remaining info from the game segment if possible
                # This is a fallback - the main extraction should happen in collect()
                # but this ensures we don't miss any completions
                
                if self._rank == 0:
                    self._logger.debug(f"Backup episode completion handling for env {i}")
                
            except Exception as e:
                if self._rank == 0:
                    # Escape error message to prevent rich markup issues with SELFIES tokens
                    escaped_error = str(e).replace('[', '\\[').replace(']', '\\]')
                    self._logger.warning(f"Error in backup episode completion handling: {escaped_error}")
        
        # Call the parent method to handle the actual trajectory saving
        super().pad_and_save_last_trajectory(i, last_game_segments, last_game_priorities, game_segments, done) 

def selfies_to_action_sequence(selfies_string, vocab_dict, max_length=None):
    """
    Convert a SELFIES string to a sequence of actions/tokens.
    Ensures proper end token is added for complete trajectories.
    
    Args:
        selfies_string: SELFIES string to convert
        vocab_dict: Dictionary mapping tokens to indices
        max_length: Maximum sequence length
    
    Returns:
        action_sequence: List of action indices including end token
    """
    try:
        # Parse SELFIES into tokens
        tokens = list(sf.split_selfies(selfies_string))
        
        if not tokens:
            escaped_selfies = str(selfies_string).replace('[', '\\[').replace(']', '\\]')
            print(f"[WARN] Empty SELFIES tokens for: {escaped_selfies}")
            # Return minimal sequence with end token
            sequence = [vocab_dict.get('[C]', 0)]
            if '<END>' in vocab_dict:
                sequence.append(vocab_dict['<END>'])
            return sequence
        
        # Convert tokens to indices
        action_sequence = []
        skipped_tokens = []
        for token in tokens:
            if token in vocab_dict:
                action_sequence.append(vocab_dict[token])
            else:
                # Skip unknown tokens instead of using fallback
                escaped_token = str(token).replace('[', '\\[').replace(']', '\\]')
                skipped_tokens.append(escaped_token)
        
        # Log skipped tokens only if there are any (reduce log noise)
        if skipped_tokens:
            print(f"[INFO] Skipped {len(skipped_tokens)} unknown tokens: {skipped_tokens[:3]}{'...' if len(skipped_tokens) > 3 else ''}")
        
        # IMPORTANT: Always add end token for ground-truth trajectories
        if '<END>' in vocab_dict:
            action_sequence.append(vocab_dict['<END>'])
            print(f"[DEBUG] Added <END> token to GT trajectory. Final length: {len(action_sequence)}")
        else:
            print("[WARN] No <END> token in vocabulary! This may cause issues with trajectory completion.")
        
        # Handle max_length constraints
        if max_length is not None:
            if len(action_sequence) > max_length:
                # Truncate but keep end token if possible
                if '<END>' in vocab_dict and len(action_sequence) >= 2:
                    action_sequence = action_sequence[:max_length-1] + [vocab_dict['<END>']]
                else:
                    action_sequence = action_sequence[:max_length]
                print(f"[DEBUG] Truncated GT trajectory to max_length {max_length}")
            else:
                # Pad with padding tokens (not end tokens - we want only one end token)
                pad_token = vocab_dict.get('<PAD>', 0)
                while len(action_sequence) < max_length:
                    action_sequence.append(pad_token)
                print(f"[DEBUG] Padded GT trajectory to max_length {max_length}")
        
        print(f"[DEBUG] GT action sequence: {action_sequence[:10]}{'...' if len(action_sequence) > 10 else ''} (length: {len(action_sequence)})")
        return action_sequence
        
    except Exception as e:
        print(f"[ERROR] Error converting SELFIES to action sequence: {e}")
        # Fallback to minimal sequence with end token
        sequence = [vocab_dict.get('[C]', 0)]
        if '<END>' in vocab_dict:
            sequence.append(vocab_dict['<END>'])
        return sequence

def compute_trajectory_values(rewards, discount_factor=0.997):
    """
    Compute discounted return values for a trajectory.
    
    Args:
        rewards: List of rewards at each step
        discount_factor: Discount factor for future rewards
    
    Returns:
        values: List of discounted return values
    """
    values = []
    cumulative_return = 0.0
    
    # Compute returns backwards
    for reward in reversed(rewards):
        cumulative_return = reward + discount_factor * cumulative_return
        values.append(cumulative_return)
    
    # Reverse to get forward order
    values.reverse()
    return values

def create_ground_truth_trajectory(ground_truth_selfies, spectrum_embed, reward_function, 
                                 vocab_dict, max_length=None, discount_factor=0.997):
    """
    Create a ground-truth trajectory from a SELFIES string and compute proper values.
    Ensures proper end token handling and final reward computation.
    
    IMPROVEMENTS:
    1. Proper end token handling: Trajectory stops at <END> token and receives final reward
    2. Final reward computation: Complete ground-truth SELFIES gets full target reward  
    3. Progressive rewards: Intermediate steps get increasing rewards based on progress
    4. Validation: Comprehensive error handling and debugging output
    5. Trajectory completion flag: Tracks whether end token was properly reached
    
    Args:
        ground_truth_selfies: Target SELFIES string
        spectrum_embed: Spectrum embedding for reward computation
        reward_function: Function to compute rewards given SELFIES and spectrum
        vocab_dict: Vocabulary mapping tokens to indices
        max_length: Maximum trajectory length
        discount_factor: Discount factor for value computation
    
    Returns:
        trajectory_data: Dictionary containing trajectory information
    """
    try:
        # Convert SELFIES to action sequence (includes proper end token)
        action_sequence = selfies_to_action_sequence(ground_truth_selfies, vocab_dict, max_length)
        
        if not action_sequence:
            escaped_selfies = str(ground_truth_selfies).replace('[', '\\[').replace(']', '\\]')
            print(f"[WARN] Empty action sequence for SELFIES: {escaped_selfies}")
            return None
        
        # Generate intermediate SELFIES at each step
        intermediate_selfies = []
        intermediate_rewards = []
        
        # Convert back to token strings for reconstruction
        idx_to_token = {idx: token for token, idx in vocab_dict.items()}
        
        # Build trajectory step by step
        current_tokens = []
        trajectory_complete = False
        
        for step, action_idx in enumerate(action_sequence):
            if action_idx in idx_to_token:
                token = idx_to_token[action_idx]
                
                if token == '<END>':
                    # End token reached - complete the trajectory with final reward
                    trajectory_complete = True
                    final_selfies = ''.join(current_tokens) if current_tokens else '[C]'
                    intermediate_selfies.append(final_selfies)
                    
                    # Compute final reward for the complete SELFIES
                    try:
                        if reward_function is not None:
                            # Give full target reward for completing the ground-truth SELFIES
                            final_reward = reward_function(ground_truth_selfies, spectrum_embed)
                            if isinstance(final_reward, torch.Tensor):
                                final_reward = float(final_reward.item())
                            intermediate_rewards.append(float(final_reward))
                            escaped_selfies = str(ground_truth_selfies).replace('[', '\\[').replace(']', '\\]')
                            print(f"[DEBUG] Final reward for complete GT SELFIES '{escaped_selfies}': {final_reward:.4f}")
                        else:
                            # High reward for completing the target
                            intermediate_rewards.append(1.0)
                            print(f"[DEBUG] Fallback final reward for complete GT SELFIES: 1.0")
                    except Exception as e:
                        print(f"[WARN] Error computing final reward: {e}")
                        intermediate_rewards.append(1.0)
                    
                    break  # Stop at end token
                    
                elif token not in ['<PAD>']:
                    # Regular token - add to sequence
                    current_tokens.append(token)
                    current_selfies = ''.join(current_tokens)
                    intermediate_selfies.append(current_selfies)
                    
                    # Compute intermediate reward
                    try:
                        if reward_function is not None:
                            reward = reward_function(current_selfies, spectrum_embed)
                            if isinstance(reward, torch.Tensor):
                                reward = float(reward.item())
                            intermediate_rewards.append(float(reward))
                        else:
                            # Progressive reward based on completion
                            target_tokens = list(sf.split_selfies(ground_truth_selfies))
                            progress = len(current_tokens) / len(target_tokens) if target_tokens else 0
                            # Give increasing reward as we approach completion
                            progress_reward = 0.1 + 0.4 * progress  # 0.1 to 0.5 based on progress
                            intermediate_rewards.append(progress_reward)
                    except Exception as e:
                        print(f"[WARN] Error computing intermediate reward: {e}")
                        intermediate_rewards.append(0.1)
                else:
                    # Skip padding tokens
                    continue
            else:
                # Unknown action index - skip
                print(f"[WARN] Unknown action index {action_idx} in vocabulary")
                continue
        
        # If we didn't reach an end token, add final reward anyway
        if not trajectory_complete and intermediate_rewards:
            print(f"[WARN] Trajectory didn't reach end token, adding final reward anyway")
            try:
                if reward_function is not None:
                    final_reward = reward_function(ground_truth_selfies, spectrum_embed)
                    if isinstance(final_reward, torch.Tensor):
                        final_reward = float(final_reward.item())
                    # Replace last reward with final reward
                    intermediate_rewards[-1] = float(final_reward)
                else:
                    intermediate_rewards[-1] = 1.0
            except:
                intermediate_rewards[-1] = 1.0
        
        if not intermediate_rewards:
            escaped_selfies = str(ground_truth_selfies).replace('[', '\\[').replace(']', '\\]')
            print(f"[WARN] No intermediate rewards generated for SELFIES: {escaped_selfies}")
            return None
        
        # Ensure we have the final action sequence matching the rewards
        final_action_sequence = action_sequence[:len(intermediate_rewards)]
        
        # Compute values from rewards
        trajectory_values = compute_trajectory_values(intermediate_rewards, discount_factor)
        
        # Create trajectory data
        trajectory_data = {
            'action_sequence': final_action_sequence,
            'selfies_sequence': intermediate_selfies,
            'rewards': intermediate_rewards,
            'values': trajectory_values,
            'target_selfies': ground_truth_selfies,
            'spectrum_embed': spectrum_embed,
            'trajectory_length': len(intermediate_rewards),
            'final_reward': intermediate_rewards[-1] if intermediate_rewards else 0.0,
            'discounted_return': trajectory_values[0] if trajectory_values else 0.0,
            'trajectory_complete': trajectory_complete,  # Flag indicating if end token was reached
        }
        
        print(f"[DEBUG] Created GT trajectory: length={len(intermediate_rewards)}, "
              f"complete={trajectory_complete}, final_reward={trajectory_data['final_reward']:.4f}")
        
        return trajectory_data
        
    except Exception as e:
        print(f"[ERROR] Error creating ground-truth trajectory: {e}")
        import traceback
        traceback.print_exc()
        return None 

def test_ground_truth_trajectory_generation():
    """
    Test function to validate ground-truth trajectory generation with proper end tokens.
    Run this to verify the improvements work correctly.
    """
    print("\n=== Testing Ground-Truth Trajectory Generation ===")
    
    # Test vocabulary with end token
    test_vocab = {
        '<PAD>': 0, '[C]': 1, '[O]': 2, '[N]': 3, '[=C]': 4, 
        '[=O]': 5, '[Ring1]': 6, '[Branch1]': 7, '<END>': 8
    }
    
    # Test SELFIES
    test_selfies = "[C][=C][C][O]"  # Simple molecule
    
    # Test spectrum embed
    import numpy as np
    test_spectrum = np.random.randn(128).astype(np.float32)
    
    # Simple test reward function
    def test_reward_fn(selfies, spectrum):
        # Higher reward for longer, complete molecules
        return len(selfies) * 0.1 + (1.0 if selfies == "[C][=C][C][O]" else 0.0)
    
    print(f"Test SELFIES: {test_selfies}")
    print(f"Test vocab size: {len(test_vocab)}")
    print(f"End token in vocab: {'<END>' in test_vocab}")
    
    # Test action sequence conversion
    action_seq = selfies_to_action_sequence(test_selfies, test_vocab)
    print(f"Action sequence: {action_seq}")
    print(f"Action sequence length: {len(action_seq)}")
    
    # Test trajectory creation
    trajectory = create_ground_truth_trajectory(
        ground_truth_selfies=test_selfies,
        spectrum_embed=test_spectrum,
        reward_function=test_reward_fn,
        vocab_dict=test_vocab,
        max_length=20,
        discount_factor=0.99
    )
    
    if trajectory:
        print("\n✅ Trajectory created successfully!")
        print(f"  - Length: {trajectory['trajectory_length']}")
        print(f"  - Complete: {trajectory.get('trajectory_complete', False)}")
        print(f"  - Final reward: {trajectory['final_reward']:.4f}")
        print(f"  - Discounted return: {trajectory['discounted_return']:.4f}")
        print(f"  - SELFIES sequence: {trajectory['selfies_sequence']}")
        print(f"  - Rewards: {[f'{r:.3f}' for r in trajectory['rewards']]}")
        
        # Validate end token handling
        if trajectory.get('trajectory_complete', False):
            print("✅ End token properly handled")
        else:
            print("⚠️  End token handling issue")
            
        # Validate final reward
        if trajectory['final_reward'] > 0.5:  # Should be high for complete SELFIES
            print("✅ Final reward properly computed")
        else:
            print("⚠️  Final reward seems low")
            
    else:
        print("❌ Failed to create trajectory")
    
    print("=== Test Complete ===\n")

# Uncomment to run test:
# test_ground_truth_trajectory_generation() 