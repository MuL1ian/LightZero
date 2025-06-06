import time
from collections import deque, namedtuple
from typing import Optional, Any, List

import numpy as np
import torch
from ding.envs import BaseEnvManager
from ding.torch_utils import to_ndarray
from ding.utils import SERIAL_COLLECTOR_REGISTRY

from lzero.mcts.buffer.game_segment import GameSegment
from lzero.worker.muzero_collector import MuZeroCollector


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
                
                # Ensure spectrum_embed has consistent shape and type
                if spectrum_embed is not None:
                    # Convert to numpy array and ensure proper shape
                    if hasattr(spectrum_embed, 'detach'):
                        # PyTorch tensor
                        spectrum_embed = spectrum_embed.detach().cpu().numpy()
                    elif hasattr(spectrum_embed, 'numpy'):
                        # Some other tensor type
                        spectrum_embed = spectrum_embed.numpy()
                    else:
                        # Already numpy array
                        spectrum_embed = np.asarray(spectrum_embed)
                    
                    # Ensure it's a 1D array with the expected shape
                    if spectrum_embed.ndim > 1:
                        spectrum_embed = spectrum_embed.squeeze()
                    
                    # Ensure it's float32 for consistency
                    spectrum_embed = spectrum_embed.astype(np.float32)
                
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

    def _log_gag_statistics(self, train_iter: int):
        """
        Overview:
            Enhanced GAG statistics logging including value computation metrics.
        """
        if self._rank == 0:
            self._logger.info(f"GAG Statistics - Collected {self._episode_pair_count} generated/ground-truth pairs")
            self._logger.info(f"Value targets computed from actual environment rewards (discount={self._discount_factor})")
            
            if self._tb_logger and self._episode_pair_count > 0:
                self._tb_logger.add_scalar('gag_collector/total_pairs', self._episode_pair_count, train_iter)
                self._tb_logger.add_scalar('gag_collector/pairs_this_collection', len(self._generated_ground_truth_pairs), train_iter)
                self._tb_logger.add_scalar('gag_collector/discount_factor', self._discount_factor, train_iter)
                
                # Log average episode rewards
                if self._generated_ground_truth_pairs:
                    avg_reward = np.mean([p['episode_reward'] for p in self._generated_ground_truth_pairs])
                    self._tb_logger.add_scalar('gag_collector/avg_episode_reward', avg_reward, train_iter)

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
                        should_collect = (
                            episode_timestep.done or 
                            (has_generated and has_target and has_spectrum and 
                             self._step_counter[env_id] % self._collection_step_interval == 0)
                        )
                        
                        if should_collect and has_generated and has_target and has_spectrum:
                            # Create a hash to check for duplicates
                            generated_selfies = (info.get('generated_selfies') or 
                                               info.get('current_selfies') or 
                                               info.get('final_selfies', ''))
                            ground_truth_selfies = (info.get('target_selfies') or 
                                                  info.get('ground_truth_selfies') or 
                                                  info.get('gt_selfies', ''))
                            
                            pair_hash = hash((generated_selfies, ground_truth_selfies, env_id))
                            
                            if pair_hash not in self._collected_pair_hashes:
                                self._extract_and_store_gag_pair(env_id, episode_timestep)
                                self._collected_pair_hashes.add(pair_hash)
                        
                        elif episode_timestep.done:
                            # Try to extract anyway in case data is available
                            self._extract_and_store_gag_pair(env_id, episode_timestep)
                            
                except Exception as e:
                    if self._rank == 0:
                        self._logger.warning(f"Error processing timestep for env {env_id}: {e}")
                
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
        finally:
            # Restore the original step method
            self._env.step = original_env_step
        
        # Log GAG statistics after collection
        self._log_gag_statistics(train_iter)
        
        return return_data
    
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
                
                if hasattr(self, '_logger') and self._rank == 0:
                    self._logger.debug(f"Collected GAG pair {self._episode_pair_count}: "
                                     f"Generated='{generated_selfies[:20]}...', "
                                     f"GT='{ground_truth_selfies[:20]}...', "
                                     f"Reward={pair_data['episode_reward']:.3f}")
            else:
                if self._rank == 0:
                    self._logger.debug(f"Failed to extract valid GAG pair from env {env_id}")
                    
        except Exception as e:
            if self._rank == 0:
                self._logger.warning(f"Exception in GAG pair extraction from env {env_id}: {e}")
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
        
        # Call the parent method to handle the actual trajectory saving
        super().pad_and_save_last_trajectory(i, last_game_segments, last_game_priorities, game_segments, done) 