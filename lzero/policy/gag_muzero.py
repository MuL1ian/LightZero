import copy
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.nn import L1Loss, KLDivLoss, BCEWithLogitsLoss

from lzero.mcts import GumbelMuZeroMCTSCtree as MCTSCtree
from lzero.mcts import MuZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, negative_cosine_similarity, \
    prepare_obs, configure_optimizers
from lzero.policy.gumbel_muzero import GumbelMuZeroPolicy
from lzero.model.global_reward_network import get_reward_function, get_reward_network, send_training_data_to_server

# Import MassSelfiesED model components
# try:
from lzero.model.muzero_transformer import MassSelfiesED, SelfiesTokenizer
from lzero.model.pretrain_transformer import load_pretrained_model, PretrainConfig
MASSSELFIESED_AVAILABLE = True
# except ImportError as e:
#     print(f"[WARN] MassSelfiesED not available: {e}")
#     MASSSELFIESED_AVAILABLE = False
#     MassSelfiesED = None
#     SelfiesTokenizer = None
#     load_pretrained_model = None
#     PretrainConfig = None
import torch.nn as nn
from typing import Optional
from easydict import EasyDict


@POLICY_REGISTRY.register('gag_muzero')
class GAGMuZeroPolicy(GumbelMuZeroPolicy):
    """
    Overview:
        GAG (Generated vs Ground-truth Adversarial) MuZero policy that extends GumbelMuZeroPolicy
        with adversarial training capabilities using a global reward network.
        
        This policy enables real-time training of the global reward network by sending
        training data to the reward server, which handles both inference and training.
    """
    
    config = dict(
        **GumbelMuZeroPolicy.config,  # Inherit base config
        
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
        adversarial_training_start_step=0,  # Start adversarial training immediately
        # Server-based reward network training configuration
        reward_network_learning_rate=1e-4,
        reward_network_weight_decay=1e-4,
        training_batch_size=32,
        training_timeout=10.0,
    )
    
    def __init__(
        self,
        cfg: Union[EasyDict, dict],
        **kwargs
    ):
        # Initialize base GAG MuZero components FIRST
        # Pass all parameters as-is to parent class
        super().__init__(cfg, **kwargs)
        
        # Now we can safely access self._cfg which was created by the parent class
        
        # Adversarial training configuration
        self._cfg.enable_adversarial_training = cfg.get('enable_adversarial_training', True)
        self._cfg.adversarial_loss_weight = cfg.get('adversarial_loss_weight', 0.1)
        self._cfg.preference_loss_weight = cfg.get('preference_loss_weight', 0.05)
        self._cfg.preference_temperature = cfg.get('preference_temperature', 1.0)
        self._cfg.use_global_reward_network = cfg.get('use_global_reward_network', True)
        self._cfg.normalize_rewards = cfg.get('normalize_rewards', True)
        self._cfg.reward_norm_scale = cfg.get('reward_norm_scale', 10.0)
        self._cfg.use_curriculum_learning = cfg.get('use_curriculum_learning', False)
        self._cfg.curriculum_steps = cfg.get('curriculum_steps', 5000)
        
        # Server-based training configuration
        self._cfg.reward_network_learning_rate = cfg.get('reward_network_learning_rate', 1e-4)
        self._cfg.reward_network_weight_decay = cfg.get('reward_network_weight_decay', 1e-4)
        self._cfg.training_batch_size = cfg.get('training_batch_size', 32)
        self._cfg.training_timeout = cfg.get('training_timeout', 10.0)
        
        # GAG-specific initialization
        self._gag_collector = None
        self._gag_pairs_seen = set()
        
        # Initialize global reward network access (server-based)
        if self._cfg.use_global_reward_network:
            try:
                self._reward_function = get_reward_function()
                self._reward_network = get_reward_network()
                print("[INFO] GAG MuZero: Initialized global reward network")
            except Exception as e:
                print(f"[WARN] GAG MuZero: Failed to initialize global reward network: {e}")
                self._reward_function = None
                self._reward_network = None
        else:
            self._reward_function = None
            self._reward_network = None
            print("[WARN] GAG MuZero: Global reward network disabled")

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method for GAG MuZero. Inherits from Gumbel MuZero and adds 
            adversarial training components.
        """
        # Initialize base components
        super()._init_learn()
        
        # Initialize GAG-specific components
        if self._cfg.enable_adversarial_training:
            # Initialize preference learning loss
            self._preference_loss = BCEWithLogitsLoss(reduction='none')
            
            # Initialize adversarial training state
            self._adversarial_training_step = 0
            self._current_adversarial_weight = 0.1
            
            print("[INFO] GAG MuZero: Adversarial training enabled using collected generated/ground-truth pairs")
        
        # Note: Transformer is loaded directly in MuZeroSelfiesTransformerEnhanced
        # via pretrained_transformer_path - no separate loading needed here
        print("[INFO] GAG MuZero: Using transformer from policy model")

    def _forward_learn(self, data: torch.Tensor) -> Dict[str, Union[float, int]]:
        """
        Overview:
            The forward function for learning policy in GAG MuZero. Extends Gumbel MuZero
            with adversarial reward training using explicitly labeled instances.
        """
        # Get base MuZero losses
        base_info = super()._forward_learn(data)
        
        # Add adversarial training if enabled
        # Use adversarial_training_start_step with default value of 0
        adversarial_start_step = getattr(self._cfg, 'adversarial_training_start_step', 0)
        if self._cfg.enable_adversarial_training and self._adversarial_training_step >= adversarial_start_step:
            try:
                adversarial_info = self._compute_adversarial_loss(data)
                
                # Update total loss with adversarial component
                adversarial_loss = adversarial_info['adversarial_loss']
                preference_loss = adversarial_info['preference_loss']
                
                # Apply curriculum learning if enabled
                if self._cfg.use_curriculum_learning:
                    progress = min(1.0, (self._adversarial_training_step - self._cfg.adversarial_training_start_step) / self._cfg.curriculum_steps)
                    self._current_adversarial_weight = self._cfg.adversarial_loss_weight * progress
                else:
                    self._current_adversarial_weight = self._cfg.adversarial_loss_weight
                
                # Add adversarial losses to base info
                base_info.update({
                    'adversarial_loss': adversarial_loss,
                    'preference_loss': preference_loss,
                    'adversarial_weight': self._current_adversarial_weight,
                    'reward_accuracy': adversarial_info.get('reward_accuracy', 0.0),
                    'mean_positive_reward': adversarial_info.get('mean_positive_reward', 0.0),
                    'mean_negative_reward': adversarial_info.get('mean_negative_reward', 0.0),
                })
                
                # Update the weighted total loss
                original_weighted_loss = base_info['weighted_total_loss']
                additional_loss = (
                    self._current_adversarial_weight * adversarial_loss +
                    self._cfg.preference_loss_weight * preference_loss
                )
                base_info['weighted_total_loss'] = original_weighted_loss + additional_loss
                
            except Exception as e:
                print(f"[WARN] GAG MuZero: Error in adversarial training: {e}")
                # Add default values to prevent errors
                base_info.update({
                    'adversarial_loss': 0.0,
                    'preference_loss': 0.0,
                    'adversarial_weight': 0.0,
                    'reward_accuracy': 0.0,
                    'mean_positive_reward': 0.0,
                    'mean_negative_reward': 0.0,
                })
        else:
            # Add default values when adversarial training is not active
            base_info.update({
                'adversarial_loss': 0.0,
                'preference_loss': 0.0,
                'adversarial_weight': 0.0,
                'reward_accuracy': 0.0,
                'mean_positive_reward': 0.0,
                'mean_negative_reward': 0.0,
            })
        
        self._adversarial_training_step += 1
        return base_info

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, including GAG-specific state.
        """
        base_state = super()._state_dict_learn()
        
        gag_state = {
            'adversarial_training_step': self._adversarial_training_step,
            'current_adversarial_weight': self._current_adversarial_weight,
        }
        
        base_state.update(gag_state)
        return base_state

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into GAG MuZero learn mode.
        """
        # Load base state
        super()._load_state_dict_learn(state_dict)
        
        # Load GAG-specific state
        self._adversarial_training_step = state_dict.get('adversarial_training_step', 0)
        self._current_adversarial_weight = state_dict.get('current_adversarial_weight', 0.0)

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode, including GAG-specific metrics.
        """
        base_vars = super()._monitor_vars_learn()
        
        gag_vars = [
            'adversarial_loss',
            'preference_loss',
            'adversarial_weight',
            'reward_accuracy',
            'mean_positive_reward',
            'mean_negative_reward',
        ]
        
        return base_vars + gag_vars

    def _compute_adversarial_loss(self, data: torch.Tensor) -> Dict[str, float]:
        """
        Overview:
            Compute adversarial loss using pairwise preference learning.
            Uses collected generated/ground-truth SELFIES pairs from the GAG collector.
        
        Arguments:
            - data: Training data from replay buffer including game segments with GAG pairs
            
        Returns:
            - info_dict: Dictionary containing adversarial loss components
        """
        if not self._cfg.use_global_reward_network or self._reward_function is None:
            return {
                'adversarial_loss': 0.0,
                'preference_loss': 0.0,
                'reward_accuracy': 0.0,
                'mean_positive_reward': 0.0,
                'mean_negative_reward': 0.0,
            }
        
        current_batch, target_batch = data
        obs_batch_ori, action_batch, improved_policy_batch, mask_batch, indices, weights, make_time = current_batch
        
        try:
            # Extract generated/ground-truth pairs from game segments 
            positive_rewards, negative_rewards = self._extract_gag_pairs_rewards(current_batch)
            
            # Compute pairwise preference loss
            preference_loss = self._compute_preference_loss(positive_rewards, negative_rewards)
            
            # Compute adversarial reward loss (encourages model to distinguish between positive and negative)
            adversarial_loss = self._compute_reward_discrimination_loss(positive_rewards, negative_rewards)
            
            # Compute accuracy metrics
            reward_accuracy = self._compute_reward_accuracy(positive_rewards, negative_rewards)
            
            return {
                'adversarial_loss': adversarial_loss,
                'preference_loss': preference_loss,
                'reward_accuracy': reward_accuracy,
                'mean_positive_reward': float(torch.mean(positive_rewards)) if len(positive_rewards) > 0 else 0.0,
                'mean_negative_reward': float(torch.mean(negative_rewards)) if len(negative_rewards) > 0 else 0.0,
            }
            
        except Exception as e:
            print(f"[WARN] GAG MuZero: Error computing adversarial loss: {e}")
            return {
                'adversarial_loss': 0.0,
                'preference_loss': 0.0,
                'reward_accuracy': 0.0,
                'mean_positive_reward': 0.0,
                'mean_negative_reward': 0.0,
            }

    def _extract_gag_pairs_rewards(self, current_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Extract positive and negative rewards from collected GAG pairs in game segments.
            
        Arguments:
            - current_batch: Current training batch data
            
        Returns:
            - positive_rewards: Rewards for ground-truth SELFIES (positive instances)
            - negative_rewards: Rewards for generated SELFIES (negative instances)
        """
        positive_rewards = []
        negative_rewards = []
        
        # Extract game segments if available (depends on data structure)
        game_segments = None
        if hasattr(current_batch, 'game_segments'):
            game_segments = current_batch.game_segments
        elif len(current_batch) > 7:  # Additional elements beyond standard batch
            game_segments = current_batch[7] if isinstance(current_batch[7], list) else None
        
        # Gracefully fallback to collector-level pairs if no game segments are attached to the current batch.
        if game_segments is None:
            # This is normal when the replay buffer does not store game segments; the dedicated
            # GAG collector already keeps a global list of generated/ground-truth pairs.
            # Use them directly instead of raising a warning and returning empty tensors.
            return self._get_collector_pairs_rewards()
        
        # Process each game segment
        for game_segment in game_segments:
            if hasattr(game_segment, 'gag_pairs') and game_segment.gag_pairs:
                for pair_data in game_segment.gag_pairs:
                    try:
                        generated_selfies = pair_data['generated_selfies']
                        ground_truth_selfies = pair_data['ground_truth_selfies']
                        spectrum_embed = pair_data['spectrum_embed']
                        
                        # Convert spectrum_embed to tensor if needed
                        if not isinstance(spectrum_embed, torch.Tensor):
                            spectrum_embed = torch.tensor(spectrum_embed, device=self._cfg.device, dtype=torch.float32)
                        else:
                            spectrum_embed = spectrum_embed.to(self._cfg.device)
                        
                        # Compute rewards using the global reward network
                        generated_reward = self._reward_function(generated_selfies, spectrum_embed)
                        ground_truth_reward = self._reward_function(ground_truth_selfies, spectrum_embed)
                        
                        # Ground-truth SELFIES are positive instances
                        positive_rewards.append(ground_truth_reward)
                        
                        # Generated SELFIES are negative instances
                        negative_rewards.append(generated_reward)
                        
                    except Exception as e:
                        print(f"[WARN] GAG MuZero: Error processing GAG pair: {e}")
                        continue
        
        # Convert to tensors
        positive_rewards = torch.tensor(positive_rewards, device=self._cfg.device, dtype=torch.float32) if positive_rewards else torch.tensor([], device=self._cfg.device, dtype=torch.float32)
        negative_rewards = torch.tensor(negative_rewards, device=self._cfg.device, dtype=torch.float32) if negative_rewards else torch.tensor([], device=self._cfg.device, dtype=torch.float32)
        
        # If we don't have pairs, try accessing collector pairs as fallback
        if len(positive_rewards) == 0 and len(negative_rewards) == 0:
            positive_rewards, negative_rewards = self._get_collector_pairs_rewards()
        
        return positive_rewards, negative_rewards

    def _get_collector_pairs_rewards(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Get collected pairs from the GAG collector as fallback.
            
        Returns:
            - positive_rewards: Rewards for ground-truth SELFIES
            - negative_rewards: Rewards for generated SELFIES
        """
        positive_rewards = []
        negative_rewards = []
        
        try:
            # Try to access the collector's collected pairs
            if hasattr(self, '_collector') and hasattr(self._collector, 'get_collected_pairs'):
                pairs = self._collector.get_collected_pairs()
                
                for pair_data in pairs:
                    try:
                        generated_selfies = pair_data['generated_selfies']
                        ground_truth_selfies = pair_data['ground_truth_selfies']
                        spectrum_embed = pair_data['spectrum_embed']
                        
                        # Convert spectrum_embed to tensor if needed
                        if not isinstance(spectrum_embed, torch.Tensor):
                            spectrum_embed = torch.tensor(spectrum_embed, device=self._cfg.device, dtype=torch.float32)
                        else:
                            spectrum_embed = spectrum_embed.to(self._cfg.device)
                        
                        # Compute rewards
                        generated_reward = self._reward_function(generated_selfies, spectrum_embed)
                        ground_truth_reward = self._reward_function(ground_truth_selfies, spectrum_embed)
                        
                        positive_rewards.append(ground_truth_reward)
                        negative_rewards.append(generated_reward)
                        
                    except Exception as e:
                        print(f"[WARN] GAG MuZero: Error processing collector pair: {e}")
                        continue
        
        except Exception as e:
            print(f"[WARN] GAG MuZero: Error accessing collector pairs: {e}")
        
        # Convert to tensors
        positive_rewards = torch.tensor(positive_rewards, device=self._cfg.device, dtype=torch.float32) if positive_rewards else torch.tensor([], device=self._cfg.device, dtype=torch.float32)
        negative_rewards = torch.tensor(negative_rewards, device=self._cfg.device, dtype=torch.float32) if negative_rewards else torch.tensor([], device=self._cfg.device, dtype=torch.float32)
        
        return positive_rewards, negative_rewards

    def set_collector(self, collector):
        """
        Overview:
            Set reference to the GAG collector for accessing collected pairs.
            
        Arguments:
            - collector: GAG MuZero collector instance
        """
        self._collector = collector
        
        # Verify that this is indeed a GAG collector
        if hasattr(collector, 'get_collected_pairs'):
            print("[INFO] GAG MuZero: Successfully connected to GAG collector")
        else:
            print("[WARN] GAG MuZero: Collector may not be a GAG collector - missing get_collected_pairs method")

    def get_gag_statistics(self) -> Dict[str, Any]:
        """
        Overview:
            Get GAG-specific training statistics.
            
        Returns:
            - stats: Dictionary containing GAG training statistics
        """
        stats = {
            'adversarial_training_enabled': self._cfg.enable_adversarial_training,
            'adversarial_training_step': getattr(self, '_adversarial_training_step', 0),
            'current_adversarial_weight': getattr(self, '_current_adversarial_weight', 0.0),
            'has_collector': hasattr(self, '_collector'),
            'has_reward_function': hasattr(self, '_reward_function') and self._reward_function is not None,
        }
        
        # Add collector statistics if available
        if hasattr(self, '_collector') and hasattr(self._collector, 'get_collected_pairs'):
            pairs = self._collector.get_collected_pairs()
            stats['collected_pairs_count'] = len(pairs)
            if pairs:
                stats['avg_episode_reward'] = np.mean([p.get('episode_reward', 0.0) for p in pairs])
                stats['avg_episode_length'] = np.mean([p.get('episode_length', 0) for p in pairs])
        
        return stats

    def _compute_preference_loss(self, positive_rewards: torch.Tensor, negative_rewards: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute pairwise preference loss (similar to RLHF).
            Encourages positive instances to have higher rewards than negative instances.
            Uses element-wise comparison between corresponding positive and negative pairs.
            
        Arguments:
            - positive_rewards: Rewards for positive (ground-truth) instances
            - negative_rewards: Rewards for negative (generated) instances
            
        Returns:
            - loss: Preference learning loss
        """
        if len(positive_rewards) == 0 or len(negative_rewards) == 0:
            return torch.tensor(0.0, device=self._cfg.device)
        
        try:
            # Ensure we have the same number of positive and negative rewards for pairwise comparison
            min_length = min(len(positive_rewards), len(negative_rewards))
            if min_length == 0:
                return torch.tensor(0.0, device=self._cfg.device)
            
            # Truncate to same length for element-wise comparison
            pos_rewards = positive_rewards[:min_length]
            neg_rewards = negative_rewards[:min_length]
            
            # Normalize rewards if configured
            if self._cfg.normalize_rewards:
                all_rewards = torch.cat([pos_rewards, neg_rewards])
                reward_mean = torch.mean(all_rewards)
                reward_std = torch.std(all_rewards) + 1e-8
                
                pos_rewards = (pos_rewards - reward_mean) / reward_std * self._cfg.reward_norm_scale
                neg_rewards = (neg_rewards - reward_mean) / reward_std * self._cfg.reward_norm_scale
            
            # Element-wise preference comparison (no cross-comparison)
            # Compute preference logits (positive should be preferred over negative)
            preference_logits = (pos_rewards - neg_rewards) / self._cfg.preference_temperature
            
            # Target: positive instances should be preferred (label = 1)
            targets = torch.ones_like(preference_logits)
            
            # Compute binary cross-entropy loss
            loss = self._preference_loss(preference_logits, targets)
            
            return torch.mean(loss)
            
        except Exception as e:
            print(f"[WARN] GAG MuZero: Error computing preference loss: {e}")
            return torch.tensor(0.0, device=self._cfg.device)

    def _compute_reward_discrimination_loss(self, positive_rewards: torch.Tensor, negative_rewards: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Compute adversarial loss that encourages the model to generate instances 
            that can be discriminated by the reward network.
            
        Arguments:
            - positive_rewards: Rewards for positive instances
            - negative_rewards: Rewards for negative instances
            
        Returns:
            - loss: Adversarial discrimination loss
        """
        if len(positive_rewards) == 0 or len(negative_rewards) == 0:
            return torch.tensor(0.0, device=self._cfg.device)
        
        try:
            # Encourage positive rewards to be high and negative rewards to be low
            positive_loss = -torch.mean(positive_rewards)  # Maximize positive rewards
            negative_loss = torch.mean(negative_rewards)   # Minimize negative rewards
            
            return positive_loss + negative_loss
            
        except Exception as e:
            print(f"[WARN] GAG MuZero: Error computing discrimination loss: {e}")
            return torch.tensor(0.0, device=self._cfg.device)

    def _compute_reward_accuracy(self, positive_rewards: torch.Tensor, negative_rewards: torch.Tensor) -> float:
        """
        Overview:
            Compute accuracy metric: fraction of cases where positive reward > negative reward.
            Uses element-wise comparison between corresponding positive and negative pairs.
            
        Arguments:
            - positive_rewards: Rewards for positive instances
            - negative_rewards: Rewards for negative instances
            
        Returns:
            - accuracy: Accuracy as a float between 0 and 1
        """
        if len(positive_rewards) == 0 or len(negative_rewards) == 0:
            return 0.0
        
        try:
            # Ensure we have the same number of rewards for element-wise comparison
            min_length = min(len(positive_rewards), len(negative_rewards))
            if min_length == 0:
                return 0.0
            
            # Truncate to same length for element-wise comparison
            pos_rewards = positive_rewards[:min_length]
            neg_rewards = negative_rewards[:min_length]
            
            # Element-wise comparison: count cases where positive > negative
            correct_preferences = (pos_rewards > neg_rewards).float()
            accuracy = torch.mean(correct_preferences)
            
            return float(accuracy.item())
            
        except Exception as e:
            print(f"[WARN] GAG MuZero: Error computing reward accuracy: {e}")
            return 0.0

    def learn(
        self,
        data: List[Dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        target_update_period: int,
        target_update_period_ratio: float,
        collector_env_num: int,
        collected_data_pairs: Optional[List] = None,
        ground_truth_trajectories: Optional[List] = None
    ) -> Dict[str, Union[float, int]]:
        """
        Overview:
            Learn function that includes GAG adversarial training using server-based reward network training
            and ground-truth trajectory integration.
        """
        # Call parent learn method to get base training results
        base_info = super().learn(data, optimizer, lr_scheduler, target_update_period, target_update_period_ratio, collector_env_num)
        
        # GAG adversarial training with server-based approach
        if (self._cfg.enable_adversarial_training and 
            collected_data_pairs is not None and 
            len(collected_data_pairs) > 0):
            
            try:
                # ... existing GAG pair processing code ...
                
                # Extract unique GAG pairs
                unique_gag_pairs = self._extract_unique_gag_pairs(collected_data_pairs)
                
                if len(unique_gag_pairs) > 0:
                    print(f"[INFO] GAG MuZero: Processing {len(unique_gag_pairs)} unique GAG pairs for server training")
                    
                    # Prepare training data for server
                    generated_selfies = []
                    ground_truth_selfies = []
                    spectrum_embeds = []
                    
                    for generated, ground_truth, spectrum_embed in unique_gag_pairs:
                        generated_selfies.append(generated)
                        ground_truth_selfies.append(ground_truth)
                        spectrum_embeds.append(spectrum_embed)
                    
                    # Send training data to reward server
                    training_result = send_training_data_to_server(
                        generated_selfies=generated_selfies,
                        ground_truth_selfies=ground_truth_selfies,
                        spectrum_embeds=spectrum_embeds,
                        learning_rate=self._cfg.reward_network_learning_rate,
                        weight_decay=self._cfg.reward_network_weight_decay,
                        timeout=self._cfg.training_timeout
                    )
                    
                    if training_result.get('status') == 'success':
                        # Server training was successful
                        adversarial_loss = training_result.get('adversarial_loss', 0.0)
                        preference_loss = training_result.get('preference_loss', 0.0)
                        total_loss = training_result.get('total_loss', 0.0)
                        mean_generated_reward = training_result.get('mean_generated_reward', 0.0)
                        mean_ground_truth_reward = training_result.get('mean_ground_truth_reward', 0.0)
                        
                        # Update adversarial weight with curriculum learning
                        if self._cfg.use_curriculum_learning:
                            progress = min(1.0, self._adversarial_training_step / self._cfg.curriculum_steps)
                            self._current_adversarial_weight = self._cfg.adversarial_loss_weight * progress
                        else:
                            self._current_adversarial_weight = self._cfg.adversarial_loss_weight
                        
                        # Update step counter
                        self._adversarial_training_step += 1
                        
                        # Add training info to base_info
                        base_info.update({
                            'adversarial_loss': adversarial_loss,
                            'preference_loss': preference_loss,
                            'reward_network_loss': total_loss,
                            'adversarial_weight': self._current_adversarial_weight,
                            'adversarial_training_step': self._adversarial_training_step,
                            'gag_pairs_collected': len(unique_gag_pairs),
                            'reward_accuracy': self._compute_gag_reward_accuracy(unique_gag_pairs),
                            'mean_generated_reward': mean_generated_reward,
                            'mean_ground_truth_reward': mean_ground_truth_reward,
                            'positive_pairs': sum(1 for _, _, _ in unique_gag_pairs),
                            'negative_pairs': 0,  # All pairs are positive in current implementation
                            'mean_positive_reward': mean_ground_truth_reward,
                            'mean_negative_reward': 0.0,
                            'server_training_status': 'success',
                            'server_training_timeout': self._cfg.training_timeout,
                            'reward_network_lr': training_result.get('learning_rate', self._cfg.reward_network_learning_rate),
                            'reward_network_wd': training_result.get('weight_decay', self._cfg.reward_network_weight_decay),
                        })
                        
                        # Add ground-truth trajectory metrics if available
                        if ground_truth_trajectories and len(ground_truth_trajectories) > 0:
                            gt_stats = self._compute_ground_truth_trajectory_stats(ground_truth_trajectories)
                            base_info.update(gt_stats)
                        
                        print(f"[INFO] GAG MuZero server training successful: "
                              f"Adversarial Loss: {adversarial_loss:.4f}, "
                              f"Preference Loss: {preference_loss:.4f}, "
                              f"Pairs: {len(unique_gag_pairs)}")
                    else:
                        # Server training failed
                        error_message = training_result.get('message', 'Unknown error')
                        print(f"[ERROR] GAG MuZero server training failed: {error_message}")
                        
                        base_info.update({
                            'adversarial_loss': 0.0,
                            'preference_loss': 0.0,
                            'reward_network_loss': 0.0,
                            'adversarial_weight': self._current_adversarial_weight,
                            'adversarial_training_step': self._adversarial_training_step,
                            'gag_pairs_collected': len(unique_gag_pairs),
                            'reward_accuracy': 0.0,
                            'mean_generated_reward': 0.0,
                            'mean_ground_truth_reward': 0.0,
                            'positive_pairs': 0,
                            'negative_pairs': 0,
                            'mean_positive_reward': 0.0,
                            'mean_negative_reward': 0.0,
                            'server_training_status': 'failed',
                            'server_training_error': error_message,
                        })
                else:
                    # No GAG pairs available
                    base_info.update({
                        'adversarial_loss': 0.0,
                        'preference_loss': 0.0,
                        'reward_network_loss': 0.0,
                        'adversarial_weight': self._current_adversarial_weight,
                        'adversarial_training_step': self._adversarial_training_step,
                        'gag_pairs_collected': 0,
                        'reward_accuracy': 0.0,
                        'mean_generated_reward': 0.0,
                        'mean_ground_truth_reward': 0.0,
                        'positive_pairs': 0,
                        'negative_pairs': 0,
                        'mean_positive_reward': 0.0,
                        'mean_negative_reward': 0.0,
                        'server_training_status': 'no_data',
                    })
                    
            except Exception as e:
                print(f"[ERROR] GAG MuZero: Error in server-based adversarial training: {e}")
                base_info.update({
                    'adversarial_loss': 0.0,
                    'preference_loss': 0.0, 
                    'reward_network_loss': 0.0,
                    'adversarial_weight': self._current_adversarial_weight,
                    'adversarial_training_step': self._adversarial_training_step,
                    'gag_pairs_collected': 0,
                    'reward_accuracy': 0.0,
                    'mean_generated_reward': 0.0,
                    'mean_ground_truth_reward': 0.0,
                    'positive_pairs': 0,
                    'negative_pairs': 0,
                    'mean_positive_reward': 0.0,
                    'mean_negative_reward': 0.0,
                    'server_training_status': 'error',
                    'server_training_error': str(e),
                })
        else:
            # GAG adversarial training disabled or no data
            base_info.update({
                'adversarial_loss': 0.0,
                'preference_loss': 0.0,
                'reward_network_loss': 0.0,
                'adversarial_weight': 0.0,
                'adversarial_training_step': self._adversarial_training_step,
                'gag_pairs_collected': 0,
                'reward_accuracy': 0.0,
                'mean_generated_reward': 0.0,
                'mean_ground_truth_reward': 0.0,
                'positive_pairs': 0,
                'negative_pairs': 0,
                'mean_positive_reward': 0.0,
                'mean_negative_reward': 0.0,
                'server_training_status': 'disabled',
            })
        
        return base_info

    def state_dict(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of the policy, including GAG-specific state but not server state.
            The reward network state is managed by the server process.
        """
        base_state = super().state_dict()
        
        # GAG-specific state (training progress tracking)
        gag_state = {
            'adversarial_training_step': self._adversarial_training_step,
            'current_adversarial_weight': self._current_adversarial_weight,
            'gag_pairs_seen': len(self._gag_pairs_seen),
            'server_training_config': {
                'learning_rate': self._cfg.reward_network_learning_rate,
                'weight_decay': self._cfg.reward_network_weight_decay,
                'training_timeout': self._cfg.training_timeout,
            }
        }
        
        base_state.update(gag_state)
        return base_state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict of the policy, including GAG-specific state.
            Server-managed reward network state is handled by the server process.
        """
        # Load base policy state
        super().load_state_dict(state_dict)
        
        # Load GAG-specific state
        self._adversarial_training_step = state_dict.get('adversarial_training_step', 0)
        self._current_adversarial_weight = state_dict.get('current_adversarial_weight', 0.0)
        
        # Load server training config if available
        server_config = state_dict.get('server_training_config', {})
        if server_config:
            self._cfg.reward_network_learning_rate = server_config.get('learning_rate', self._cfg.reward_network_learning_rate)
            self._cfg.reward_network_weight_decay = server_config.get('weight_decay', self._cfg.reward_network_weight_decay)
            self._cfg.training_timeout = server_config.get('training_timeout', self._cfg.training_timeout)

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the variables to be monitored during learning, including server training metrics.
        """
        base_vars = super()._monitor_vars_learn()
        
        # Add GAG and server training monitoring variables
        gag_vars = [
            'adversarial_loss',
            'preference_loss', 
            'reward_network_loss',
            'adversarial_weight',
            'adversarial_training_step',
            'gag_pairs_collected',
            'reward_accuracy',
            'mean_generated_reward',
            'mean_ground_truth_reward',
            'positive_pairs',
            'negative_pairs',
            'mean_positive_reward',
            'mean_negative_reward',
            'server_training_status',
            'gt_trajectory_count',
            'gt_avg_final_reward',
            'gt_avg_discounted_return',
            'gt_avg_trajectory_length',
            'gt_min_final_reward',
            'gt_max_final_reward',
            'gt_trajectories_with_positive_reward',
        ]
        
        return base_vars + gag_vars
    
    def _extract_unique_gag_pairs(self, collected_data_pairs: List) -> List:
        """Extract unique GAG pairs from collected data"""
        unique_pairs = []
        for pair_data in collected_data_pairs:
            if len(pair_data) >= 3:
                generated, ground_truth, spectrum_embed = pair_data[:3]
                # Create a simple hash for deduplication
                pair_hash = hash((generated, ground_truth, tuple(spectrum_embed.flatten().tolist()) if torch.is_tensor(spectrum_embed) else str(spectrum_embed)))
                if pair_hash not in self._gag_pairs_seen:
                    self._gag_pairs_seen.add(pair_hash)
                    unique_pairs.append((generated, ground_truth, spectrum_embed))
        return unique_pairs
    
    def _compute_gag_reward_accuracy(self, gag_pairs) -> float:
        """Compute reward accuracy for GAG pairs"""
        if not gag_pairs or not self._reward_function:
            return 0.0
        
        try:
            total_accuracy = 0.0
            valid_pairs = 0
            
            for generated, ground_truth, spectrum_embed in gag_pairs:
                generated_reward = self._reward_function(generated, spectrum_embed)
                ground_truth_reward = self._reward_function(ground_truth, spectrum_embed)
                
                # Accuracy: ground truth should have higher reward
                if ground_truth_reward > generated_reward:
                    total_accuracy += 1.0
                valid_pairs += 1
            
            return total_accuracy / valid_pairs if valid_pairs > 0 else 0.0
            
        except Exception as e:
            print(f"[WARN] GAG MuZero: Error computing GAG reward accuracy: {e}")
            return 0.0
    
    def _compute_ground_truth_trajectory_stats(self, ground_truth_trajectories: List) -> Dict[str, float]:
        """
        Compute statistics for ground-truth trajectories.
        
        Args:
            ground_truth_trajectories: List of ground-truth trajectory data
            
        Returns:
            stats: Dictionary of trajectory statistics
        """
        try:
            if not ground_truth_trajectories:
                return {
                    'gt_trajectory_count': 0,
                    'gt_avg_final_reward': 0.0,
                    'gt_avg_discounted_return': 0.0,
                    'gt_avg_trajectory_length': 0.0,
                    'gt_min_final_reward': 0.0,
                    'gt_max_final_reward': 0.0,
                    'gt_trajectories_with_positive_reward': 0,
                }
            
            final_rewards = [t.get('final_reward', 0.0) for t in ground_truth_trajectories]
            discounted_returns = [t.get('discounted_return', 0.0) for t in ground_truth_trajectories]
            trajectory_lengths = [t.get('trajectory_length', 0) for t in ground_truth_trajectories]
            
            stats = {
                'gt_trajectory_count': len(ground_truth_trajectories),
                'gt_avg_final_reward': sum(final_rewards) / len(final_rewards) if final_rewards else 0.0,
                'gt_avg_discounted_return': sum(discounted_returns) / len(discounted_returns) if discounted_returns else 0.0,
                'gt_avg_trajectory_length': sum(trajectory_lengths) / len(trajectory_lengths) if trajectory_lengths else 0.0,
                'gt_min_final_reward': min(final_rewards) if final_rewards else 0.0,
                'gt_max_final_reward': max(final_rewards) if final_rewards else 0.0,
                'gt_trajectories_with_positive_reward': sum(1 for r in final_rewards if r > 0),
            }
            
            return stats
            
        except Exception as e:
            print(f"[WARN] GAG MuZero: Error computing ground-truth trajectory stats: {e}")
            return {
                'gt_trajectory_count': 0,
                'gt_avg_final_reward': 0.0,
                'gt_avg_discounted_return': 0.0,
                'gt_avg_trajectory_length': 0.0,
                'gt_min_final_reward': 0.0,
                'gt_max_final_reward': 0.0,
                'gt_trajectories_with_positive_reward': 0,
            }
