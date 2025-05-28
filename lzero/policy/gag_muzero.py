import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ding.model import model_wrap
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.nn import L1Loss, KLDivLoss

from lzero.mcts import GumbelMuZeroMCTSCtree as MCTSCtree
from lzero.mcts import MuZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, negative_cosine_similarity, \
    prepare_obs, \
    configure_optimizers
from lzero.policy.muzero import MuZeroPolicy 

# Import global reward network
try:
    from lzero.model import global_reward_network
    GLOBAL_REWARD_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Global reward network not available in GAG MuZero: {e}")
    GLOBAL_REWARD_AVAILABLE = False
    global_reward_network = None

@POLICY_REGISTRY.register('gag_muzero')
class GAGMuZeroPolicy(MuZeroPolicy):
    # The default_config for Gumbel MuZero policy.
    config = dict(
        model=dict(
            # (str) The model type. For 1-dimensional vector obs, we use mlp model. For the image obs, we use conv model.
            model_type='conv',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) The stacked obs shape.
            # observation_shape=(1, 96, 96),  # if frame_stack_num=1
            observation_shape=(4, 96, 96),  # if frame_stack_num=4
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=False,
            # (bool) Whether to use discrete support to represent categorical distribution for value/reward/value_prefix.
            categorical_distribution=True,
            # (int) The image channel in image observation.
            image_channel=1,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The number of res blocks in MuZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in MuZero model.
            num_channels=64,
            # (int) The scale of supports used in categorical distribution.
            # This variable is only effective when ``categorical_distribution=True``.
            support_scale=300,
            # (bool) whether to learn bias in the last linear layer in value and policy head.
            bias=True,
            # (str) The type of action encoding. Options are ['one_hot', 'not_one_hot']. Default to 'one_hot'.
            discrete_action_encoding_type='one_hot',
            # (bool) whether to use res connection in dynamics.
            res_connection_in_dynamics=True,
            # (str) The type of normalization in MuZero model. Options are ['BN', 'LN']. Default to 'LN'.
            norm_type='BN',
        ),
        # ****** common ******
        # (bool) Whether to use multi-gpu training.
        multi_gpu=False,
        # (bool) Whether to enable the sampled-based algorithm (e.g. Sampled EfficientZero)
        # this variable is used in ``collector``.
        sampled_algo=False,
        # (bool) Whether to enable the gumbel-based algorithm (e.g. Gumbel Muzero).
        gumbel_algo=True,
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=True,
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (str) The type of environment. Options is ['not_board_games', 'board_games'].
        env_type='not_board_games',
        # (str) The type of action space. Options are ['fixed_action_space', 'varied_action_space'].
        action_type='fixed_action_space',
        # (str) The type of battle mode. Options is ['play_with_bot_mode', 'self_play_mode'].
        battle_mode='play_with_bot_mode',
        # (bool) Whether to monitor extra statistics in tensorboard.
        monitor_extra_statistics=True,
        # (int) The transition number of one ``GameSegment``.
        game_segment_length=200,
        # (bool): Indicates whether to perform an offline evaluation of the checkpoint (ckpt).
        # If set to True, the checkpoint will be evaluated after the training process is complete.
        # IMPORTANT: Setting eval_offline to True requires configuring the saving of checkpoints to align with the evaluation frequency.
        # This is done by setting the parameter learn.learner.hook.save_ckpt_after_iter to the same value as eval_freq in the train_muzero.py automatically.
        eval_offline=False,

        # ****** observation ******
        # (bool) Whether to transform image to string to save memory.
        transform2string=False,
        # (bool) Whether to use gray scale image.
        gray_scale=False,
        # (bool) Whether to use data augmentation.
        use_augmentation=False,
        # (list) The style of augmentation.
        augmentation=['shift', 'intensity'],

        # ******* learn ******
        # (bool) Whether to ignore the done flag in the training data. Typically, this value is set to False.
        # However, for some environments with a fixed episode length, to ensure the accuracy of Q-value calculations,
        # we should set it to True to avoid the influence of the done flag.
        ignore_done=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.replay_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        replay_ratio=0.25,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network. ['SGD' or 'Adam']
        optim_type='SGD',
        # (float) Learning rate for training policy network. Ininitial lr for manually decay schedule.
        learning_rate=0.2,
        # (int) Frequency of target network update.
        target_update_freq=100,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=10,
        # (int) The number of episode in each collecting stage.
        n_episode=8,
        # (int) the number of simulations in MCTS.
        num_simulations=50,
        # (int) the max considred number in Gumbel MuZero MCTS simulation.
        max_num_considered_actions=4,
        # (float) Discount factor (gamma) for returns.
        discount_factor=0.997,
        # (int) The number of step for calculating target q_value.
        td_steps=5,
        # (int) The number of unroll steps in dynamics network.
        num_unroll_steps=5,
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=0,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        piecewise_decay_lr_scheduler=True,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e4),
        # (bool) Whether to use manually decayed temperature.
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        # (bool) Whether to add noise to roots during reanalyze process.
        reanalyze_noise=True,

        # ****** Priority ******
        # (bool) Whether to use priority when sampling training data from the buffer.
        use_priority=False,
        # (float) The degree of prioritization to use. A value of 0 means no prioritization,
        # while a value of 1 means full prioritization.
        priority_prob_alpha=0.6,
        # (float) The degree of correction to use. A value of 0 means no correction,
        # while a value of 1 means full correction.
        priority_prob_beta=0.4,

        # ****** UCB ******
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,

        # ****** Explore by random collect ******
        # (int) The number of episodes to collect data randomly before training.
        random_collect_episode_num=0,

        # ****** Explore by eps greedy ******
        eps=dict(
            # (bool) Whether to use eps greedy exploration in collecting data.
            eps_greedy_exploration_in_collect=False,
            # (str) The type of decaying epsilon. Options are 'linear', 'exp'.
            type='linear',
            # (float) The start value of eps.
            start=1.,
            # (float) The end value of eps.
            end=0.05,
            # (int) The decay steps from start to end eps.
            decay=int(1e5),
        ),
        
        # ****** Global Reward Network ******
        # (int) Vocabulary size for SELFIES tokenizer in reward network.
        reward_vocab_size=1000,
        # (int) Maximum SELFIES sequence length for reward network.
        max_selfies_len=100,
        # (str) Path to pretrained reward network checkpoint.
        reward_network_checkpoint=None,
        # (bool) Whether to use global reward network for reward computation.
        use_global_reward_network=True,
        
        # ****** Adversarial Training ******
        # (float) Learning rate for reward network adversarial training.
        reward_learning_rate=1e-4,
        # (float) Weight decay for reward network optimizer.
        reward_weight_decay=1e-5,
        # (float) Weight for L2 regularization on reward predictions.
        reward_regularization_weight=0.01,
        # (bool) Whether to enable adversarial training.
        enable_adversarial_training=True,
    )
    def _adversarial_reward_loss(self, policy_logits, improved_policy_batch, mask_batch):
        """
        Overview:
            Calculate the reward loss.
        """
        pass



    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``. Initialize the learn model, optimizer and MCTS utils.
        """
        assert self._cfg.optim_type in ['SGD', 'Adam', 'AdamW'], self._cfg.optim_type
        # NOTE: in board_games, for fixed lr 0.003, 'Adam' is better than 'SGD'.
        if self._cfg.optim_type == 'SGD':
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._cfg.learning_rate,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.weight_decay,
            )
        elif self._cfg.optim_type == 'Adam':
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._cfg.learning_rate, weight_decay=self._cfg.weight_decay
            )
        elif self._cfg.optim_type == 'AdamW':
            self._optimizer = configure_optimizers(model=self._model, weight_decay=self._cfg.weight_decay,
                                                   learning_rate=self._cfg.learning_rate, device_type=self._cfg.device)

        if self._cfg.piecewise_decay_lr_scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.target_update_freq}
        )
        self._learn_model = self._model

        if self._cfg.use_augmentation:
            self.image_transforms = ImageTransforms(
                self._cfg.augmentation,
                image_shape=(self._cfg.model.observation_shape[1], self._cfg.model.observation_shape[2])
            )
        self.value_support = DiscreteSupport(-self._cfg.model.support_scale, self._cfg.model.support_scale, delta=1)
        self.reward_support = DiscreteSupport(-self._cfg.model.support_scale, self._cfg.model.support_scale, delta=1)
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )
        self.kl_loss = KLDivLoss(reduction='none')
        
        # Initialize global reward network if available
        if GLOBAL_REWARD_AVAILABLE and global_reward_network is not None:
            try:
                global_reward_network.initialize_global_reward_network(
                    vocab_size=getattr(self._cfg, 'reward_vocab_size', 1000),
                    max_selfies_len=getattr(self._cfg, 'max_selfies_len', 100),
                    device=self._cfg.device,
                    checkpoint_path=getattr(self._cfg, 'reward_network_checkpoint', None)
                )
                self.reward_function = global_reward_network.get_reward_function()
                self.reward_network = global_reward_network.get_reward_network()
                
                # Initialize optimizer for reward network adversarial training
                if self.reward_network is not None:
                    self.reward_optimizer = optim.Adam(
                        self.reward_network.parameters(),
                        lr=getattr(self._cfg, 'reward_learning_rate', 1e-4),
                        weight_decay=getattr(self._cfg, 'reward_weight_decay', 1e-5)
                    )
                    print("[INFO] Reward network optimizer initialized for adversarial training")
                
                print("[INFO] Global reward network initialized in GAG MuZero policy")
            except Exception as e:
                print(f"[WARN] Failed to initialize global reward network in GAG MuZero: {e}")
                self.reward_function = None
                self.reward_network = None
                self.reward_optimizer = None
        else:
            self.reward_function = None
            self.reward_network = None
            self.reward_optimizer = None

    def _forward_learn(self, data: torch.Tensor) -> Dict[str, Union[float, int]]:
        """
        Overview:
            Simplified forward function for learning policy in learn mode.
            Focus on imitating step-wise policy distribution and value prediction only.
        Arguments:
            - data (:obj:`Tuple[torch.Tensor]`): The data sampled from replay buffer.
        Returns:
            - info_dict (:obj:`Dict[str, Union[float, int]]`): Basic learning statistics.
        """
        self._learn_model.train()
        
        current_batch, target_batch = data
        obs_batch_ori, action_batch, improved_policy_batch, mask_batch, indices, weights, make_time = current_batch
        target_reward, target_value, target_policy = target_batch

        obs_batch, _ = prepare_obs(obs_batch_ori, self._cfg)

        # Basic data preparation
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
        mask_batch = torch.from_numpy(mask_batch).to(self._cfg.device).float()
        target_value = torch.from_numpy(target_value.astype('float32')).to(self._cfg.device)
        improved_policy_batch = torch.from_numpy(improved_policy_batch).to(self._cfg.device).float()
        weights = torch.from_numpy(weights).to(self._cfg.device).float()

        target_value = target_value.view(self._cfg.batch_size, -1)

        # Initial inference
        network_output = self._learn_model.initial_inference(obs_batch)
        latent_state, _, value, policy_logits = mz_network_output_unpack(network_output)

        # Initialize losses
        policy_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        value_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)

        # Loss for initial step
        policy_loss += self.kl_loss(
            torch.log_softmax(policy_logits, dim=1),
            improved_policy_batch[:, 0]
        ).mean(dim=-1) * mask_batch[:, 0]
        
        value_loss += L1Loss(reduction='none')(
            value.squeeze(-1), target_value[:, 0]
        ) * mask_batch[:, 0]

        # Unroll steps
        for step_k in range(self._cfg.num_unroll_steps):
            network_output = self._learn_model.recurrent_inference(latent_state, action_batch[:, step_k])
            latent_state, _, value, policy_logits = mz_network_output_unpack(network_output)

            # Policy loss for this step
            policy_loss += self.kl_loss(
                torch.log_softmax(policy_logits, dim=1),
                improved_policy_batch[:, step_k + 1]
            ).mean(dim=-1) * mask_batch[:, step_k + 1]
            
            # Value loss for this step
            value_loss += L1Loss(reduction='none')(
                value.squeeze(-1), target_value[:, step_k + 1]
            ) * mask_batch[:, step_k + 1]

        # Total loss
        total_loss = (
            self._cfg.policy_loss_weight * policy_loss +
            self._cfg.value_loss_weight * value_loss
        )
        weighted_total_loss = (weights * total_loss).mean()

        # Optimization step
        self._optimizer.zero_grad()
        weighted_total_loss.backward()
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self._learn_model.parameters(), self._cfg.grad_clip_value
        )
        self._optimizer.step()
        
        if self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # Generative Adversarial Training with Preference Learning
        reward_loss = 0.0
        adversarial_loss = 0.0
        if (self.reward_network is not None and hasattr(self, 'reward_function') and 
            getattr(self._cfg, 'enable_adversarial_training', True)):
            try:
                # Extract spectrum data and SELFIES from observation batch
                positive_data, negative_data = self._extract_adversarial_training_data(
                    obs_batch, latent_state, action_batch, mask_batch
                )
                
                if positive_data is not None and negative_data is not None:
                    # Compute preference learning loss
                    preference_loss = self._compute_preference_loss(positive_data, negative_data)
                    
                    # Update reward network with adversarial training
                    if hasattr(self, 'reward_optimizer'):
                        self.reward_optimizer.zero_grad()
                        preference_loss.backward(retain_graph=True)
                        self.reward_optimizer.step()
                        
                        reward_loss = preference_loss.item()
                        adversarial_loss = preference_loss.item()
                    
                # Update step count for logging
                if hasattr(self, '_reward_network_step_count'):
                    self._reward_network_step_count += 1
                else:
                    self._reward_network_step_count = 1
                    
                if self._reward_network_step_count % 100 == 0:
                    print(f"[INFO] Adversarial reward training step {self._reward_network_step_count}, loss: {reward_loss:.4f}")
                    
            except Exception as e:
                print(f"[WARN] Error in adversarial reward network training: {e}")
                import traceback
                traceback.print_exc()

        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_loss': total_loss.mean().item(),
            'policy_loss': policy_loss.mean().item(),
            'value_loss': value_loss.mean().item(),
            'reward_loss': reward_loss,
            'adversarial_loss': adversarial_loss,
            'total_grad_norm_before_clip': total_grad_norm_before_clip.item()
        }

    def _extract_adversarial_training_data(self, obs_batch, latent_state, action_batch, mask_batch):
        """
        Extract positive (training data) and negative (generated) examples for adversarial training.
        
        Args:
            obs_batch: Observation batch containing spectrum + formula + selfies
            latent_state: Current latent state from the model
            action_batch: Action batch for generating negative examples
            mask_batch: Mask for valid timesteps
            
        Returns:
            tuple: (positive_data, negative_data) where each contains:
                - selfies_tokens: Tokenized SELFIES
                - selfies_mask: Attention mask for SELFIES
                - spectrum_batch: Spectrum data in reward network format
        """
        try:
            batch_size = obs_batch.shape[0]
            
            # Extract components from observation (spectrum + selfies + formula)
            # Based on massgymenv.py: spectrum (4096) + selfies (100) + formula (50) = 4246
            spectrum_embeds = obs_batch[:, :4096]  # Pre-computed spectrum embeddings
            selfies_tokens = obs_batch[:, 4096:4196].long()  # SELFIES tokens
            formula_tokens = obs_batch[:, 4196:4246].long()  # Formula tokens
            
            # Positive examples: Use the training data (current observations)
            positive_data = self._prepare_reward_network_data(
                spectrum_embeds, selfies_tokens, formula_tokens, is_positive=True
            )
            
            # Negative examples: Generate from current policy
            negative_data = self._generate_negative_examples(
                spectrum_embeds, latent_state, action_batch, mask_batch
            )
            
            return positive_data, negative_data
            
        except Exception as e:
            print(f"[WARN] Error extracting adversarial training data: {e}")
            return None, None

    def _prepare_reward_network_data(self, spectrum_embeds, selfies_tokens, formula_tokens, is_positive=True):
        """
        Prepare data in the format expected by the reward network.
        
        Args:
            spectrum_embeds: Pre-computed spectrum embeddings [batch_size, 4096]
            selfies_tokens: SELFIES token IDs [batch_size, max_len]
            formula_tokens: Formula token IDs [batch_size, formula_max_len]
            is_positive: Whether this is positive training data
            
        Returns:
            dict: Data formatted for reward network input
        """
        try:
            batch_size = spectrum_embeds.shape[0]
            
            # Create attention mask for SELFIES (non-zero tokens are valid)
            selfies_mask = (selfies_tokens != 0).long()
            
            # For the reward network, we need to create spectrum_batch in the format expected by PeakFormula
            # Since we have pre-computed embeddings, we'll create a dummy spectrum batch
            # and use the embeddings directly in the reward network
            spectrum_batch = {
                'peaks': torch.zeros(batch_size, 100, 2).to(spectrum_embeds.device),  # Dummy peaks
                'num_peaks': torch.full((batch_size,), 100).to(spectrum_embeds.device),
                'precursor_mz': torch.zeros(batch_size).to(spectrum_embeds.device),
                'spectrum_embeds': spectrum_embeds,  # Use pre-computed embeddings
            }
            
            return {
                'selfies_tokens': selfies_tokens,
                'selfies_mask': selfies_mask,
                'spectrum_batch': spectrum_batch,
                'is_positive': is_positive
            }
            
        except Exception as e:
            print(f"[WARN] Error preparing reward network data: {e}")
            return None

    def _generate_negative_examples(self, spectrum_embeds, latent_state, action_batch, mask_batch):
        """
        Generate negative examples by sampling from the current policy.
        
        Args:
            spectrum_embeds: Spectrum embeddings [batch_size, 4096]
            latent_state: Current latent state from model
            action_batch: Action batch for reference
            mask_batch: Mask for valid timesteps
            
        Returns:
            dict: Negative examples in reward network format
        """
        try:
            batch_size = spectrum_embeds.shape[0]
            device = spectrum_embeds.device
            
            # Generate negative SELFIES by sampling from the current policy
            # This is a simplified approach - in practice, you might want to use MCTS rollouts
            negative_selfies_tokens = self._sample_negative_selfies(batch_size, device)
            
            # Create attention mask for negative SELFIES
            negative_selfies_mask = (negative_selfies_tokens != 0).long()
            
            # Use the same spectrum embeddings but with generated (negative) SELFIES
            spectrum_batch = {
                'peaks': torch.zeros(batch_size, 100, 2).to(device),  # Dummy peaks
                'num_peaks': torch.full((batch_size,), 100).to(device),
                'precursor_mz': torch.zeros(batch_size).to(device),
                'spectrum_embeds': spectrum_embeds,  # Same spectrum, different molecules
            }
            
            return {
                'selfies_tokens': negative_selfies_tokens,
                'selfies_mask': negative_selfies_mask,
                'spectrum_batch': spectrum_batch,
                'is_positive': False
            }
            
        except Exception as e:
            print(f"[WARN] Error generating negative examples: {e}")
            return None

    def _sample_negative_selfies(self, batch_size, device):
        """
        Sample negative SELFIES sequences from the current policy.
        
        Args:
            batch_size: Number of negative examples to generate
            device: Device to create tensors on
            
        Returns:
            torch.Tensor: Negative SELFIES token sequences [batch_size, max_len]
        """
        try:
            max_len = getattr(self._cfg, 'max_selfies_len', 100)
            vocab_size = min(getattr(self._cfg, 'reward_vocab_size', 1000), 500)  # Limit vocab size
            
            # Simple random sampling for negative examples
            # In practice, you might want to use the current policy to generate more realistic negatives
            negative_tokens = torch.randint(
                1, vocab_size, (batch_size, max_len), device=device
            )
            
            # Add some padding to make it more realistic
            for i in range(batch_size):
                # Random length between 10 and max_len
                seq_len = torch.randint(10, max_len, (1,)).item()
                negative_tokens[i, seq_len:] = 0  # Pad with zeros
            
            return negative_tokens
            
        except Exception as e:
            print(f"[WARN] Error sampling negative SELFIES: {e}")
            # Return dummy tokens as fallback
            max_len = getattr(self._cfg, 'max_selfies_len', 100)
            return torch.zeros(batch_size, max_len, device=device, dtype=torch.long)

    def _compute_preference_loss(self, positive_data, negative_data):
        """
        Compute RLHF-style pairwise preference loss for adversarial training.
        
        Uses the Bradley-Terry model from RLHF where the probability that positive
        example is preferred over negative example is modeled as:
        P(positive > negative) = sigmoid(reward(positive) - reward(negative))
        
        The loss is: -log(sigmoid(reward(positive) - reward(negative)))
        
        Args:
            positive_data: Positive training examples (preferred)
            negative_data: Negative generated examples (not preferred)
            
        Returns:
            torch.Tensor: RLHF-style pairwise preference loss
        """
        try:
            # Get similarity scores from reward network (these act as rewards)
            positive_rewards = self.reward_network.predict_similarity(
                positive_data['selfies_tokens'],
                positive_data['selfies_mask'],
                positive_data['spectrum_batch']
            )
            
            negative_rewards = self.reward_network.predict_similarity(
                negative_data['selfies_tokens'],
                negative_data['selfies_mask'],
                negative_data['spectrum_batch']
            )
            
            # RLHF-style pairwise preference loss using Bradley-Terry model
            # P(positive > negative) = sigmoid(reward_positive - reward_negative)
            # Loss = -log(P(positive > negative)) = -log(sigmoid(reward_positive - reward_negative))
            # This is equivalent to: log(1 + exp(-(reward_positive - reward_negative)))
            # Which is the same as: F.softplus(-(reward_positive - reward_negative))
            
            reward_diff = positive_rewards - negative_rewards
            preference_loss = F.softplus(-reward_diff).mean()
            
            # Optional: Add regularization to prevent reward collapse
            # Encourage diversity in reward predictions
            reward_regularization = getattr(self._cfg, 'reward_regularization_weight', 0.01)
            if reward_regularization > 0:
                # L2 regularization on rewards to prevent them from becoming too large
                positive_reg = (positive_rewards ** 2).mean()
                negative_reg = (negative_rewards ** 2).mean()
                regularization_loss = reward_regularization * (positive_reg + negative_reg)
            else:
                regularization_loss = 0.0
            
            total_loss = preference_loss + regularization_loss
            
            # Log some statistics for monitoring
            if hasattr(self, '_preference_step_count'):
                self._preference_step_count += 1
            else:
                self._preference_step_count = 1
                
            if self._preference_step_count % 100 == 0:
                with torch.no_grad():
                    accuracy = (reward_diff > 0).float().mean()
                    print(f"[INFO] Preference accuracy: {accuracy.item():.3f}, "
                          f"Avg reward diff: {reward_diff.mean().item():.3f}, "
                          f"Loss: {preference_loss.item():.4f}")
            
            return total_loss
            
        except Exception as e:
            print(f"[WARN] Error computing RLHF preference loss: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=self._cfg.device, requires_grad=True)

    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            ready_env_id: np.array = None,
            **kwargs,
    ) -> Dict:
        """
        Overview:
            Simplified forward function for collecting data in collect mode.
            Focus on basic MCTS search and action selection.
        Arguments:
            - data (:obj:`torch.Tensor`): The input observation data.
            - action_mask (:obj:`list`): The action mask for valid actions.
            - temperature (:obj:`float`): The temperature for action selection.
            - to_play (:obj:`List`): The player to play.
            - ready_env_id (:obj:`np.array`): The environment IDs ready to collect.
        Returns:
            - output (:obj:`Dict[int, Any]`): Basic collection results with action and policy info.
        """
        self._collect_model.eval()
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}

        with torch.no_grad():
            # Initial inference
            network_output = self._collect_model.initial_inference(data)
            latent_state_roots, _, pred_values, policy_logits = mz_network_output_unpack(network_output)

            # Convert to numpy for MCTS
            pred_values = pred_values.detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            # Prepare legal actions
            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_collect_env_num)]
            
            # Simple noise for exploration
            noises = [
                np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                    ).astype(np.float32).tolist() for j in range(active_collect_env_num)
            ]
            
            # Initialize MCTS roots
            if self._cfg.mcts_ctree:
                roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
            else:
                roots = MCTSPtree.roots(active_collect_env_num, legal_actions)

            # Prepare and search
            roots.prepare(self._cfg.root_noise_weight, noises, [0] * active_collect_env_num, 
                         list(pred_values), policy_logits, to_play)
            self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)

            # Get improved policies from MCTS
            roots_improved_policy_probs = roots.get_policies(self._cfg.discount_factor,
                                                             self._cfg.model.action_space_size)
            roots_values = roots.get_values()

            for i, env_id in enumerate(ready_env_id):
                improved_policy_probs = roots_improved_policy_probs[i]
                value = roots_values[i]
                
                # Select action based on improved policy
                valid_probs = np.where(action_mask[i] == 1.0, improved_policy_probs, 0.0)
                action = np.argmax(valid_probs)

                output[env_id] = {
                    'action': action,
                    'improved_policy_probs': improved_policy_probs,
                    'predicted_value': pred_values[i],
                    'searched_value': value,
                }

        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: List = [-1],
                      ready_env_id: np.array = None, **kwargs) -> Dict:
        """
        Overview:
            The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
            Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            # data shape [B, S x C, W, H], e.g. {Tensor:(B, 12, 96, 96)}
            network_output = self._collect_model.initial_inference(data)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape（B, 1）
                latent_state_roots = latent_state_roots.detach().cpu().numpy()
                policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

            legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(active_eval_env_num)]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(active_eval_env_num, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)
            roots.prepare_no_noise(reward_roots, list(pred_values), policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            # ==============================================================
            # The core difference between GumbelMuZero and MuZero
            # ==============================================================
            # Gumbel MuZero selects the action according to the improved policy
            roots_improved_policy_probs = roots.get_policies(self._cfg.discount_factor,
                                                             self._cfg.model.action_space_size)  # new policy constructed with completed Q in gumbel muzero
            roots_improved_policy_probs = np.array(roots_improved_policy_probs)

            for i, env_id in enumerate(ready_env_id):
                distributions, value, improved_policy_probs = roots_visit_count_distributions[i], roots_values[i], \
                roots_improved_policy_probs[i]
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                #  Setting deterministic=True implies choosing the action with the highest value (argmax) rather than
                # sampling during the evaluation phase.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the
                # entire action set.
                # action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                valid_value = np.where(action_mask[i] == 1.0, improved_policy_probs, 0.0)
                # print("debug: valid_value: ", valid_value)
                action = np.argmax([v for v in valid_value])

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }

        return output