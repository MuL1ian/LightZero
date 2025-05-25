import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
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
from lzero.policy.muzero import MuZeroPolicy, GumbelMuZeroPolicy

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
            Initialize the learn mode of the policy.
        """
        pass

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

        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_loss': total_loss.mean().item(),
            'policy_loss': policy_loss.mean().item(),
            'value_loss': value_loss.mean().item(),
            'total_grad_norm_before_clip': total_grad_norm_before_clip.item()
        }

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

