from easydict import EasyDict
import numpy as np
import os

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 1
n_episode = 1
evaluator_env_num = 1
num_simulations = 20
update_per_collect = 100
batch_size = 1
max_env_step = int(300)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================


massspecgym_gumbel_muzero_config = dict(
    exp_name=f'data_muzero/massspecgym_gumbel_muzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_seed0',
    env=dict(
        env_id='massgym',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        
        render_mode='text_mode',
        replay_format='svg',
        replay_name_suffix='eval',
        replay_path=None,
        obs_type='fingerprint',
        reward_normalize=False,
        reward_norm_scale=1.0,
        reward_type='cosine_similarity',
        max_len=100,
        max_episode_steps=1000,
        channel_last=True,
        need_flatten=False,

        use_all_atom_tokens=True,
        use_bonded_atom_tokens=True,
        use_branch_tokens=True,  
        use_ring_tokens=True,  
        use_element_groups=False,  
        use_massspecgym_data=False,
        formula_masking=True,
          
    ),
    policy=dict(
        model=dict(
            observation_shape=1100, #
            action_space_size=62,    # action size  
            model_type='mlp',              
            lstm_hidden_size=128,           
            latent_state_dim=128,           
            self_supervised_learning_loss=False,  
            discrete_action_encoding_type='one_hot',
            norm_type='LN',
            fc_reward_size=64,
            fc_value_size=64,
            fc_policy_size=64,
        ),
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        action_type='varied_action_space',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        max_num_considered_actions=2,
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e2),
        replay_buffer_size=int(1e6),  
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

massspecgym_gumbel_muzero_config = EasyDict(massspecgym_gumbel_muzero_config)
main_config = massspecgym_gumbel_muzero_config

massspecgym_gumbel_muzero_create_config = dict(
    env=dict(
        type='massgym_lightzero',
        import_names=['zoo.masspecgym.envs.massgym_wrapper'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='gumbel_muzero',
        import_names=['lzero.policy.gumbel_muzero'],
    ),
)
massspecgym_gumbel_muzero_create_config = EasyDict(massspecgym_gumbel_muzero_create_config)
create_config = massspecgym_gumbel_muzero_create_config

if __name__ == "__main__":
    entry_type = "train_muzero" 

    if entry_type == "train_muzero":
        from lzero.entry import train_muzero
    elif entry_type == "train_muzero_with_gym_env":
        from lzero.entry import train_muzero_with_gym_env as train_muzero

    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.model_path, max_env_step=max_env_step)