import numpy as np
import gymnasium as gym
from ding.utils import ENV_REGISTRY
from .massgymenv import MassGymEnv
import copy
from typing import List
from ding.envs.env.base_env import BaseEnvTimestep

@ENV_REGISTRY.register('massgym_lightzero')
class MassGymLightZeroEnv(gym.Wrapper):
    """
    LightZero wrapper for MassGymEnv.
    This wrapper ensures that the environment returns observations in the format expected by LightZero.
    """
    def __init__(self, cfg):
        complete_cfg = copy.deepcopy(cfg)
        
        complete_cfg.setdefault('obs_type', 'dict_encoded_selfies')
        complete_cfg.setdefault('reward_type', 'cosine_similarity')
        complete_cfg.setdefault('render_mode', 'text_mode')
        complete_cfg.setdefault('max_episode_steps', 100)
        complete_cfg.setdefault('channel_last', True)
        complete_cfg.setdefault('need_flatten', False)
        
        env = MassGymEnv(complete_cfg)
        super().__init__(env)
        self._cfg = complete_cfg
        
        vocab_size = env.vocab_size
        action_size = len(env.actions_list)
        max_len = env.max_len
        # print(f"initialize the MassGymLightZeroEnv wrapper: vocab_size={vocab_size}, action_size={action_size}, max_len={max_len}")
        
    def _process_observation_for_mlp(self, observation):
        if len(observation.shape) == 3 and observation.shape[0] == 1:
            observation = observation[0]
        
        # print(f"观察原始形状: {observation.shape}")
        
        flattened = observation.reshape(-1)
        # print(f"展平后形状: {flattened.shape}")
        
        return flattened

    def reset(self, *args, **kwargs):
        """
        rewrite the reset method, ensure the observation dictionary returned is符合LightZero要求的
        """
        obs_timestep = self.env.reset(*args, **kwargs)
        
        if hasattr(obs_timestep, 'obs'):
            obs = obs_timestep.obs
        else:
            obs = obs_timestep
            
        if isinstance(obs, dict) and 'observation' in obs:
            observation = obs['observation']
        else:
            observation = obs
    
        if hasattr(self.env, 'get_action_mask'):
            action_mask = self.env.get_action_mask()
        elif isinstance(obs, dict) and 'action_mask' in obs:
            action_mask = obs['action_mask']
        else:
            action_mask = np.ones(self.env.action_space.n, dtype=np.int8)


        observation = self._process_observation_for_mlp(observation)
            
        lightzero_obs = {
            'observation': observation,
            'action_mask': action_mask,
            'to_play': -1,
            'chance': 0.0
        }
        
        return lightzero_obs
        
    def step(self, action):
        result = self.env.step(action)
        
        if hasattr(result, 'obs'):
            obs = result.obs
            reward = result.reward.item() if hasattr(result.reward, 'item') else result.reward
            done = result.done
            info = result.info
        else:
            obs, reward, done, info = result
            
        if isinstance(obs, dict) and 'observation' in obs:
            observation = obs['observation']
        else:
            observation = obs
        
        if hasattr(self.env, 'get_action_mask'):
            action_mask = self.env.get_action_mask()
        elif isinstance(obs, dict) and 'action_mask' in obs:
            action_mask = obs['action_mask']
        else:
            action_mask = np.ones(self.env.action_space.n, dtype=np.int8)
            
        observation = self._process_observation_for_mlp(observation)
            
        lightzero_obs = {
            'observation': observation,
            'action_mask': action_mask,
            'to_play': -1,
            'chance': 0.0
        }
        
        
        return BaseEnvTimestep(lightzero_obs, reward, done, info)
    
    def seed(self, seed=None, dynamic_seed=None, **kwargs):
        return self.env.seed(seed, dynamic_seed, **kwargs)
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def __repr__(self):
        return "LightZero Wrapper for MassGym"
        
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        return MassGymEnv.create_collector_env_cfg(cfg)
    
    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        return MassGymEnv.create_evaluator_env_cfg(cfg)
