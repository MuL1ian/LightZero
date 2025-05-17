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
        
        complete_cfg.setdefault('obs_type', 'fingerprint')
        complete_cfg.setdefault('reward_type', 'cosine_similarity')
        complete_cfg.setdefault('render_mode', 'text_mode')
        complete_cfg.setdefault('max_episode_steps', 100)
        complete_cfg.setdefault('channel_last', True)
        complete_cfg.setdefault('need_flatten', True)
        
        env = MassGymEnv(complete_cfg)
        super().__init__(env)
        self._cfg = complete_cfg
        
        self.action_size = len(env.actions_list)
        self.max_len = env.max_len

    # def _process_observation_for_mlp(self, observation):
    #     if isinstance(observation, np.ndarray):
    #         flattened = observation
    #     else:  # torch tensor
    #         flattened = observation.cpu().numpy()
    #     return flattened

    def _process_observation_for_mlp(self, observation): # Batch * 4196
        return observation

    def reset(self, *args, **kwargs):
        obs_timestep = self.env.reset(*args, **kwargs)
        obs = obs_timestep.obs

        observation = self._process_observation_for_mlp(obs['observation'])
        action_mask = obs['action_mask']

        lightzero_obs = {
            'observation': observation,
            'action_mask': action_mask,
            'to_play': obs.get('to_play', -1),
            'chance': obs.get('chance', 0.0),
            'timestep': obs.get('timestep', 0)
        }
        assert observation.shape[-1] == 4196, "The last dimension of the observation must be 4196, but got {}".format(observation.shape[-1])
        return lightzero_obs

    def step(self, action):
        obs_timestep = self.env.step(action)
        obs = obs_timestep.obs
        reward = obs_timestep.reward.item() if hasattr(obs_timestep.reward, 'item') else obs_timestep.reward
        done = obs_timestep.done
        info = obs_timestep.info

        observation = self._process_observation_for_mlp(obs['observation'])
        action_mask = obs['action_mask']


        lightzero_obs = {
            'observation': observation,
            'action_mask': action_mask,
            'to_play': obs.get('to_play', -1),
            'chance': obs.get('chance', 0.0),
            'timestep': obs.get('timestep', 0)
        }
        assert observation.shape[-1] == 4196, "The last dimension of the observation must be 4196, but got {}".format(observation.shape[-1])

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
