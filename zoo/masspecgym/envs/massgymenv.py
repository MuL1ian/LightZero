import copy
import logging
import os
import sys
from typing import List
import gymnasium as gym
import imageio
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from ding.envs import BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces
from gymnasium.utils import seeding
import selfies as sf
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import Draw
from typing import Any, Dict, List, Union
import re
import pandas as pd
import torch as th
from datasets import load_dataset
import numpy as np
import warnings
import torch
import random
from torch.utils.data import Dataset



class DebugSpectrumDataset(Dataset):
    def __init__(self, file_path="/Users/boyeniu/Desktop/nus_course/NUS_JAX/MassEnv/policy_model/debug_spectrum_embeds.pt"):
        self.data = torch.load(file_path)

        list_lengths = []
        for key, value in self.data.items():
            if isinstance(value, list):
                list_lengths.append(len(value))
            elif isinstance(value, torch.Tensor) and len(value.shape) > 0:
                list_lengths.append(value.shape[0])
        if len(set(list_lengths)) > 1:
            print(f"Error: data fields have different lengths: {list_lengths}")

        self.size = list_lengths[0] if list_lengths else 0
        print(f"Loaded {self.size} samples")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        result = {}
        for key, value in self.data.items():
            if isinstance(value, list):
                if idx < len(value):
                    result[key] = value[idx]
            elif isinstance(value, torch.Tensor) and len(value.shape) > 0:
                if idx < value.shape[0]:
                    result[key] = value[idx]
        return result

    def random_sample(self):
        idx = random.randint(0, self.size - 1)
        return self.__getitem__(idx)



def extract_element(token):
    """
    Extract the element symbol from a SELFIES token.
    
    Args:
        token (str): A SELFIES token string.
        
    Returns:
        str: The extracted element symbol, or None if no element is found.
    """
    match = re.match(r'^\[([A-Za-z]+)', token)
    return match.group(1) if match else None

def is_valid_token(token):
    """
    Check if a SELFIES token represents a valid chemical element.
    
    Args:
        token (str): A SELFIES token string.
        
    Returns:
        bool: True if the token represents a valid chemical element, False otherwise.
    """
    element = extract_element(token)
    if element and element not in ['Ring', 'Branch']:
        try:
            atom = Chem.Atom(element)
            return atom.GetAtomicNum() > 0
        except:
            return False
    return False

def filter_valid_atoms(atoms):
    """
    Filter a list of SELFIES tokens to keep only valid chemical elements.
    
    Args:
        atoms (list): List of SELFIES tokens.
        
    Returns:
        list: Filtered list containing only valid chemical element tokens.
    """
    valid_atoms = [atom for atom in atoms if is_valid_token(atom)]
    return valid_atoms

def get_bond_constraints():
    """
    Get bond constraints for each atom element based on RDKit's valence data.
    
    Returns:
        dict: A dictionary mapping element symbols to their maximum valence (number of bonds).
    """
    atoms = sf.get_semantic_robust_alphabet()
    bond_constraints = {}
    for atom in atoms:
        element = extract_element(atom)
        if element and is_valid_token(f"[{element}]"): 
            try:
                max_valence = Chem.GetPeriodicTable().GetDefaultValence(Chem.Atom(element).GetAtomicNum())
                if max_valence is not None:
                    bond_constraints[element] = max_valence
            except:
                bond_constraints[element] = None 
    return bond_constraints

def split_atoms(atom_tokens):
    """
    Split a list of SELFIES tokens into different categories.
    
    Args:
        atom_tokens (list): List of SELFIES tokens.
        
    Returns:
        tuple: Four lists containing pure atoms, bonded atoms, branch tokens, and ring tokens.
    """
    pure_atom_tokens = []
    bonded_atom_tokens = []
    branch_tokens = []
    ring_tokens = []

    default_branch = {"[Branch1]", "[Branch2]", "[Branch3]"}
    default_ring = {"[Ring1]", "[Ring2]", "[Ring3]"}

    for token in atom_tokens:
        if token in default_branch:
            branch_tokens.append(token)
        elif token in default_ring:
            ring_tokens.append(token)
        elif token.startswith("[=") or token.startswith("[#"):
            if "Branch" in token or "Ring" in token:
                continue
            else:
                bonded_atom_tokens.append(token)
        else:
            pure_atom_tokens.append(token)
    return pure_atom_tokens, bonded_atom_tokens, branch_tokens, ring_tokens

all_atom_tokens = sf.get_semantic_robust_alphabet()
#{'[Branch2]', '[=P]', '[=S]', '[#P-1]', '[=B-1]', '[#N+1]', '[N+1]', '[=N-1]', '[=Ring3]', '[#O+1]', '[#C+1]', '[=N+1]', '[O+1]', '[P-1]', '[=S+1]', '[=P+1]', '[O-1]', '[#C-1]', '[#B-1]', '[=S-1]', '[H]}

atom_tokens, bonded_atom_tokens, branch_tokens, ring_tokens = split_atoms(all_atom_tokens)

valid_atoms = filter_valid_atoms(atom_tokens)


bond_constraints = get_bond_constraints()
# Example of bond constraints: {'N': 3, 'O': 2, 'P': 3, 'H': 1, 'B': 3, 'Br': 1, 'S': 2, 'C': 4, 'Cl': 1, 'F': 1, 'I': 1}






@ENV_REGISTRY.register('massgym')
class MassGymEnv(gym.Env):

    config = dict(
        env_id="mass_spec_env",

        render_mode=None,
        
        obs_type='fingerprint',
        
        reward_normalize=False,
        
        reward_norm_scale=1.0,
        
        reward_type='cosine_similarity',
        
        target_spectrum={
            'embeds': torch.tensor([]),
            'formulas': ''
        },
        
        delay_reward_step=0,
        
        prob_random_agent=0.,
        
        max_episode_steps=100,
        
        is_collect=True,
        
        ignore_legal_actions=False,

        need_flatten=False,
        
        atom_tokens=[],
        pure_atom_tokens=[],
        bonded_atom_tokens=[],
        branch_tokens=[],
        ring_tokens=[],
        
        max_len=10,
        formula_masking=True,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._env_id = cfg.get('env_id', 'mass_spec_env')
        
        self.replay_format = cfg.get('replay_format', 'svg')
        self.replay_name_suffix = cfg.get('replay_name_suffix', 'eval')
        self.replay_path = cfg.get('replay_path', None)
        self.render_mode = cfg.get('render_mode', None)
    
        self.channel_last = cfg.get('channel_last', True)

        self.obs_type = cfg.get('obs_type', 'fingerprint')
        self.reward_type = cfg.get('reward_type', 'cosine_similarity')
        self.reward_normalize = cfg.get('reward_normalize', False)
        self.reward_norm_scale = cfg.get('reward_norm_scale', 1.0)
        assert self.reward_type in ['cosine_similarity']
        
        self.max_episode_steps = cfg.get('max_episode_steps', 100)
        self.is_collect = cfg.get('is_collect', True)
        self.ignore_legal_actions = cfg.get('ignore_legal_actions', False)
        self.need_flatten = cfg.get('need_flatten', False)

        self.formula_masking = cfg.get('formula_masking', True)
        
        self.chance = 0.0
        
        self.frames = []

        self.target_embeds = cfg.get('target_spectrum', {}).get('embeds', [])
        self.target_formula = cfg.get('target_spectrum', {}).get('formulas', '')
        
        use_all_tokens = cfg.get('use_all_atom_tokens', False)
         
        if use_all_tokens:
            all_atom_tokens = sf.get_semantic_robust_alphabet()
            
            atom_tokens, bonded_atom_tokens, branch_tokens, ring_tokens = split_atoms(all_atom_tokens)
            
            use_branch = cfg.get('use_branch_tokens', True)
            use_ring = cfg.get('use_ring_tokens', True)
            
            self.atom_tokens = filter_valid_atoms(atom_tokens)
            self.bonded_atom_tokens = bonded_atom_tokens
            self.branch_tokens = branch_tokens if use_branch else []
            self.ring_tokens = ring_tokens if use_ring else []
            
        else:
            self.atom_tokens = cfg.get('atom_tokens', [])
            self.bonded_atom_tokens = cfg.get('bonded_atom_tokens', [])
            self.branch_tokens = cfg.get('branch_tokens', [])
            self.ring_tokens = cfg.get('ring_tokens', [])
        
        
        self.element_groups = cfg.get('element_groups', [])
        
        # Tokens for terminating molecule building or removing the last atom
        self.remove_token = "<REMOVE>"
        self.end_token = "<END>"
        
        # Combine all possible actions into a single list
        self.actions_list = (self.atom_tokens + 
                             self.bonded_atom_tokens +
                             self.branch_tokens + 
                             self.ring_tokens + 
                             self.element_groups +
                             [self.remove_token, self.end_token])
        

        self.max_len = cfg.get('max_len', 100)
        
        
        self.episode_return = 0
        self.episode_length = 0
        self.should_done = False

        # set the action space and observation space for the gym interface
        self._action_space = spaces.Discrete(len(self.actions_list))
        
        self._reward_range = (0., 1.)
        
        # initialize the state
        self.current_selfies = ""
        self.bond_counts = []
        self.smiles = ""
        

        self.bond_constraints = cfg.get('bond_constraints', get_bond_constraints())
        
        self.train_info = DebugSpectrumDataset()

        self.reset()
        self._init_flag = True

    
    def random_massspecgym_data(self):

        sample = self.train_info.random_sample()
        embeds = sample['embeds']
        formula = sample['formulas']
        
        self.target_spectrum = {
            'embeds': embeds,
            'formulas': formula
        }
        self.smiles = sample['smiles']

    
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            BaseEnvTimestep: Initial observation, reward, done flag, and info dictionary.
        """
        self.episode_length = 0
        self.current_selfies = ""
        self.bond_counts = []
        self.episode_return = 0
        self._final_eval_reward = 0.0
        self.should_done = False
        
        self.random_massspecgym_data()
        

        action_mask = self.get_valid_actions().astype(np.int8)


        obs_dict = {
            'observation': self.target_spectrum['embeds'],
            'prefix': self.current_selfies,
            'action_mask': action_mask,  
            'to_play': -1,
            'chance': self.chance
        }
        
        if self.render_mode is not None:
            self.render(self.render_mode)
        
        return BaseEnvTimestep(obs_dict, to_ndarray([0.0], dtype=np.float32), False, {})



    def _get_allowed_elements_from_formula(self):
        """
        Extract allowed elements from a formula and return a set of all variants
        
        Args:
            formula (str): The formula string, e.g. "C6H12O6"
                
        Returns:
            set: The set of allowed element symbols, including all variants (with charges and different bond types)
        """
        formula = self.target_spectrum.get('formulas')


        if formula is None:
            return set()
        
        base_elements = set()
        i = 0
        
        element_variants = {
            'C': ['[C]', '[C+1]', '[C-1]', '[=C]', '[#C]', '[=C+1]', '[=C-1]', '[#C+1]', '[#C-1]'],
            'N': ['[N]', '[N+1]', '[N-1]', '[=N]', '[#N]', '[=N+1]', '[=N-1]', '[#N+1]'],
            'O': ['[O]', '[O+1]', '[O-1]', '[=O]', '[#O]', '[=O+1]', '[#O+1]'],
            'S': ['[S]', '[S+1]', '[S-1]', '[=S]', '[#S]', '[=S+1]', '[#S+1]', '[=S-1]', '[#S-1]'],
            'P': ['[P]', '[P+1]', '[P-1]', '[=P]', '[#P]', '[=P+1]', '[#P+1]', '[=P-1]', '[#P-1]'],
            'B': ['[B]', '[B+1]', '[B-1]', '[=B]', '[#B]', '[=B+1]', '[=B-1]', '[#B-1]'],
            'F': ['[F]'],
            'I': ['[I]'],
            'Cl': ['[Cl]'],
            'Br': ['[Br]'],
            'H': ['[H]']
        }
        
        ring_variants = {
            'Ring': ['[Ring1]', '[Ring2]', '[Ring3]']
        }
        
        branch_variants = {
            'Branch': ['[Branch1]', '[Branch2]', '[Branch3]']
        }
    
        while i < len(formula):

            if i + 1 < len(formula) and formula[i].isupper() and formula[i+1].islower():
                symbol = formula[i:i+2]
                i += 2
            
            elif formula[i].isupper():
                symbol = formula[i]
                i += 1
            
            elif formula[i].isdigit():
                i += 1
                continue
            
            else:
                i += 1
                continue
            
            base_elements.add(symbol)
            
            while i < len(formula) and formula[i].isdigit():
                i += 1
                
        allowed_tokens = set()
        for element in base_elements:
            if element in element_variants:
                allowed_tokens.update(element_variants[element])
        
        for variants in ring_variants.values():
            allowed_tokens.update(variants)
        for variants in branch_variants.values():
            allowed_tokens.update(variants)
        


        return allowed_tokens

    def get_valid_actions(self): 
        """
        Generate a boolean mask over the full action space indicating which actions are valid.
        This does not change the size of the action space (always same as len(self.actions_list)).
        """
        mask = np.ones(len(self.actions_list), dtype=np.bool_)

        try:
            # Only filter if formula_masking is enabled and formula exists
            if self.formula_masking and hasattr(self, 'target_spectrum') and self.target_spectrum.get('formulas'):
                formula = self.target_spectrum.get('formulas')
                allowed_elements = self._get_allowed_elements_from_formula()
                for i, action in enumerate(self.actions_list):
                    if action in self.atom_tokens or action in self.bonded_atom_tokens:
                        if action not in allowed_elements:
                            mask[i] = False

        except Exception as e:
            print(f"[WARN] get_valid_actions formula filtering failed: {e}")
            # fallback: allow all
            mask[:] = True

        # End token is always allowed
        if self.end_token in self.actions_list:
            idx = self.actions_list.index(self.end_token)
            mask[idx] = True

        if not self.current_selfies and self.remove_token in self.actions_list:
            idx = self.actions_list.index(self.remove_token)
            mask[idx] = False

        return mask



    def reset_with_info(self):
        """
        Reset the environment and return additional information.
        
        Returns:
            tuple: Initial observation and info dictionary containing episode information.
        """
        timestep = self.reset()
        info = {
            'episode_info': {},
            'eval_episode_return': 0.0,
            'current_molecule': ""
        }
        return timestep.obs, info

    def step(self, action):
        """
        Execute a step in the environment using the given action.
        
        Args:
            action (int): The index of the action to take.
            
        Returns:
            BaseEnvTimestep: Next observation, reward, done flag, and info dictionary.
        """
        self.episode_length += 1
        
        action_mask = self.get_valid_actions().astype(np.int8)
        
        # Execute the action
        action_name = self.actions_list[action]
        raw_reward = 0.0
        info = {}
        done = False
        
        if action_name == self.remove_token and len(self.bond_counts) == 0:
            raw_reward = -0.1
        else:
            if action_name == self.end_token:
                done = True
                self.should_done = True
                if self.current_selfies == sf.encoder(self.smiles):  #selfes to smiles
                    raw_reward = 1.0
                else:
                    raw_reward = 0.0

            # Handle atom addition
            elif action_name in self.actions_list:
                new_selfies_candidate = self.current_selfies + action_name
                valid_selfies = True
                try:
                    sf.split_selfies(new_selfies_candidate)
                except:
                    valid_selfies = False
                
                if valid_selfies:
                    if not self.bond_counts:
                        self.current_selfies = new_selfies_candidate
                        self.bond_counts.append(0)
                    else:
                        bond_info = self._implied_bond_order(action_name)
                        new_order = bond_info[0] if bond_info else 1
                        self.bond_counts[-1] += new_order
                        self.bond_counts.append(new_order)
                        self.current_selfies = new_selfies_candidate
                else:
                    raw_reward = -0.1
            else:
                raw_reward = -0.1
            
            #TODO adding reward here

            # if self.current_mol:
            #     molecule_size = len(self.bond_counts)
            #     size_reward = 0.01 * molecule_size
            #     fake_spectrum = {
            #         'mz_values': np.array([0]),
            #         'intensities': np.array([0])
            #     }
            #     similarity_reward = self._cosine_similarity(
            #         fake_spectrum['mz_values'],
            #         fake_spectrum['intensities'],
            #         self.target_mzs,
            #         self.target_ints
            #     ) * 0.1
            #     raw_reward = size_reward + similarity_reward
            # else:
            #     raw_reward = -0.1
            
            self.episode_return += raw_reward
            self._final_eval_reward += raw_reward
            
            if self.episode_length >= self.max_episode_steps:
                done = True
        
        obs_dict = {
            'observation': self.target_spectrum['embeds'],
            'prefix': self.current_selfies,
            'action_mask': action_mask,
            'to_play': -1,
            'chance': self.chance
        }
        
        if self.reward_normalize:
            reward = raw_reward / self.reward_norm_scale
        else:
            reward = raw_reward
        
        info["raw_reward"] = raw_reward
        if self.current_selfies:
            info["current_selfies"] = self.current_selfies
        
        if done:
            info['eval_episode_return'] = self._final_eval_reward
            if self.render_mode == 'image_savefile_mode':
                self.save_render_output(
                    replay_name_suffix=self.replay_name_suffix,
                    replay_path=self.replay_path,
                    format=self.replay_format
                )
        
        if self.render_mode is not None:
            self.render(self.render_mode)
        
        reward = to_ndarray([float(raw_reward)], dtype=np.float32)
        
        return BaseEnvTimestep(obs_dict, reward, done, info)

    
    def _implied_bond_order(self, token):
        """
        Extract the bond order and element from a SELFIES token.
        
        Args:
            token (str): SELFIES token.
            
        Returns:
            tuple: (bond_order, element) or None if parsing fails.
        """
        match = re.match(r'^\[([-=#+]*)([A-Za-z]+)', token)
        if not match:
            return None
        prefix, element = match.groups()
        order = 1
        if '=' in prefix:
            order = 2
        elif '#' in prefix:
            order = 3
        return order, element
    
    def _bond_order(self, token):
        """
        Get the bond order from a SELFIES token.
        
        Args:
            token (str): SELFIES token.
            
        Returns:
            int: Bond order (1, 2, or 3).
        """
        result = self._implied_bond_order(token)
        if result is None:
            return 1  
        order, _ = result
        return order
    
    '''not use'''
    def _cosine_similarity(self, true_mzs, true_ints, pred_mzs, pred_ints):
        """
        Calculate the cosine similarity between two mass spectra.
        
        Args:
            true_mzs (np.ndarray): True m/z values.
            true_ints (np.ndarray): True intensities.
            pred_mzs (np.ndarray): Predicted m/z values.
            pred_ints (np.ndarray): Predicted intensities.
            
        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        mz_max = 1000
        mz_bin_res = 1
        n_bins = int(mz_max / mz_bin_res)

        true_spectrum = np.zeros(n_bins)
        pred_spectrum = np.zeros(n_bins)

        true_bin_indices = (true_mzs / mz_bin_res).astype(int)
        pred_bin_indices = (pred_mzs / mz_bin_res).astype(int)

        true_mask = (true_bin_indices >= 0) & (true_bin_indices < n_bins)
        pred_mask = (pred_bin_indices >= 0) & (pred_bin_indices < n_bins)

        np.add.at(true_spectrum, true_bin_indices[true_mask], true_ints[true_mask])
        np.add.at(pred_spectrum, pred_bin_indices[pred_mask], pred_ints[pred_mask])

        norm_true = np.linalg.norm(true_spectrum)
        norm_pred = np.linalg.norm(pred_spectrum)
        if norm_true == 0 or norm_pred == 0:
            return 0.0 

        similarity = np.dot(true_spectrum, pred_spectrum) / (norm_true * norm_pred)

        random_similarity = np.random.rand()
        return random_similarity
    
    def _group_to_selfies(self, group_token):
        """
        Convert an element group token to its SELFIES string representation.
        
        Args:
            group_token (str): Element group token.
            
        Returns:
            str: SELFIES string representation of the group.
        """
        # this method needs to be improved to handle more complex element groups
        # currently, only simple mappings are provided
        group_to_selfies_map = {
            "[CH3]": "[C][H][H][H]",  
            "[OH]": "[O][H]",         # hydroxyl
            "[NH2]": "[N][H][H]",     # amine
            "[C=O]": "[C][=O]",       # carbonyl
            "[COOH]": "[C][O][O][H]", # carboxyl
            "[NO2]": "[N][=O][=O]",   # nitro
            "[CF3]": "[C][F][F][F]",  # trifluoro
        }
        
        # if the mapping is not found, return the original token
        return group_to_selfies_map.get(group_token, group_token)
    
    def is_done(self):
        """
        Check if the episode is finished.
        
        Returns:
            bool: True if the episode should end, False otherwise.
        """
        return self.should_done or self.episode_length >= self.max_episode_steps
    
    def seed(self, seed=None, dynamic_seed=None, **kwargs):
        """
        Set the random seed for the environment.
        
        Args:
            seed (int, optional): Fixed seed value.
            dynamic_seed (int, optional): Dynamic seed value.
            
        Returns:
            list: The used seed value.
        """
        if seed is None and dynamic_seed is None:
            seed = 0
        elif dynamic_seed is not None:
            seed = dynamic_seed
        self.np_random, used_seed = seeding.np_random(seed)
        return [used_seed]
    
    def render(self, mode='human'):
        """
        Render the current state of the environment.
        
        Args:
            mode (str): Rendering mode ('text_mode', 'molecule_image_mode', or 'image_savefile_mode').
        """
        if mode == 'text_mode':
            s = 'current total reward: {}, '.format(self.episode_return)
            s += 'current SELFIES: {}\n'.format(self.current_selfies)
            if self.smiles:
                s += 'SMILES: {}\n'.format(self.smiles)
            else:
                s += 'SMILES: (invalid molecule)\n'
            # print(s)
        elif mode == 'molecule_image_mode' or mode == 'image_savefile_mode':
            if self.smiles:
                # use RDKit to render the molecule
                img = Draw.MolToImage(self.current_mol, size=(300, 300))
                
                # add text information
                img_with_text = Image.new('RGB', (400, 350), color=(255, 255, 255))
                img_with_text.paste(img, (50, 0))
                
                draw = ImageDraw.Draw(img_with_text)
                fnt_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
                fnt = ImageFont.truetype(fnt_path, 12)
                
                # add SMILES information
                smiles = self.smiles
                draw.text((10, 310), f"SMILES: {smiles[:40]}", font=fnt, fill=(0, 0, 0))
                if len(smiles) > 40:
                    draw.text((10, 330), f"{smiles[40:]}", font=fnt, fill=(0, 0, 0))
                
                # if in real-time mode, show the image
                if mode == 'molecule_image_mode':
                    plt.imshow(np.asarray(img_with_text))
                    plt.draw()
                    plt.pause(0.001)
                elif mode == 'image_savefile_mode':
                    # add the frame to the frames list, for saving the animation
                    self.frames.append(np.asarray(img_with_text))
            else:
                # if no valid molecule, create a blank image
                img = Image.new('RGB', (400, 350), color=(255, 255, 255))
                draw = ImageDraw.Draw(img)
                fnt_path = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
                fnt = ImageFont.truetype(fnt_path, 12)
                draw.text((10, 175), "无效分子", font=fnt, fill=(0, 0, 0))
                
                if mode == 'molecule_image_mode':
                    plt.imshow(np.asarray(img))
                    plt.draw()
                    plt.pause(0.001)
                elif mode == 'image_savefile_mode':
                    self.frames.append(np.asarray(img))
        return
    
    def save_render_output(self, replay_name_suffix='', replay_path=None, format='svg'):
        """
        Save the rendered output to a file.
        
        Args:
            replay_name_suffix (str): Suffix for the output filename.
            replay_path (str, optional): Path to save the output.
            format (str): Output format ('svg', 'gif', or 'mp4').
        """
        if replay_path is None:
            filename = f'molecule_{replay_name_suffix}.{format}'
        else:
            if not os.path.exists(replay_path):
                os.makedirs(replay_path)
            filename = os.path.join(replay_path, f'molecule_{replay_name_suffix}.{format}')
        
        if format == 'gif':
            imageio.mimsave(filename, self.frames, 'GIF', duration=0.5)
        elif format == 'mp4':
            imageio.mimsave(filename, self.frames, fps=2, codec='mpeg4')
        elif format == 'svg' and self.current_mol:
            from rdkit.Chem.Draw import rdMolDraw2D
            drawer = rdMolDraw2D.MolDraw2DSVG(400, 350)
            drawer.DrawMolecule(self.current_mol)
            drawer.FinishDrawing()
            with open(filename, 'w') as f:
                f.write(drawer.GetDrawingText())
        else:
            if self.frames:
                imageio.imwrite(filename, self.frames[-1])
        
        logging.info("save render output to {}".format(filename))
        self.frames = []
    
    # def random_action(self) -> np.ndarray:
    #     """return a random legal action"""
    #     legal = self.legal_actions
    #     random_action = np.random.choice(legal)
    #     if isinstance(random_action, np.ndarray):
    #         pass
    #     elif isinstance(random_action, int):
    #         random_action = to_ndarray([random_action], dtype=np.int64)
    #     return random_action
    
    
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.target_spectrum['embeds'].shape[0]
    
    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space
    
    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_range
    
    def to_play(self):
        """return the ID of the current player - for single-player environments, always return 0"""
        return 0
    
    def get_smiles(self):
        if self.smiles:
            return self.smiles
        return ""

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """create the collector environment configuration"""
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_collect = True
        return [cfg for _ in range(collector_env_num)]
    
    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """create the evaluator environment configuration"""
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # in the evaluation stage, we do not need to normalize the reward
        cfg.reward_normalize = False
        cfg.is_collect = False
        return [cfg for _ in range(evaluator_env_num)]
    
    def __repr__(self) -> str:
        return "LightZero MassSpec Env."

    