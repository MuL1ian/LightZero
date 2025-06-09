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
from zoo.masspecgym.envs.mass_tokenizers import SelfiesTokenizer
from zoo.masspecgym.envs.utils import (
    parse_formula_counts,
    extract_element_from_token,
    get_allowed_elements_from_formula,
    get_action_mask,
    update_atom_counts,
    validate_selfies_addition,
    remove_last_token_from_selfies,
    calculate_formula_completion_reward,
    check_formula_match,
    get_state_info
)

# Import global reward network
try:
    from lzero.model import global_reward_network
    GLOBAL_REWARD_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Global reward network not available: {e}")
    GLOBAL_REWARD_AVAILABLE = False
    global_reward_network = None
    get_reward_function = lambda: lambda *args, **kwargs: 0.0

# Import the proper dataset
try:
    # Add the DataLoader path to sys.path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataloader_path = os.path.join(current_dir, '../../../../../DataLoader')
    dataloader_path = os.path.abspath(dataloader_path)
    reward_model_path = os.path.join(current_dir, '../../../../../reward_model/src')
    reward_model_path = os.path.abspath(reward_model_path)
    root_path = os.path.abspath(os.path.join(current_dir, '../../../../..'))  # Add root path for reward_model imports
    
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    if dataloader_path not in sys.path:
        sys.path.insert(0, dataloader_path)
    if reward_model_path not in sys.path:
        sys.path.insert(0, reward_model_path)
    
    from encoder_dataset import FormulaEncoderDataset, create_formula_encoder_dataloader
    from encoder import SpectrumEncoder, load_encoder
    DATASET_AVAILABLE = True
    ENCODER_AVAILABLE = True
    # print("[INFO] FormulaEncoderDataset and SpectrumEncoder imported successfully")
except ImportError as e:
    print(f"[WARN] FormulaEncoderDataset or SpectrumEncoder not available: {e}")
    DATASET_AVAILABLE = False
    ENCODER_AVAILABLE = False

# Custom Chemical Formula Tokenizer (same as in muzero_transformer.py)
class ChemicalFormulaTokenizer:
    """
    A case-sensitive tokenizer specifically designed for chemical formulas.
    Preserves capitalization which is crucial for chemical elements.
    """
    def __init__(self, max_length=50):
        self.max_length = max_length
        
        # Common chemical elements (case-sensitive)
        self.elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        ]
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        
        # Build vocabulary: special tokens + elements + digits
        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.cls_token: 2,
            self.sep_token: 3,
        }
        
        # Add elements to vocabulary
        for i, element in enumerate(self.elements):
            self.vocab[element] = i + 4
            
        # Add digits 0-9
        for digit in '0123456789':
            self.vocab[digit] = len(self.vocab)
            
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
    
    def tokenize_formula(self, formula: str) -> List[str]:
        """
        Tokenize a chemical formula into elements and numbers.
        
        Args:
            formula (str): Chemical formula like "C6H12O6"
            
        Returns:
            List[str]: List of tokens preserving case
        """
        if not formula:
            return []
            
        tokens = []
        i = 0
        
        while i < len(formula):
            # Try to match two-letter element (e.g., Cl, Br, Ca)
            if i + 1 < len(formula) and formula[i:i+2] in self.elements:
                tokens.append(formula[i:i+2])
                i += 2
            # Try to match single-letter element (e.g., C, H, O)
            elif formula[i] in self.elements:
                tokens.append(formula[i])
                i += 1
            # Match digits
            elif formula[i].isdigit():
                # Collect consecutive digits
                num_start = i
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                # Add each digit separately for better tokenization
                for digit in formula[num_start:i]:
                    tokens.append(digit)
            else:
                # Skip unknown characters
                i += 1
                
        return tokens
    
    def encode(self, formula: str, add_special_tokens=True, max_length=None, 
               padding='max_length', truncation=True, return_tensors=None):
        """
        Encode a chemical formula to token IDs.
        
        Args:
            formula (str): Chemical formula
            add_special_tokens (bool): Whether to add [CLS] and [SEP]
            max_length (int): Maximum sequence length
            padding (str): Padding strategy
            truncation (bool): Whether to truncate
            return_tensors (str): Return format
            
        Returns:
            torch.Tensor or List[int]: Token IDs
        """
        if max_length is None:
            max_length = self.max_length
            
        # Tokenize the formula
        tokens = self.tokenize_formula(formula)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.unk_token_id)
        
        # Truncate if necessary
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            
        # Pad if necessary
        if padding == 'max_length':
            while len(token_ids) < max_length:
                token_ids.append(self.pad_token_id)
                
        # Return as tensor if requested
        if return_tensors == 'pt':
            return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs back to formula string.
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens (bool): Whether to skip special tokens
            
        Returns:
            str: Decoded formula string
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
            
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in [self.pad_token, self.cls_token, 
                                                   self.sep_token, self.unk_token]:
                    continue
                    
                tokens.append(token)
        
        # Join tokens to form formula
        return ''.join(tokens)

class MassGymDataset(Dataset):
    """
    Wrapper around FormulaEncoderDataset for MassGym environment.
    Provides pre-computed spectrum embeddings and proper data format.
    """
    def __init__(self, data_dir="../reward_model/diffms/data/msg", split="train", use_precomputed=True,
                 filter_len = 50, use_filter = False,
                 ):
        self.use_precomputed = use_precomputed
        self.split = split
        self.spectrum_encoder = None
        self.dataset = None
        
        # First try to load pre-computed embeddings
        if use_precomputed:
            base_path = os.path.join(os.path.dirname(__file__), '../../../../../DataLoader')
            if split == "train":
                embed_file = os.path.join(base_path, "Trainning_spectrum_embeds.pt")
            else:
                embed_file = os.path.join(base_path, "debug_spectrum_embeds.pt")
            
            try:
                self.data = torch.load(embed_file, weights_only=False)
                self.size = len(self.data['formulas'])
                if use_filter:
                    # filter the data by the length of the selfies string
                    # print(f"[INFO] Filtering data by SELFIES length <= {filter_len}")
                    
                    # Convert all SMILES to SELFIES and check their lengths
                    valid_indices = []
                    for i, smiles in enumerate(self.data['smiles']):
                        try:
                            selfies_str = sf.encoder(smiles)
                            if selfies_str and len(selfies_str) <= filter_len:
                                valid_indices.append(i)
                        except Exception as e:
                            # Skip invalid SMILES that can't be converted to SELFIES
                            continue
                    
                    # Filter all data fields based on valid indices
                    filtered_data = {}
                    for key, value in self.data.items():
                        if isinstance(value, list):
                            filtered_data[key] = [value[i] for i in valid_indices]
                        elif isinstance(value, torch.Tensor):
                            filtered_data[key] = value[valid_indices]
                        else:
                            filtered_data[key] = value
                    
                    self.data = filtered_data
                    self.size = len(self.data['formulas'])
                    # print(f"[INFO] Filtered dataset size: {self.size} samples (SELFIES length <= {filter_len})")
                
                # print(f"[INFO] Loaded {self.size} pre-computed samples from {embed_file}")
                return  # Successfully loaded pre-computed data
            except FileNotFoundError:
                print(f"[WARN] Pre-computed file {embed_file} not found, falling back to real dataset")
                self.use_precomputed = False
        
        # Load the real dataset using FormulaEncoderDataset - NO FALLBACK TO DUMMY DATA
        if not DATASET_AVAILABLE:
            raise ImportError("FormulaEncoderDataset not available and no pre-computed data found")
            
        try:
            # Use absolute path for data directory
            abs_data_dir = os.path.join(os.path.dirname(__file__), '../../../../../reward_model/diffms/data/msg')
            self.dataset = FormulaEncoderDataset(data_dir=abs_data_dir)
            self.size = len(self.dataset)
            # print(f"[INFO] Using real FormulaEncoderDataset with {self.size} samples")
            
            # Initialize spectrum encoder for on-the-fly encoding
            if not ENCODER_AVAILABLE:
                raise ImportError("SpectrumEncoder not available for on-the-fly encoding")
                
            try:
                self.spectrum_encoder = load_encoder(device_str="cpu")
                print("[INFO] Spectrum encoder loaded for on-the-fly encoding")
            except Exception as e:
                raise RuntimeError(f"Failed to load spectrum encoder: {e}")
                
        except Exception as e:
            print(f"[ERROR] Failed to load FormulaEncoderDataset: {e}")
            import traceback
            traceback.print_exc()
            raise

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.use_precomputed:
            # Get basic data from pre-computed embeddings
            embeds = self.data['embeds'][idx]
            formulas = self.data['formulas'][idx]
            smiles = self.data['smiles'][idx]
            
            # Convert SMILES to SELFIES
            try:
                selfies_str = sf.encoder(smiles)
            except:
                selfies_str = ""
            
            # For pre-computed data, we only have embeddings, so create minimal spectrum batch
            # Note: This is only for pre-computed data where we don't have access to raw spectrum
            spectrum_batch = {
                'spectrum_embeds': embeds.unsqueeze(0) if embeds.dim() == 1 else embeds
            }
            
            return {
                'embeds': embeds,
                'formulas': formulas,
                'smiles': smiles,
                'selfies_string': selfies_str,
                'spectrum_batch': spectrum_batch,
                'formula': formulas,  # Alias for compatibility
                'spectrum_embed': embeds  # For direct use in reward function
            }
        else:
            # Get data from real FormulaEncoderDataset
            try:
                item = self.dataset[idx]
                
                # Extract formula from the molecule object
                mol = self.dataset.mol_list[idx]
                formula = None
                if hasattr(mol, 'get_molform'):
                    formula = mol.get_molform() 
                elif hasattr(mol, 'mol_formula'):
                    formula = mol.mol_formula
                else:
                    # Fallback: try to compute formula from SMILES
                    try:
                        from rdkit import Chem
                        from rdkit.Chem import rdMolDescriptors
                        mol_obj = Chem.MolFromSmiles(item['smiles'])
                        if mol_obj:
                            formula = rdMolDescriptors.CalcMolFormula(mol_obj)
                    except:
                        formula = "Unknown"
                
                # Convert SMILES to SELFIES
                try:
                    selfies_str = sf.encoder(item['smiles'])
                except:
                    selfies_str = ""
                
                # Get real spectrum features - this is already properly formatted by the featurizer
                spectrum_features = item['spectrum']
                
                # The spectrum_features from featurize() need to be collated to get the proper format with num_peaks
                # We need to use the collate_fn to get the proper spectrum_batch format
                if self.spectrum_encoder is not None:
                    try:
                        # Use the featurizer's collate_fn to create proper spectrum_batch format
                        # The collate_fn expects a list of featurized spectra
                        spectrum_batch = self.dataset.spec_featurizer.collate_fn([spectrum_features])
                        
                        # Create a proper batch format for the encoder - it expects smiles, formulas, and spectrum_batch
                        batch_for_encoder = {
                            'spectrum_batch': spectrum_batch,
                            'smiles': [item['smiles']],  # List format as expected by encoder
                            'formulas': [formula]  # List format as expected by encoder
                        }
                        embeds, _, _ = self.spectrum_encoder.encode_batch(batch_for_encoder)
                        embeds = embeds[0] if embeds.dim() > 1 else embeds  # Get first item from batch
                        
                        # Add the computed embeddings to the spectrum_batch
                        spectrum_batch['spectrum_embeds'] = embeds.unsqueeze(0) if embeds.dim() == 1 else embeds
                        
                    except Exception as e:
                        print(f"[WARN] Error encoding spectrum: {e}")
                        # If encoding fails, raise the error instead of falling back to dummy data
                        raise RuntimeError(f"Failed to encode spectrum for item {idx}: {e}")
                else:
                    raise RuntimeError("Spectrum encoder is required but not available")
                
                return {
                    'embeds': embeds,
                    'formulas': formula,
                    'smiles': item['smiles'],
                    'selfies_string': selfies_str,
                    'spectrum_batch': spectrum_batch,
                    'formula': formula,  # Alias for compatibility
                    'spectrum_embed': embeds  # For direct use in reward function
                }
            except Exception as e:
                print(f"[ERROR] Error getting item {idx}: {e}")
                import traceback
                traceback.print_exc()
                # Re-raise the error instead of returning dummy data
                raise RuntimeError(f"Failed to get item {idx} from dataset: {e}")

    def random_sample(self):
        idx = random.randint(0, self.size - 1)
        return self.__getitem__(idx)




class DebugSpectrumDataset(Dataset):
    def __init__(self, file_path="../../DataLoader/debug_spectrum_embeds.pt"):
        self.data = torch.load(file_path, weights_only=False)

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
    Excludes hydrogen [H] tokens since hydrogens are implicit in SELFIES.
    
    Args:
        atoms (list): List of SELFIES tokens.
        
    Returns:
        list: Filtered list containing only valid chemical element tokens (excluding hydrogen).
    """
    valid_atoms = []
    for atom in atoms:
        if is_valid_token(atom):
            # Extract element from token to check if it's hydrogen
            element = extract_element(atom)
            # Exclude hydrogen tokens since hydrogens are implicit in SELFIES
            if element != 'H':
                valid_atoms.append(atom)
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

    default_branch = {"[Branch1]", "[Branch2]", "[Branch3]", "[#Branch1]", "[#Branch2]", "[#Branch3]", "[=Branch1]", "[=Branch2]", "[=Branch3]"}
    default_ring = {"[Ring1]", "[Ring2]", "[Ring3]", "[=Ring1]", "[=Ring2]", "[=Ring3]"}

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


selfies_tokenizer = SelfiesTokenizer(max_len=100)



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
        use_filter=True,
        filter_len=30,
        
        atom_tokens=[],
        pure_atom_tokens=[],
        bonded_atom_tokens=[],
        branch_tokens=[],
        ring_tokens=[],
        
        max_len=100,
        formula_masking=True,
        formula_max_len=50,  # Maximum length for formula tokens
        debug=True,
        
        # Intelligent END token masking to prevent early termination
        prevent_early_termination=True,      # Whether to prevent early END token selection
        min_formula_completion=0.8,          # Minimum completion ratio before END is allowed  
        allow_early_end_after_steps=20,      # Allow END after this many steps even if incomplete
        
        # Batched reward computation settings
        enable_batched_rewards=False,  # Enable batched reward computation (deprecated)
        batch_size=32,                 # Maximum batch size for reward computation
        batch_timeout=0.1,             # Timeout for batching (seconds)
        client_timeout=None,           # Timeout for client requests to reward server (seconds, auto-calculated if None)
        reward_network_checkpoint=None, # Path to reward network checkpoint
        use_reward_server=None,        # Whether to use reward server (auto-detect if None)
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

        self.debug = cfg.get('debug', True)
        
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
        
        # Initialize global reward network if available
        if GLOBAL_REWARD_AVAILABLE and global_reward_network is not None:
            try:
                # First, determine what type of reward computation to use based on configuration
                use_reward_server = cfg.get('use_reward_server', None)
                enable_batching = cfg.get('enable_batched_rewards', False)
                batch_size = cfg.get('batch_size', 32)
                batch_timeout = cfg.get('batch_timeout', 0.1)
                client_timeout = cfg.get('client_timeout', None)
                
                # Determine the reward computation strategy
                if use_reward_server is True:
                    pass
                    # print(f"[INFO] Configured to use reward server (batch_size={batch_size}, timeout={batch_timeout}s)")
                elif use_reward_server is None:
                    # Auto-detect: use reward server for subprocess environments
                    import multiprocessing as mp
                    current_process = mp.current_process()
                    # if current_process.name != 'MainProcess':
                    #     print(f"[INFO] Auto-detected subprocess environment, will use reward server")
                    # else:
                    #     print(f"[INFO] Auto-detected main process, will start reward server if needed")
                elif enable_batching:
                    print(f"[INFO] Configured to use batched rewards (batch_size={batch_size}, timeout={batch_timeout}s)")
                else:
                    print("[INFO] Configured to use individual reward computation")
                
                # Initialize the global reward network with the determined strategy
                # Check if reward server is already enabled before trying to initialize
                try:
                    server_already_enabled = global_reward_network.is_reward_server_enabled()
                except Exception as e:
                    print(f"[WARN] Error checking reward server status: {e}")
                    server_already_enabled = False
                
                if not server_already_enabled:
                    global_reward_network.initialize_global_reward_network(
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        checkpoint_path=cfg.get('reward_network_checkpoint', None),
                        enable_batching=enable_batching,
                        batch_size=batch_size,
                        batch_timeout=batch_timeout,
                        use_reward_server=use_reward_server,
                        client_timeout=client_timeout
                    )
                
                self.reward_function = global_reward_network.get_reward_function()
                self.use_reward_network = True
                
                # Report the actual reward computation method being used
                try:
                    if global_reward_network.is_reward_server_enabled():
                        pass
                        # print(f"[INFO] ✓ Using reward server (batch_size={batch_size}, timeout={batch_timeout}s)")
                    elif global_reward_network.is_batching_enabled():
                        print(f"[INFO] ✓ Using batched reward network (batch_size={batch_size}, timeout={batch_timeout}s)")
                    else:
                        print("[INFO] ✓ Using individual reward network computation")
                except Exception as e:
                    print(f"[WARN] Error checking reward computation method: {e}")
                    print("[INFO] ✓ Using reward network (method unknown)")
                    
            except Exception as e:
                print(f"[WARN] Failed to initialize global reward network: {e}")
                self.use_reward_network = False
                self.reward_function = None
        else:
            self.use_reward_network = False
            self.reward_function = None
        
        self.max_episode_steps = cfg.get('max_episode_steps', 100)
        self.is_collect = cfg.get('is_collect', True)
        self.ignore_legal_actions = cfg.get('ignore_legal_actions', False)
        self.need_flatten = cfg.get('need_flatten', False)

        self.formula_masking = cfg.get('formula_masking', True)
        
        # Intelligent END token masking configuration
        self.prevent_early_termination = cfg.get('prevent_early_termination', True)
        self.min_formula_completion = cfg.get('min_formula_completion', 0.8)
        self.allow_early_end_after_steps = cfg.get('allow_early_end_after_steps', 20)
        
        self.chance = 0.0
        
        self.frames = []

        self.target_embeds = cfg.get('target_spectrum', {}).get('embeds', [])
        self.target_formula = cfg.get('target_spectrum', {}).get('formulas', '')
        
        # initialize the tokenizer
        self.tokenizer = selfies_tokenizer
        self.formula_tokenizer = ChemicalFormulaTokenizer(max_length=50)
        
        # Formula tokenization parameters
        self.formula_max_len = cfg.get('formula_max_len', 50)
        
        # get the semantic robust alphabet
        all_atom_tokens = sf.get_semantic_robust_alphabet()
        
        # split all tokens
        self.atom_tokens, self.bonded_atom_tokens, self.branch_tokens, self.ring_tokens = split_atoms(all_atom_tokens)
        
        # filter out valid atoms
        self.atom_tokens = filter_valid_atoms(self.atom_tokens)
        
        # control tokens
        self.remove_token = "<REMOVE>"
        self.end_token = "<END>"

                # Record tokenizer special tokens
        self.tokenizer_special_tokens = [
                self.tokenizer.pad_token,
                self.tokenizer.sos_token,
                self.tokenizer.eos_token,
                self.tokenizer.unk_token
        ]
        
        # Include ALL tokens (including special tokens and hydrogen) to match vocabulary size
        # This ensures action space size = vocabulary size = 75
        # Add [H] explicitly since it may be filtered out but is in the vocabulary
        additional_tokens = []
        if '[H]' not in self.atom_tokens and '[H]' in self.tokenizer.get_vocab():
            additional_tokens.append('[H]')
            
        self.actions_list = (self.atom_tokens + 
                            self.bonded_atom_tokens +
                            self.branch_tokens + 
                            self.ring_tokens + 
                            [self.end_token] +
                            [self.remove_token] +
                            self.tokenizer_special_tokens +
                            additional_tokens
                            )
        
        vocab = self.tokenizer.get_vocab()
        for action in self.actions_list:
            if action not in vocab:
                raise ValueError(f"Action {action} not in tokenizer vocabulary!")

        # build the mapping between action and token id
        action_token_ids = [self.tokenizer.token_to_id(tok) for tok in self.actions_list]
        self.action_index2token_id = {
            i: tid for i, tid in enumerate(action_token_ids)
        }
        self.token_id2action_index = {
            tid: i for i, tid in enumerate(action_token_ids)
        }

        self.max_len = cfg.get('max_len', 100)
        
        
        self.episode_return = 0
        self.episode_length = 0
        self.should_done = False
        self._timestep = 0

        # set the action space and observation space for the gym interface
        self._action_space = spaces.Discrete(len(self.actions_list))
        
        self._reward_range = (0., 1.)
        
        # initialize the state
        self.current_selfies = ""
        self.bond_counts = []
        self.smiles = ""
        self.gt_selfies = ""

        self.bond_constraints = cfg.get('bond_constraints', get_bond_constraints())
        
        # Initialize dataset with proper integration
        if self.debug:
            # print("Using debug dataset")
            # print("================")
            self.train_info = MassGymDataset(split="debug", use_precomputed=True)
        else:
            # print("Using train dataset")
            # print("================")
            self.train_info = MassGymDataset(split="train", use_precomputed=True, 
                                             use_filter=cfg.get('use_filter', False),
                                             filter_len=cfg.get('filter_len', 50))

        self.reset()
        self._init_flag = True
        self.reward_obtained = False

    
    def random_massspecgym_data(self):

        sample = self.train_info.random_sample()
        embeds = sample['embeds']
        formula = sample['formulas']
        self.gt_selfies = sample['selfies_string']
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
        
        # Initialize atom tracking
        self.target_element_counts = parse_formula_counts(self.target_spectrum.get('formulas', ''))
        self.used_element_counts = {}
        
        self.token_ids = self._encode_selfies()
        self._timestep = 0

        self.reward_obtained = False

        action_mask = self.get_valid_actions().astype(np.int8)

        # TODO 应该不是在这里batch，先取消这里的batch（目前）
        # spectrum_obs = self.target_spectrum['embeds'].unsqueeze(0)  # [1, 4096]
        # token_obs = self.token_ids.float().unsqueeze(0)  # [1, 100]

        # combined_obs = torch.cat([spectrum_obs, token_obs], dim=1)
        
        spectrum = self.target_spectrum['embeds'] 
        formula = self.target_spectrum['formulas']
        
        # Encode formula using the helper method with fixed length
        formula_token_ids = self._encode_formula(formula).float()
        token = self.token_ids.float()  
        
        # Combine all observations: spectrum (4096) + selfies tokens (100) + formula tokens (50)
        combined_obs = torch.cat([spectrum, token, formula_token_ids], dim=-1)

        # Calculate expected observation dimension
        expected_dim = 4096 + self.max_len + self.formula_max_len  # 4096 + 100 + 50 = 4246
        
        obs_dict = {
            'observation': combined_obs,
            'action_mask': action_mask,  
            'to_play': -1,
            'chance': self.chance,
            'timestep': self._timestep
        }
        
        if self.render_mode is not None:
            self.render(self.render_mode)
            
        assert obs_dict['observation'].shape[-1] == expected_dim, f"The last dimension of the observation must be {expected_dim}, but got {combined_obs.shape[-1]}"
        return BaseEnvTimestep(obs_dict, to_ndarray([0.0], dtype=np.float32), False, {})


    def _encode_selfies(self):
        """
        Encode a SELFIES string into a list of token IDs.
        
        Returns:
            Tensor: The token IDs of the SELFIES string.
        """
        token_ids_list = self.tokenizer.encode_selfies(self.current_selfies)
        # ---- DEBUG START ----
        if not isinstance(token_ids_list, list) or len(token_ids_list) != self.max_len:
            print(f"DEBUG: MassGymEnv._encode_selfies: tokenizer returned problematic token_ids_list!")
            print(f"DEBUG: current_selfies = '{self.current_selfies}'")
            print(f"DEBUG: type(token_ids_list) = {type(token_ids_list)}")
            if isinstance(token_ids_list, list):
                print(f"DEBUG: len(token_ids_list) = {len(token_ids_list)}, but max_len = {self.max_len}")
                print(f"DEBUG: token_ids_list content (first 10): {token_ids_list[:10]}...")
            else:
                print(f"DEBUG: token_ids_list content: {token_ids_list}")
            # Consider raising an error here or returning a dummy tensor for further debugging
            # For example, to force a crash if this problematic case is hit:
            # raise ValueError("Tokenizer returned malformed token_ids_list")
        # ---- DEBUG END ----
        return torch.tensor(token_ids_list, dtype=torch.long)
        
        
    def _encode_formula(self, formula):
        """
        Encode a chemical formula using the custom chemical formula tokenizer with fixed length.
        
        Args:
            formula (str): Chemical formula string.
            
        Returns:
            torch.Tensor: Fixed-length tensor of formula token IDs.
        """
        # Encode the formula using the custom chemical formula tokenizer
        formula_token_ids = self.formula_tokenizer.encode(
            formula,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.formula_max_len,
            padding='max_length',  # Pad to max_length
            truncation=True,  # Truncate if longer than max_length
            return_tensors='pt'  # Return PyTorch tensors
        )
        
        # Remove batch dimension and return as 1D tensor
        return formula_token_ids.squeeze(0)

    def _get_allowed_elements_from_formula(self):
        """
        Extract allowed elements from a formula and return a set of all variants
        
        Returns:
            set: The set of allowed element symbols, including all variants (with charges and different bond types)
        """
        formula = self.target_spectrum.get('formulas', '')
        return get_allowed_elements_from_formula(formula)

    def get_valid_actions(self): 
        """
        Generate a boolean mask over the full action space indicating which actions are valid.
        This does not change the size of the action space (always same as len(self.actions_list)).
        """
        formula = self.target_spectrum.get('formulas', '') if hasattr(self, 'target_spectrum') else ''
        used_counts = getattr(self, 'used_element_counts', {})
        
        # Apply intelligent END token masking if enabled
        if self.prevent_early_termination:
            return get_action_mask(
                formula=formula,
                used_element_counts=used_counts,
                actions_list=self.actions_list,
                atom_tokens=self.atom_tokens,
                bonded_atom_tokens=self.bonded_atom_tokens,
                current_selfies=self.current_selfies,
                formula_masking=self.formula_masking,
                end_token=self.end_token,
                remove_token=self.remove_token,
                special_tokens=self.tokenizer_special_tokens,
                min_formula_completion=self.min_formula_completion,
                allow_early_end_after_steps=self.allow_early_end_after_steps
            )
        else:
            # Use original behavior for backward compatibility
            return get_action_mask(
                formula=formula,
                used_element_counts=used_counts,
                actions_list=self.actions_list,
                atom_tokens=self.atom_tokens,
                bonded_atom_tokens=self.bonded_atom_tokens,
                current_selfies=self.current_selfies,
                formula_masking=self.formula_masking,
                end_token=self.end_token,
                remove_token=self.remove_token,
                special_tokens=self.tokenizer_special_tokens
            )



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

        # for i in range(len(action_mask)):
        #     if action_mask[i]:
        #         print("debug: available action: ", self.actions_list[i])

        # Execute the action
        action_name = self.actions_list[action]
        raw_reward = 0.0
        info = {}
        done = False
        self._timestep += 1
        if self.episode_length >= self.max_episode_steps:
            print("debug: episode length >= max_episode_steps")
            done = True
            
        # Handle special tokens (should not be executed as actions)
        if action_name in self.tokenizer_special_tokens:
            # Special tokens are invalid actions during gameplay
            raw_reward = -0.5
            print(f"[WARN] Invalid special token action attempted: {action_name}")
        elif action_name == self.remove_token:
            if len(self.bond_counts) == 0:
                raw_reward = -0.5  # Can't remove from empty molecule
            else:
                # Remove the last token and update atom tracking
                if self.current_selfies:
                    # Use utility function to remove last token
                    updated_selfies, removed_token = remove_last_token_from_selfies(self.current_selfies)
                    
                    if removed_token:
                        # Update atom counts using utility function
                        self.used_element_counts = update_atom_counts(
                            removed_token, self.used_element_counts, increment=False
                        )
                        
                        self.current_selfies = updated_selfies
                        
                        # Update bond counts
                        if self.bond_counts:
                            self.bond_counts.pop()
                        
                        raw_reward = 0.0  # Neutral reward for successful removal
                    else:
                        raw_reward = -0.5
                else:
                    raw_reward = -0.5
        else:
            if action_name == self.end_token or self.episode_length >= self.max_episode_steps:
                # if self.episode_length >= self.max_episode_steps:
                    # print("debug: done by episode length >= max_episode_steps") 
                # else:
                    # print("debug: done by action_name == self.end_token")
                    # print(f"debug: current selfies: {self.current_selfies},\ntarget selfies: {sf.encoder(self.smiles)}") 
                done = True
                self.should_done = True
                # Use reward network for final reward computation
                if self.use_reward_network and self.reward_function:
                    try:
                        spectrum_embed = self.target_spectrum['embeds']
                        formula = self.target_spectrum['formulas']
                        similarity_score = self.reward_function(
                            self.current_selfies, 
                            spectrum_embed, 
                            formula
                        )
                        if not self.reward_obtained:
                            self.reward_obtained = True
                            raw_reward = similarity_score  # Use similarity as reward
                        else:
                            raw_reward = 0.0
                    except Exception as e:
                        print(f"[WARN] Error computing reward with network: {e}")
                        # Fallback to exact match
                        if self.current_selfies == sf.encoder(self.smiles):
                            raw_reward = 1.0
                        else:
                            raw_reward = -1.0
                else:
                    # Fallback to exact match
                    if self.current_selfies == sf.encoder(self.smiles):
                        raw_reward = 1.0
                    else:
                        raw_reward = -1.0

            # Handle atom addition
            elif action_name in self.actions_list:
                valid_selfies = validate_selfies_addition(self.current_selfies, action_name)
                
                if valid_selfies:
                    new_selfies_candidate = self.current_selfies + action_name
                    
                    # Track atom usage using utility function
                    self.used_element_counts = update_atom_counts(
                        action_name, self.used_element_counts, increment=True
                    )
                    
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
                    raw_reward = -0.5
            else:
                raw_reward = -0.5
            
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
            self.token_ids = self._encode_selfies()

            self.episode_return += raw_reward
            self._final_eval_reward += raw_reward
            

        

        # Construct observation consistently with reset method
        spectrum = self.target_spectrum['embeds'] 
        formula = self.target_spectrum['formulas']
        
        # Encode formula using the helper method with fixed length
        formula_token_ids = self._encode_formula(formula).float()
        token = self.token_ids.float()  
        # print(f"debug: current selfies: {self.current_selfies},\ntarget selfies: {sf.encoder(self.smiles)}") 

        # Combine all observations: spectrum (4096) + selfies tokens (100) + formula tokens (50)
        combined_obs = torch.cat([spectrum, token, formula_token_ids], dim=-1)

        # Calculate expected observation dimension
        expected_dim = 4096 + self.max_len + self.formula_max_len  # 4096 + 100 + 50 = 4246

        obs_dict = {
            'observation': combined_obs,
            'action_mask': action_mask,
            'to_play': -1,
            'chance': self.chance,
            'timestep': self._timestep
        }
        
        if self.reward_normalize:
            reward = raw_reward / self.reward_norm_scale
        else:
            reward = raw_reward
        
        info["raw_reward"] = raw_reward
        if self.current_selfies:
            info["current_selfies"] = self.current_selfies
        
        # Add GAG-specific information at every step for faster data collection
        # This allows the GAG collector to extract pairs without waiting for episode completion
        info['generated_selfies'] = self.current_selfies  # What the agent has produced so far
        info['target_selfies'] = self.gt_selfies  # Ground-truth target SELFIES
        info['ground_truth_selfies'] = self.gt_selfies  # Alternative key for ground-truth
        info['spectrum_embed'] = self.target_spectrum['embeds']  # Target spectrum embedding
        info['target_spectrum'] = self.target_spectrum['embeds']  # Alternative key for spectrum
        info['final_selfies'] = self.current_selfies  # Alternative key for generated
        info['agent_selfies'] = self.current_selfies  # Alternative key for generated
        info['episode_reward'] = self._final_eval_reward  # Current episode reward
        info['episode_length'] = self.episode_length  # Current episode length
        
        if done:
            info['eval_episode_return'] = self._final_eval_reward
            
            # Mark this as final episode data for GAG logging
            info['episode_complete'] = True
            
            if self.render_mode == 'image_savefile_mode':
                self.save_render_output(
                    replay_name_suffix=self.replay_name_suffix,
                    replay_path=self.replay_path,
                    format=self.replay_format
                )
        
        if self.render_mode is not None:
            self.render(self.render_mode)
        
        reward = to_ndarray([float(raw_reward)], dtype=np.float32)
        assert obs_dict['observation'].shape[-1] == expected_dim, f"The last dimension of the observation must be {expected_dim}, but got {combined_obs.shape[-1]}"
        
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



    