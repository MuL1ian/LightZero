import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from ding.utils import MODEL_REGISTRY, SequenceType
from lzero.model.common import MZNetworkOutput
from lzero.model.selfies_tokenizer import SelfiesTokenizer, pad_to_maxlen
from zoo.masspecgym.envs.massgymenv import MassGymEnv
import re

# Import global reward network
try:
    from lzero.model import global_reward_network
    GLOBAL_REWARD_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Global reward network not available in transformer: {e}")
    GLOBAL_REWARD_AVAILABLE = False
    global_reward_network = None
# -----------------------------------------------------------------------------
# Env for get the action list 
# -----------------------------------------------------------------------------
def get_actions_list():
    """Get the actions list from the environment"""
    try:
        # cfg not used, just for init 
        cfg = {
            'env_id': "mass_spec_env",
            'render_mode': None,
            'obs_type': 'fingerprint',
            'reward_normalize': False,
            'reward_norm_scale': 1.0,
            'reward_type': 'cosine_similarity',
            'target_spectrum': {
                'embeds': torch.tensor([]), 
                'formulas': ''  
            },
            'max_episode_steps': 100,
            'is_collect': True,
            'ignore_legal_actions': False,
            'need_flatten': False,
            'max_len': 100,
            'formula_masking': True,
        }

        env = MassGymEnv(cfg)
        return env.actions_list  # length 71
    except:
        # Fallback actions list for testing (excluding [H] since hydrogens are implicit in SELFIES)
        return ['[C]', '[O]', '[N]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]', 
                '[=C]', '[=N]', '[=O]', '[=S]', '[#C]', '[#N]', '[Ring1]', '[Ring2]', '[Ring3]',
                '[Branch1]', '[Branch2]', '[Branch3]', '<END>', '<REMOVE>'] + ['[UNK]'] * 47

# Get actions list (lazy loading)
actions_list = None

# Custom Chemical Formula Tokenizer
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

# -----------------------------------------------------------------------------
# Encoder-Decoder Transformer
# -----------------------------------------------------------------------------
class MassSelfiesED(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 128,
        d_model=512,
        n_enc=4,
        n_dec=6,
        n_head=8,
        dropout=0.1,
        n_spectrum_heads=32,
        device="cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.spectrum_dim = 4096
        # tokenizer for special ids and pad
        self.tokenizer = SelfiesTokenizer(max_len=max_len)
        # compute action_token_ids from global actions_list
        global actions_list
        if actions_list is None:
            actions_list = get_actions_list()
        self.action_token_ids = [self.tokenizer.token_to_id(tok) for tok in actions_list]

        # record special token ids
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_token_id = self.tokenizer.sos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        # Enhanced Encoder for spectrum with multi-head self-attention
        self.n_spectrum_heads = n_spectrum_heads 
        assert d_model % self.n_spectrum_heads == 0, f"d_model {d_model} must be divisible by n_spectrum_heads {self.n_spectrum_heads}"
        self.spectrum_head_dim = d_model // self.n_spectrum_heads
        
        # Project spectrum to multi-head format
        self.spec_proj = nn.Sequential(
            nn.Linear(self.spectrum_dim, d_model * self.n_spectrum_heads),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Spectrum self-attention layer
        self.spectrum_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        
        # Spectrum encoder layers
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        # Decoder
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head,
                                               dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)

        # Heads
        self.action_head = nn.Linear(d_model, vocab_size, bias=False)
        self.value_head  = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.to(self.device)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward_pretrain(
        self,
        spectrum_embed: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播用于预训练 - 返回完整序列的logits
        
        Args:
            spectrum_embed: (B, spectrum_dim) 质谱嵌入
            tgt_tokens: (B, T) 目标token序列
            tgt_mask: (B, T) attention mask
            
        Returns:
            torch.Tensor: (B, T, vocab_size) 每个位置的logits
        """
        spectrum_embed = spectrum_embed.to(self.device)
        tgt_tokens = tgt_tokens.long().to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        B, T = tgt_tokens.shape

        # Enhanced Encoder with multi-head spectrum processing
        # Project spectrum to multi-head format
        spec_projected = self.spec_proj(spectrum_embed)  # (B, d_model * n_spectrum_heads)
        
        # Reshape to multi-head format
        spec_multihead = spec_projected.view(B, self.n_spectrum_heads, -1)  # (B, n_spectrum_heads, d_model)
        
        # Apply self-attention across spectrum heads
        spec_attended, _ = self.spectrum_self_attn(
            spec_multihead, spec_multihead, spec_multihead
        )  # (B, n_spectrum_heads, d_model)
        
        # Use the attended spectrum as memory for decoder
        mem = self.encoder(spec_attended)  # (B, n_spectrum_heads, d_model)

        # Decoder
        pos_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
        dec_in = self.token_embed(tgt_tokens) + self.pos_embed(pos_ids)
        causal = self._generate_square_subsequent_mask(T).to(self.device)
        
        dec_out = self.decoder(
            tgt=dec_in, 
            memory=mem,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt_mask == 0)
        )  # (B, T, d_model)

        # 获取每个位置的logits
        logits = self.action_head(dec_out)  # (B, T, vocab_size)
        
        return logits

    def forward(
        self,
        combined_embed: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # split spectrum vs prefix
        spectrum_embed = combined_embed[:, :self.spectrum_dim].to(self.device)
        tgt_tokens = tgt_tokens.long().to(self.device)
        tgt_mask   = tgt_mask.to(self.device)
        B, T = tgt_tokens.shape

        # Enhanced Encoder with multi-head spectrum processing
        # Project spectrum to multi-head format
        spec_projected = self.spec_proj(spectrum_embed)  # (B, d_model * n_spectrum_heads)
        
        # Reshape to multi-head format
        spec_multihead = spec_projected.view(B, self.n_spectrum_heads, -1)  # (B, n_spectrum_heads, d_model)
        
        # Apply self-attention across spectrum heads
        spec_attended, _ = self.spectrum_self_attn(
            spec_multihead, spec_multihead, spec_multihead
        )  # (B, n_spectrum_heads, d_model)
        
        # Use the attended spectrum as memory for decoder
        mem = self.encoder(spec_attended)  # (B, n_spectrum_heads, d_model)

        # Decoder
        pos_ids  = torch.arange(T, device=self.device).unsqueeze(0)
        dec_in   = self.token_embed(tgt_tokens) + self.pos_embed(pos_ids)
        causal   = self._generate_square_subsequent_mask(T).to(self.device)
        dec_out  = self.decoder(
            tgt=dec_in, memory=mem,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt_mask==0)
        )
        last     = dec_out[:, -1, :]

        # full logits & value
        full_logits = self.action_head(last)  # (B, vocab_size)
        value       = self.value_head(last).squeeze(-1)

        # mask special tokens
        for sid in [self.pad_token_id, self.sos_token_id,
                    self.eos_token_id, self.unk_token_id]:
            full_logits[:, sid] = float('-1e9')

        # slice to environment action space (71)
        policy_logits = full_logits[:, self.action_token_ids]  # (B,71)
        return policy_logits, value

# -----------------------------------------------------------------------------
# Greedy step prediction helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def step_prediction(model: MassSelfiesED,
                   tokenizer: SelfiesTokenizer,
                   combined_vec: torch.Tensor,
                   device: Optional[str]=None):
    """Single-step greedy prediction - logit index = token_id"""
    model.eval()
    dev = device or next(model.parameters()).device
    vec = combined_vec.to(dev)
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)

    # extract prefix IDs from embedded vec
    prefix_ids = vec[:, model.spectrum_dim:].long()  # (B, prefix_len)

    # The prefix_ids already include SOS/EOS tokens and padding from the environment
    # So we can use them directly without adding another SOS token
    ids = prefix_ids.to(dev)  # (B, T)
    mask = (ids != tokenizer.pad_token_id).to(dev)  # (B, T)
    
    # forward - returns full vocabulary logits where index = token_id
    full_logits, value = model(vec, ids, mask)  # (B, vocab_size)
    
    # Mask out non-action tokens, keeping only valid action tokens
    global actions_list
    if actions_list is None:
        actions_list = get_actions_list()
    
    # Create action mask - only allow valid action tokens
    action_mask = torch.full_like(full_logits[0], float('-1e9'), device=dev)
    for action_token in actions_list:
        token_id = tokenizer.token_to_id(action_token)
        action_mask[token_id] = 0  # Allow this token
    
    # Apply action mask
    masked_logits = full_logits + action_mask.unsqueeze(0)
    
    # Get the token ID with highest probability
    next_token_id = torch.argmax(masked_logits, dim=-1)[0].item()
    
    return {
        'logits': masked_logits.squeeze(0),  # Full vocabulary logits with action masking
        'value':  value, 
        'probs':  F.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy(),
        'current_prefix': ids[0].tolist(),
        'next_token_id': next_token_id,  # The token ID (which equals the logit index)
    }

# -----------------------------------------------------------------------------
# MuZero transformer wrapper
# -----------------------------------------------------------------------------
@MODEL_REGISTRY.register('MuZeroSelfiesTransformer', force_overwrite=True)
class MuZeroSelfiesTransformer(nn.Module):
    def __init__(self, observation_shape=4246, max_len=100,
                 d_model=512, n_enc=4, n_dec=6, n_head=8,
                 dropout=0.1, device='cuda', **kwargs):
        super().__init__()
        # tokenizer and transformer
        self.spectrum_dim = 4096
        self.tok = SelfiesTokenizer(max_len=max_len)
        vocab_size = len(self.tok.get_vocab())
        self.transformer = MassSelfiesED(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=d_model,
            n_enc=n_enc,
            n_dec=n_dec,
            n_head=n_head,
            dropout=dropout,
            device=device
        )
        self.device = torch.device(device)
        self.to(self.device)
        self.cached_spectrum = None

    def initial_inference(self, obs: torch.Tensor):
        vec = obs if obs.dim()==2 else obs.unsqueeze(0)
        self.cached_spectrum = vec.to(self.device)
        B = vec.size(0)

        # For base class, extract only spectrum and SELFIES parts for transformer
        # Assume observation is spectrum (4096) + SELFIES (max_len) + potentially other data
        spectrum = vec[:, :self.spectrum_dim]
        selfies_part = vec[:, self.spectrum_dim:self.spectrum_dim + self.tok.max_length]
        
        # Clamp SELFIES token IDs to valid vocabulary range to handle random data
        vocab_size = len(self.tok.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1).float()
        
        combined_for_transformer = torch.cat([spectrum, selfies_part_clamped], dim=-1)
        
        pred = step_prediction(self.transformer, self.tok, combined_for_transformer, device=self.device)
        val = pred['value'].unsqueeze(-1).expand(B, 1)
        pol = pred['logits']  # Full vocabulary logits with action masking
        rew = [0.0] * B

        return MZNetworkOutput(value=val, reward=rew, policy_logits=pol, latent_state=obs)

    def _representation(self, observation: torch.Tensor) -> torch.Tensor:
        """Simply return the prefix as the latent state representation"""
        return observation

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update latent state by replacing last padding token with action (action = token_id)"""
        action = action.squeeze().float()

        # Ensure latent_state has batch dimension
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        
        # clone for help gradient computation
        next_latent_state = latent_state.clone()

        # Only look for padding tokens in the SELFIES part (after spectrum)
        # For base class, extract only the SELFIES portion (max_len tokens after spectrum)
        selfies_part = next_latent_state[:, self.spectrum_dim:self.spectrum_dim + self.tok.max_length]
        padding_mask = selfies_part == self.tok.pad_token_id
        last_padding_token_index = torch.sum(padding_mask, dim=1) - 1
        
        # Update the latent state at the correct global position
        batch_indices = torch.arange(next_latent_state.size(0))
        global_indices = self.spectrum_dim + last_padding_token_index
        next_latent_state[batch_indices, global_indices] = action

        reward = torch.zeros(latent_state.size(0), 1, device=latent_state.device)

        return next_latent_state, reward

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor):
        """Perform recurrent inference step"""
        next_latent_state, reward = self._dynamics(latent_state, action)
        # action is a tensor containing token IDs directly
        # check if any of the actions is the end token
        end_token_id = self.tok.token_to_id("<END>") if hasattr(self.tok, 'token_to_id') else self.tok.end_token_id
        if (action == end_token_id).any():
            print("end token found in recurrent inference")

        # For base class, extract only spectrum and SELFIES parts for transformer
        spectrum = next_latent_state[:, :self.spectrum_dim]
        selfies_part = next_latent_state[:, self.spectrum_dim:self.spectrum_dim + self.tok.max_length]
        
        # Clamp SELFIES token IDs to valid vocabulary range
        vocab_size = len(self.tok.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1)
        
        mask = (selfies_part_clamped != self.tok.pad_token_id) & (selfies_part_clamped != self.tok.end_token_id)

        # Get full vocabulary logits
        full_logits, value = self.transformer(spectrum, selfies_part_clamped, mask)
        
        # Apply action masking - mask out non-action tokens
        global actions_list
        if actions_list is None:
            actions_list = get_actions_list()
        
        # Create action mask
        action_mask = torch.full_like(full_logits, float('-1e9'), device=self.device)
        for action_token in actions_list:
            token_id = self.tok.token_to_id(action_token)
            action_mask[:, token_id] = 0  # Allow this token
        
        # Apply mask
        policy_logits = full_logits + action_mask
        value = value.unsqueeze(-1)

        return MZNetworkOutput(value=value, reward=reward, policy_logits=policy_logits, latent_state=next_latent_state)

    def _pad_batch_prefix(self, prefix_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = prefix_ids.shape
        pad_len = self.tok.max_length - L
        if pad_len < 0:
            raise ValueError(f"prefix length {L} exceeds max_len {self.tok.max_length}")
        pad = prefix_ids.new_full((B, pad_len), self.tok.pad_token_id)
        inp = torch.cat([prefix_ids, pad], dim=1)
        msk = torch.cat([torch.ones(B, L, dtype=torch.bool, device=prefix_ids.device),
                         torch.zeros(B, pad_len, dtype=torch.bool, device=prefix_ids.device)], dim=1)
        return inp, msk

# -----------------------------------------------------------------------------
# Enhanced MuZero transformer with formula masking and proper end detection
# -----------------------------------------------------------------------------
@MODEL_REGISTRY.register('MuZeroSelfiesTransformerEnhanced', force_overwrite=True)
class MuZeroSelfiesTransformerEnhanced(MuZeroSelfiesTransformer):
    def __init__(self, observation_shape=4246, max_len=100,
                 d_model=512, n_enc=4, n_dec=6, n_head=8,
                 dropout=0.1, device='cuda', target_formula=None,  # Deprecated: formula extracted from observations
                 formula_max_len=50, pretrained_transformer_path=None, 
                 # Intelligent END token masking parameters
                 prevent_early_termination=True,
                 min_formula_completion=0.8,
                 allow_early_end_after_steps=20,
                 **kwargs):
        super().__init__(observation_shape, max_len, d_model, n_enc, n_dec, 
                        n_head, dropout, device, **kwargs)
        
        # Load pretrained transformer weights if path is provided
        if pretrained_transformer_path is not None:
            self._load_pretrained_transformer(pretrained_transformer_path)
        
        # Note: target_formula is now extracted dynamically from observations
        
        # Update dimensions for new observation structure
        # Observation: spectrum (4096) + selfies tokens (max_len) + formula tokens (formula_max_len)
        self.formula_max_len = formula_max_len
        self.selfies_start_idx = self.spectrum_dim  # 4096
        self.formula_start_idx = self.spectrum_dim + max_len  # 4096 + max_len
        # print("debug: max_len: ", max_len)
        # print("debug: formula_max_len: ", formula_max_len)
        # print("debug: observation_shape: ", observation_shape)
        assert self.formula_start_idx + self.formula_max_len == observation_shape, \
            f"formula_start_idx + formula_max_len != observation_shape, {self.formula_start_idx} + {self.formula_max_len} != {observation_shape}"
        
        # Use custom chemical formula tokenizer instead of BERT
        self.formula_tokenizer = ChemicalFormulaTokenizer(max_length=self.formula_max_len)
        
        # Import the utility functions for action masking
        try:
            import sys
            import os
            # Try absolute import first
            from zoo.masspecgym.envs.utils import get_action_mask_from_selfies_string
            self.get_action_mask_from_selfies_string = get_action_mask_from_selfies_string
        except ImportError:
            try:
                # Try relative path import
                utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../zoo/masspecgym/envs'))
                if utils_path not in sys.path:
                    sys.path.insert(0, utils_path)
                import utils as massgym_utils
                self.get_action_mask_from_selfies_string = massgym_utils.get_action_mask_from_selfies_string
            except ImportError as e:
                print(f"[WARN] Could not import action masking utilities: {e}")
                # Fallback: disable formula masking
                self.get_action_mask_from_selfies_string = None
        
        # Get action lists from environment for masking
        global actions_list
        if actions_list is None:
            actions_list = get_actions_list()
        self.atom_tokens = [token for token in actions_list if token.startswith('[') and 
                           not any(special in token for special in ['Ring', 'Branch'])]
        self.bonded_atom_tokens = []  # Can be extended if needed
        
        # Store configuration
        self.formula_max_len = formula_max_len
        self.pretrained_transformer_path = pretrained_transformer_path
        
        # Intelligent END token masking configuration
        self.prevent_early_termination = prevent_early_termination
        self.min_formula_completion = min_formula_completion
        self.allow_early_end_after_steps = allow_early_end_after_steps
        
    def _load_pretrained_transformer(self, pretrained_path: str):
        """
        Load pretrained weights into the self.transformer (MassSelfiesED) component.
        
        Args:
            pretrained_path (str): Path to the pretrained model checkpoint
        """
        try:
            print(f"[INFO] Loading pretrained transformer from: {pretrained_path}")
            
            # Handle module compatibility for checkpoints saved with different module paths
            import sys
            
            # Create module alias for compatibility with checkpoints that reference 'pretrain_transformer'
            if 'pretrain_transformer' not in sys.modules:
                try:
                    from lzero.model import pretrain_transformer
                    sys.modules['pretrain_transformer'] = pretrain_transformer
                except ImportError:
                    # If the import fails, create a minimal compatibility module
                    import types
                    compat_module = types.ModuleType('pretrain_transformer')
                    from lzero.model.pretrain_transformer import PretrainConfig
                    compat_module.PretrainConfig = PretrainConfig
                    sys.modules['pretrain_transformer'] = compat_module
            
            # Load the checkpoint with weights_only=False for PyTorch 2.6 compatibility
            # This is safe since we trust the source of our pretrained models
            try:
                checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions that don't have weights_only parameter
                checkpoint = torch.load(pretrained_path, map_location=self.device)
            
            # Extract model state dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("[INFO] Found 'model_state_dict' in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("[INFO] Found 'state_dict' in checkpoint")
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
                print("[INFO] Using checkpoint as state dict directly")
            
            # Handle shape mismatches, particularly for positional embeddings
            current_state = self.transformer.state_dict()
            adjusted_state_dict = {}
            
            for key, value in state_dict.items():
                if key in current_state:
                    current_shape = current_state[key].shape
                    checkpoint_shape = value.shape
                    
                    if current_shape != checkpoint_shape:
                        print(f"[INFO] Shape mismatch for {key}: checkpoint {checkpoint_shape} vs current {current_shape}")
                        
                        if key == 'pos_embed.weight' and len(current_shape) == 2 and len(checkpoint_shape) == 2:
                            # Handle positional embedding size mismatch
                            max_len_current = current_shape[0]
                            max_len_checkpoint = checkpoint_shape[0]
                            
                            if max_len_checkpoint > max_len_current:
                                # Truncate if checkpoint has larger max_len
                                adjusted_value = value[:max_len_current, :]
                                print(f"[INFO] Truncated {key} from {checkpoint_shape} to {adjusted_value.shape}")
                            elif max_len_checkpoint < max_len_current:
                                # Pad with zeros if checkpoint has smaller max_len
                                padding_size = max_len_current - max_len_checkpoint
                                padding = torch.zeros(padding_size, value.shape[1], device=value.device, dtype=value.dtype)
                                adjusted_value = torch.cat([value, padding], dim=0)
                                print(f"[INFO] Padded {key} from {checkpoint_shape} to {adjusted_value.shape}")
                            else:
                                adjusted_value = value
                                
                            adjusted_state_dict[key] = adjusted_value
                        else:
                            # For other mismatches, skip loading this parameter
                            print(f"[WARN] Skipping {key} due to incompatible shape mismatch")
                            continue
                    else:
                        adjusted_state_dict[key] = value
                else:
                    # Key not in current model, skip
                    print(f"[WARN] Key {key} not found in current model, skipping")
            
            # Load the adjusted state dict into the transformer
            missing_keys, unexpected_keys = self.transformer.load_state_dict(adjusted_state_dict, strict=False)
            
            if missing_keys:
                print(f"[WARN] Missing keys when loading pretrained transformer: {missing_keys}")
            if unexpected_keys:
                print(f"[WARN] Unexpected keys when loading pretrained transformer: {unexpected_keys}")
            
            print(f"[INFO] Successfully loaded pretrained transformer weights!")
            print(f"[INFO] Transformer parameters: {sum(p.numel() for p in self.transformer.parameters()):,}")
            print(f"[INFO] Trainable parameters: {sum(p.numel() for p in self.transformer.parameters() if p.requires_grad):,}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load pretrained transformer from {pretrained_path}: {e}")
            import traceback
            traceback.print_exc()
            print("[WARN] Continuing with randomly initialized transformer weights")
            
    def set_target_formula(self, formula: str):
        """Deprecated: target formula is now extracted dynamically from observations"""
        print(f"[WARN] set_target_formula is deprecated. Formula is now extracted from observations automatically.")
        
    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update latent state by replacing last padding token with action in SELFIES part only"""
        action = action.squeeze().float()

        # Ensure latent_state has batch dimension
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        
        # Clone for gradient computation
        next_latent_state = latent_state.clone()

        # Extract SELFIES part only for modification
        selfies_part = next_latent_state[:, self.selfies_start_idx:self.formula_start_idx]
        
        # Find padding tokens only in SELFIES part
        padding_mask = selfies_part == self.tok.pad_token_id
        last_padding_token_index = torch.sum(padding_mask, dim=1) - 1
        
        # Update only the SELFIES part
        batch_indices = torch.arange(next_latent_state.size(0))
        global_indices = self.selfies_start_idx + last_padding_token_index
        next_latent_state[batch_indices, global_indices] = action

        reward = torch.zeros(latent_state.size(0), 1, device=latent_state.device)

        return next_latent_state, reward

    def _extract_selfies_from_latent_state(self, latent_state: torch.Tensor) -> List[str]:
        """Extract SELFIES strings from latent state tensor"""
        # Ensure latent_state has batch dimension
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
            
        # Extract SELFIES token IDs (excluding spectrum and formula parts)
        selfies_ids = latent_state[:, self.selfies_start_idx:self.formula_start_idx].long()
        
        selfies_strings = []
        for batch_idx in range(selfies_ids.size(0)):
            # Get non-padding tokens
            token_ids = selfies_ids[batch_idx]
            valid_mask = (token_ids != self.tok.pad_token_id) & (token_ids != self.tok.sos_token_id)
            valid_ids = token_ids[valid_mask].tolist()
            
            # Convert token IDs back to SELFIES string
            try:
                selfies_str = self.tok.decode_to_selfies(valid_ids, skip_special_tokens=True)
                selfies_strings.append(selfies_str)
            except:
                selfies_strings.append("")
                
        return selfies_strings
    
    def _extract_formula_from_latent_state(self, latent_state: torch.Tensor) -> List[str]:
        """Extract formula strings from latent state tensor"""
        # Ensure latent_state has batch dimension
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
            
        # Extract formula token IDs
        # print(f"debug: spectrum_dim: {self.spectrum_dim}, max_len: {self.max_len}, formula_max_len: {self.formula_max_len}")
        # print('debug: formula_start_idx: ', self.formula_start_idx)
        # print("debug: latent_state shape: ", latent_state.shape)
        formula_ids = latent_state[:, self.formula_start_idx:].long()
        # print("debug: formula_ids shape: ", formula_ids.shape)
        
        # Import BERT tokenizer for decoding
        formula_strings = []
        for batch_idx in range(formula_ids.size(0)):
            token_ids = formula_ids[batch_idx].tolist()
            # Decode formula tokens
            # try:
            formula_str = self.formula_tokenizer.decode(token_ids, skip_special_tokens=True)
            
            formula_strings.append(formula_str.strip())
            # except:
            #     formula_strings.append("")
                
        return formula_strings
    
    def _apply_formula_mask(self, full_logits: torch.Tensor, current_selfies_list: List[str], formula_list: List[str]) -> torch.Tensor:
        """Apply formula-based action masking to full vocabulary logits at token_id positions"""
        # Check for NaN in input logits first
        if torch.isnan(full_logits).any():
            print(f"[WARN] NaN detected in input full_logits before masking")
            full_logits = torch.where(torch.isnan(full_logits), torch.tensor(0.0, device=full_logits.device), full_logits)
        
        # Start with basic action masking - mask out non-action tokens
        global actions_list
        if actions_list is None:
            actions_list = get_actions_list()
        
        # Create base action mask - allow only valid action tokens
        action_mask = torch.full_like(full_logits, float('-1e9'), device=full_logits.device)
        for action_token in actions_list:
            token_id = self.tok.token_to_id(action_token)
            action_mask[:, token_id] = 0  # Allow this token
        
        # Apply base action mask
        masked_logits = full_logits + action_mask
        
        # Apply formula-based masking if available
        if not formula_list or not any(formula_list):
            return masked_logits
            
        batch_size = full_logits.size(0)
        
        for batch_idx in range(batch_size):
            current_selfies = current_selfies_list[batch_idx]
            target_formula = formula_list[batch_idx] if batch_idx < len(formula_list) else ""
            
            if not target_formula:
                continue
                
            try:
                # Get action mask using the utility function
                formula_action_mask = self.get_action_mask_from_selfies_string(
                    formula=target_formula,
                    current_selfies=current_selfies,
                    actions_list=actions_list,
                    atom_tokens=self.atom_tokens,
                    bonded_atom_tokens=self.bonded_atom_tokens,
                    formula_masking=True,
                    end_token="<END>",
                    remove_token="<REMOVE>",
                    special_tokens=[],
                    min_formula_completion=self.min_formula_completion if self.prevent_early_termination else 0.0,
                    allow_early_end_after_steps=self.allow_early_end_after_steps if self.prevent_early_termination else 0
                )
                
                # Apply formula mask at token_id positions
                for action_idx, is_allowed in enumerate(formula_action_mask):
                    if not is_allowed:
                        action_token = actions_list[action_idx]
                        token_id = self.tok.token_to_id(action_token)
                        # Mask out this token by setting a very negative value
                        masked_logits[batch_idx, token_id] = float('-1e9')
                
            except Exception as e:
                print(f"[WARN] Failed to apply formula mask for batch {batch_idx}: {e}")
                # Continue without formula masking for this batch
                
        # Final check for NaN values
        if torch.isnan(masked_logits).any():
            print(f"[WARN] NaN detected in final masked_logits, replacing with safe values")
            masked_logits = torch.where(torch.isnan(masked_logits), torch.tensor(-1e9, device=full_logits.device), masked_logits)
                
        return masked_logits
    
    def _check_completion_status(self, current_selfies_list: List[str], formula_list: List[str]) -> List[bool]:
        """Check if molecules are complete based on available actions"""
        if not formula_list or not any(formula_list):
            return [False] * len(current_selfies_list)
            
        completion_status = []
        
        for idx, current_selfies in enumerate(current_selfies_list):
            target_formula = formula_list[idx] if idx < len(formula_list) else ""
            
            if not target_formula:
                completion_status.append(False)
                continue
                
            try:
                # Get action mask
                action_mask = self.get_action_mask_from_selfies_string(
                    formula=target_formula,
                    current_selfies=current_selfies,
                    actions_list=actions_list,
                    atom_tokens=self.atom_tokens,
                    bonded_atom_tokens=self.bonded_atom_tokens,
                    formula_masking=True,
                    end_token="<END>",
                    remove_token="<REMOVE>",
                    special_tokens=[],
                    min_formula_completion=self.min_formula_completion if self.prevent_early_termination else 0.0,
                    allow_early_end_after_steps=self.allow_early_end_after_steps if self.prevent_early_termination else 0
                )
                
                # Check if only END token is available
                end_token_idx = actions_list.index("<END>") if "<END>" in actions_list else -1
                
                if end_token_idx >= 0:
                    # Count available actions (excluding END token)
                    available_non_end_actions = sum(action_mask) - (1 if action_mask[end_token_idx] else 0)
                    is_complete = available_non_end_actions == 0 and action_mask[end_token_idx]
                else:
                    # If no END token, check if no actions are available
                    is_complete = sum(action_mask) == 0
                    
                completion_status.append(is_complete)
                
            except Exception as e:
                print(f"[WARN] Failed to check completion status: {e}")
                completion_status.append(False)
                
        return completion_status
    
    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor):
        """Enhanced recurrent inference with formula masking and proper end detection"""
        next_latent_state, reward = self._dynamics(latent_state, action)
        
        # Extract current SELFIES strings from latent state
        current_selfies_list = self._extract_selfies_from_latent_state(next_latent_state)
        
        # Extract formula strings for debugging/logging
        formula_list = self._extract_formula_from_latent_state(next_latent_state)
        
        # Check completion status
        completion_status = self._check_completion_status(current_selfies_list, formula_list)
        
        # Compute reward using global reward network only if action is end token
        if GLOBAL_REWARD_AVAILABLE and global_reward_network is not None:
            try:
                # Get the token ID of the end token
                end_token_id = self.tok.token_to_id("<END>") if hasattr(self.tok, 'token_to_id') else self.tok.end_token_id
                
                # Check if the current action is the end token (action contains token IDs directly)
                action_token_ids = action.squeeze().long()
                if action_token_ids.dim() == 0:  # Single action
                    action_token_ids = action_token_ids.unsqueeze(0)
                
                reward_function = global_reward_network.get_reward_function()
                batch_size = next_latent_state.size(0)
                reward_scores = []
                
                for batch_idx in range(batch_size):
                    current_token_id = action_token_ids[batch_idx].item() if batch_idx < len(action_token_ids) else action_token_ids[0].item()
                    
                    # Only use reward network if this is the end token action
                    if current_token_id == end_token_id:
                        current_selfies = current_selfies_list[batch_idx]
                        formula = formula_list[batch_idx] if batch_idx < len(formula_list) else ""
                        spectrum_embed = next_latent_state[batch_idx, :self.spectrum_dim]
                        
                        # Compute reward using the global reward network
                        similarity_score = reward_function(current_selfies, spectrum_embed, formula)
                        reward_scores.append(similarity_score)
                    else:
                        # For non-end actions, use zero reward
                        reward_scores.append(0.0)
                
                # Update reward tensor
                reward = torch.tensor(reward_scores, device=next_latent_state.device).unsqueeze(-1)
                
            except Exception as e:
                print(f"[WARN] Error computing reward with global network: {e}")
                # Keep the original zero reward
        
        # Print completion info if any molecule is complete
        if any(completion_status):
            complete_indices = [i for i, complete in enumerate(completion_status) if complete]
            print(f"Molecules complete at indices {complete_indices}")
            for idx in complete_indices:
                print(f"  Batch {idx}: SELFIES='{current_selfies_list[idx]}', Formula='{formula_list[idx]}', Reward={reward[idx].item():.4f}")
        
        # Get transformer output - only use spectrum and SELFIES parts for transformer
        spectrum = next_latent_state[:, :self.spectrum_dim]
        selfies_ids = next_latent_state[:, self.selfies_start_idx:self.formula_start_idx]
        
        # Clamp SELFIES token IDs to valid vocabulary range
        vocab_size = len(self.tok.get_vocab())
        selfies_ids_clamped = torch.clamp(selfies_ids.long(), 0, vocab_size - 1)
        
        mask = (selfies_ids_clamped != self.tok.pad_token_id) & (selfies_ids_clamped != self.tok.end_token_id)

        assert not torch.isnan(spectrum).any(), f"spectrum has nan: {spectrum}"
        assert not torch.isnan(selfies_ids_clamped).any(), f"selfies_ids_clamped has nan: {selfies_ids_clamped}"
        assert not torch.isnan(mask).any(), f"mask has nan: {mask}"
        
        # Get full vocabulary logits
        full_logits, value = self.transformer(spectrum, selfies_ids_clamped, mask)
        assert not torch.isnan(full_logits).any(), f"full_logits has nan: {full_logits}"

        # Apply formula-based action masking (includes basic action masking)
        masked_logits = self._apply_formula_mask(full_logits, current_selfies_list, formula_list)
        value = value.unsqueeze(-1)
        
        return MZNetworkOutput(
            value=value, 
            reward=reward, 
            policy_logits=masked_logits, 
            latent_state=next_latent_state
        )
    
    def initial_inference(self, obs: torch.Tensor):
        """Enhanced initial inference with formula masking"""
        vec = obs if obs.dim()==2 else obs.unsqueeze(0)
        self.cached_spectrum = vec.to(self.device)
        B = vec.size(0)

        # Extract formula from observation
        formula_list = self._extract_formula_from_latent_state(vec)

        # Get initial prediction - only use spectrum and SELFIES parts
        spectrum = vec[:, :self.spectrum_dim]
        selfies_part = vec[:, self.selfies_start_idx:self.formula_start_idx]
        
        # Clamp SELFIES token IDs to valid vocabulary range to handle random data
        vocab_size = len(self.tok.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1)
        
        # Create mask for transformer
        mask = (selfies_part_clamped != self.tok.pad_token_id) & (selfies_part_clamped != self.tok.end_token_id)
        
        # Use transformer directly for batch processing - get full vocabulary logits
        full_logits, value = self.transformer(spectrum, selfies_part_clamped, mask)
        
        # Extract initial SELFIES (should be empty)
        current_selfies_list = self._extract_selfies_from_latent_state(vec)
        
        # Apply formula masking to logits (includes basic action masking)
        masked_logits = self._apply_formula_mask(full_logits, current_selfies_list, formula_list)
        
        val = value.unsqueeze(-1)
        pol = masked_logits
        rew = [0.0] * B

        return MZNetworkOutput(value=val, reward=rew, policy_logits=pol, latent_state=obs)


if __name__ == "__main__":
    # quick sanity check
    data   = torch.randn(4246)  # Updated to new observation dimension
    model  = MuZeroSelfiesTransformer()
    out = model.initial_inference(data)
    print([t.shape for t in (out.value, out.reward, out.policy_logits, out.latent_state)])
    print(out)