import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from ding.utils import MODEL_REGISTRY, SequenceType
from lzero.model.common import MZNetworkOutput
from lzero.model.selfies_tokenizer import SelfiesTokenizer, pad_to_maxlen
from zoo.masspecgym.envs.massgymenv import MassGymEnv
import re
import numpy as np

# Import global reward network
try:
    from lzero.model import global_reward_network
    GLOBAL_REWARD_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Global reward network not available in transformer: {e}")
    GLOBAL_REWARD_AVAILABLE = False
    global_reward_network = None
# Remove actions_list complexity - now using tokenizer vocabulary directly

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
# Now is decoder only Transformer (06 12 2025)
# -----------------------------------------------------------------------------
class MassSelfiesED(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 120,
        n_dec=12, 
        dropout=0.1,
        spectrum_chunk_size=256, # each chunk size  
        spectrum_attention_heads=4, # attention heads for spectrum decomposer
        n_head=16, # heads for main transformer decoder
        d_model=512, # dimension of the model
        device="cuda",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.spectrum_dim = 4096
        self.spectrum_chunk_size = spectrum_chunk_size
        self.spectrum_attention_heads = spectrum_attention_heads  # 清晰命名
        self.n_head = n_head
        self.d_model = d_model

        self.num_chunks = self.spectrum_dim // self.spectrum_chunk_size # how many chunks 

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

        # Token embeddings for SELFIES
        self.token_embed = nn.Embedding(vocab_size, self.d_model)
        self.pos_embed   = nn.Embedding(max_len, self.d_model)
 


        # Multi-head self-attention for spectrum decomposition
        self.spectrum_decomposer = nn.MultiheadAttention(
            embed_dim=self.spectrum_chunk_size,  # 256
            num_heads=self.spectrum_attention_heads,  # 4
            dropout=dropout,
            batch_first=True
        )

        # Projection from chunk_size to d_model
        self.chunk_to_model_proj = nn.Linear(self.spectrum_chunk_size, self.d_model) 
        
        # Decoder Only
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,  
            nhead=self.n_head,
            dropout=dropout, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)

<<<<<<< HEAD
        # Heads - output full vocabulary size since action_index = token_id
        self.action_head = nn.Linear(d_model, vocab_size, bias=False)
        self.value_head  = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
=======
        # Outputs Heads
        self.action_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), 
            nn.Tanh(), 
            nn.Linear(self.d_model, 1)
        )
>>>>>>> 458e2b62532f853ada563c7fafe3a977da2d5c4c
        self.to(self.device)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _prepare_spectrum_memory(self, spectrum_embed: torch.Tensor) -> torch.Tensor:
        """
        Spectrum embedding is decomposed using direct chunking and self-attention is applied.
        
        Args:
            spectrum_embed: (B, spectrum_dim) spectrum embedding
            
        Returns:
            torch.Tensor: (B, spectrum_heads, d_model) spectrum memory
        """
        batch_size = spectrum_embed.size(0)
        
        spectrum_sequence = spectrum_embed.view(batch_size, self.num_chunks, self.spectrum_chunk_size)
        
        # Apply multi-head self-attention to spectrum chunks
        spectrum_attended, _ = self.spectrum_decomposer(
            query=spectrum_sequence,
            key=spectrum_sequence, 
            value=spectrum_sequence
        )  # [batch_size, num_chunks, chunk_size]
        
        # Project to model dimension
        spectrum_memory = self.chunk_to_model_proj(spectrum_attended) # [batch_size, num_chunks, d_model]

        return spectrum_memory

    def forward_pretrain(
        self,
        spectrum_embed: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: torch.Tensor,
        return_value: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: # return logits or (logits, values)
        """
        forward for logits and value (if return_value=True)
        
        Args:
            spectrum_embed: (B, spectrum_dim) fingerprint 4096
            tgt_tokens: (B, T) tgt selfies tokens
            tgt_mask: (B, T) attention mask
            
        Returns:
            torch.Tensor: (B, T, vocab_size) logits for each position
            or Tuple[torch.Tensor, torch.Tensor]: (B, T, vocab_size), (B, T, 1)
        """
        spectrum_embed = spectrum_embed.to(self.device)
        tgt_tokens = tgt_tokens.long().to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        B, T = tgt_tokens.shape

        spectrum_memory = self._prepare_spectrum_memory(spectrum_embed)  # (B, spectrum_heads, d_model)
        memory = spectrum_memory  # (B, spectrum_heads, d_model)

        # Decoder
        pos_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
        dec_in = self.token_embed(tgt_tokens) + self.pos_embed(pos_ids)
        causal = self._generate_square_subsequent_mask(T).to(self.device)
        
        dec_out = self.decoder(
            tgt=dec_in, 
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt_mask == 0).bool()
        )  # (B, T, embed_dim)

        # Action logits for each position
        logits = self.action_head(dec_out)  # (B, T, vocab_size)
        
        if not return_value:
            return logits
        
        # Value prediction for each position  
        values = self.value_head(dec_out)  # (B, T, 1)
        
        return logits, values

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

        spectrum_memory = self._prepare_spectrum_memory(spectrum_embed)  # (B, spectrum_heads, d_model)
        
        memory = spectrum_memory  # (B, spectrum_heads, d_model)

        # Decoder
        pos_ids  = torch.arange(T, device=self.device).unsqueeze(0)
        dec_in   = self.token_embed(tgt_tokens) + self.pos_embed(pos_ids)
        causal   = self._generate_square_subsequent_mask(T).to(self.device)
        dec_out  = self.decoder(
            tgt=dec_in, memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt_mask==0).bool()
        )
        last     = dec_out[:, -1, :]

        # full logits & value
        full_logits = self.action_head(last)  # (B, vocab_size)
        value       = self.value_head(last).squeeze(-1)

        # Apply basic masking: mask out special tokens except EOS (which can be used for termination)
        full_logits[:, self.pad_token_id] = float('-1e9')
        full_logits[:, self.sos_token_id] = float('-1e9') 
        full_logits[:, self.unk_token_id] = float('-1e9')
        # Note: EOS token is kept unmasked as it's used for episode termination

        # Return full vocabulary logits - action_index = token_id
        return full_logits, value

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
    
    # The model already applies basic masking (pad, sos, unk tokens)
    # Additional environment-specific masking will be handled by the environment
    masked_logits = full_logits
    
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
        # check if any of the actions is the EOS token
        eos_token_id = self.tok.eos_token_id
        if (action == eos_token_id).any():
            print("EOS token found in recurrent inference")

        # For base class, extract only spectrum and SELFIES parts for transformer
        spectrum = next_latent_state[:, :self.spectrum_dim]
        selfies_part = next_latent_state[:, self.spectrum_dim:self.spectrum_dim + self.tok.max_length]
        
        # Clamp SELFIES token IDs to valid vocabulary range
        vocab_size = len(self.tok.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1)
        
        mask = (selfies_part_clamped != self.tok.pad_token_id) & (selfies_part_clamped != self.tok.eos_token_id)

        # Get full vocabulary logits - action_index = token_id
        policy_logits, value = self.transformer(spectrum, selfies_part_clamped, mask)
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
                 # Dynamic episode length prediction parameters
                 use_dynamic_max_steps=True,
                 dynamic_safety_margin=1.1,
                 **kwargs):
        """
        Enhanced MuZero Selfies Transformer with formula extraction and intelligent masking.
        
        Args:
            observation_shape: Total observation dimension (4096 spectrum + max_len SELFIES + formula_max_len formula = 4296)
            max_len: Maximum SELFIES token length (updated to 150)
            d_model: Transformer hidden dimension
            n_enc: Number of encoder layers
            n_dec: Number of decoder layers  
            n_head: Number of attention heads
            dropout: Dropout rate
            device: Device to run on
            target_formula: Deprecated - formula is now extracted from observations
            formula_max_len: Maximum length for chemical formula tokens
            pretrained_transformer_path: Path to pretrained transformer weights
            prevent_early_termination: Whether to apply intelligent END token masking
            min_formula_completion: Minimum completion ratio before END is allowed
            allow_early_end_after_steps: Allow END after this many steps even if incomplete
            use_dynamic_max_steps: Whether to use dynamic episode length prediction
            dynamic_safety_margin: Safety margin for dynamic episode length prediction
        """
        # Initialize parent class
        super().__init__(
            observation_shape=observation_shape,
            max_len=max_len,
            d_model=d_model,
            n_enc=n_enc,
            n_dec=n_dec,
            n_head=n_head,
            dropout=dropout,
            device=device,
            **kwargs
        )
        
        # Enhanced configuration
        self.observation_shape = observation_shape
        self.spectrum_dim = 4096
        self.formula_max_len = formula_max_len
        
        # Intelligent END token masking configuration
        self.prevent_early_termination = prevent_early_termination
        self.min_formula_completion = min_formula_completion
        self.allow_early_end_after_steps = allow_early_end_after_steps
        
        # Dynamic episode length prediction configuration
        self.use_dynamic_max_steps = use_dynamic_max_steps
        self.dynamic_safety_margin = dynamic_safety_margin
        self.predicted_max_steps_cache = {}  # Cache predictions by formula
        
        # Initialize enhanced tokenizer with vocab size 142
        self.tokenizer = SelfiesTokenizer(max_len=max_len)
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.eos_token_id = self.tokenizer.eos_token_id
        
        print(f"[INFO] Enhanced MuZero SELFIES Transformer initialized:")
        print(f"  - Observation shape: {observation_shape}")
        print(f"  - SELFIES max_len: {max_len}")  
        print(f"  - Formula max_len: {formula_max_len}")
        print(f"  - Vocab size: {self.vocab_size}")
        print(f"  - Intelligent END masking: {prevent_early_termination}")
        print(f"  - Dynamic max steps: {use_dynamic_max_steps}")
        
        # Load pretrained transformer if specified
        if pretrained_transformer_path:
            self._load_pretrained_transformer(pretrained_transformer_path)
        
        # Note: target_formula is now extracted dynamically from observations
        
        # Update dimensions for new observation structure
        # Observation: spectrum (4096) + selfies tokens (max_len) + formula tokens (formula_max_len)
        self.selfies_start_idx = self.spectrum_dim  # 4096
        self.formula_start_idx = self.spectrum_dim + max_len  # 4096 + max_len
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
    
    def get_predicted_max_steps(self, formula: str) -> int:
        """
        Get predicted max steps for a given formula, with caching.
        
        Args:
            formula (str): Chemical formula
            
        Returns:
            int: Predicted maximum steps
        """
        if formula in self.predicted_max_steps_cache:
            return self.predicted_max_steps_cache[formula]
        
        if self.use_dynamic_max_steps and formula:
            prediction = predict_episode_length_from_formula_transformer(formula, self.dynamic_safety_margin)
            predicted_steps = prediction['predicted_max_steps']
            self.predicted_max_steps_cache[formula] = predicted_steps
            return predicted_steps
        else:
            # Default fallback
            default_steps = 100
            self.predicted_max_steps_cache[formula] = default_steps
            return default_steps
    
    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced dynamics function with dynamic episode length checking and forced EOS termination.
        
        Args:
            latent_state: Current latent state
            action: Action taken
            
        Returns:
            Tuple of (next_state, reward)
        """
        batch_size = latent_state.shape[0]
        device = latent_state.device
        
        # Extract current SELFIES and formula from latent state
        current_selfies_list = self._extract_selfies_from_latent_state(latent_state)
        formula_list = self._extract_formula_from_latent_state(latent_state)
        
        # Check if we should force EOS termination based on predicted max steps
        forced_eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, (selfies, formula) in enumerate(zip(current_selfies_list, formula_list)):
            # Use proper tokenization method
            if hasattr(self.tokenizer, 'tokenize_selfies'):
                current_length = len(self.tokenizer.tokenize_selfies(selfies)) if selfies else 0
            else:
                # Fallback: estimate length from string length
                current_length = len(selfies) // 3 if selfies else 0  # Rough estimate
            predicted_max = self.get_predicted_max_steps(formula)
            
            if current_length >= predicted_max:
                forced_eos_mask[i] = True
                print(f"[INFO] Forcing EOS for batch {i}: length {current_length} >= predicted max {predicted_max}")
        
        # Force EOS action for sequences that have reached predicted limit
        modified_action = action.clone()
        modified_action[forced_eos_mask] = self.eos_token_id
        
        # Update latent state by appending the action token
        # latent_state shape: [batch_size, max_len] (current SELFIES tokens)
        # action shape: [batch_size]
        
        # Shift existing tokens left and append new action
        next_latent_state = latent_state.clone()
        
        # For each batch item, append the action if there's space
        for i in range(batch_size):
            current_tokens = next_latent_state[i]
            action_token = modified_action[i].item()
            
            # Find the first padding token position
            pad_positions = (current_tokens == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                # There's space, insert the action token
                insert_pos = pad_positions[0].item()
                next_latent_state[i, insert_pos] = action_token
            else:
                # No space, shift left and append at the end
                next_latent_state[i, :-1] = current_tokens[1:]
                next_latent_state[i, -1] = action_token
        
        # Compute reward
        reward = self._compute_reward(next_latent_state, modified_action, forced_eos_mask)
        
        return next_latent_state, reward

    def _compute_reward(self, latent_state: torch.Tensor, action: torch.Tensor, forced_eos_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for the current state, ensuring global reward server is called for EOS actions.
        
        Args:
            latent_state: Current latent state
            action: Action taken
            forced_eos_mask: Mask indicating which sequences were forced to terminate
            
        Returns:
            Tensor: Computed rewards
        """
        batch_size = latent_state.shape[0]
        device = latent_state.device
        rewards = torch.zeros(batch_size, device=device)
        
        # Check for EOS actions (both natural and forced)
        eos_mask = (action == self.eos_token_id)
        
        if eos_mask.any():
            # Extract SELFIES and spectrum information for reward computation
            current_selfies_list = self._extract_selfies_from_latent_state(latent_state)
            formula_list = self._extract_formula_from_latent_state(latent_state)
            
            # Get spectrum embeddings from latent state (assuming they're stored somehow)
            # This would need to be implemented based on how spectrum info is stored in latent state
            spectrum_embeds = self._extract_spectrum_from_latent_state(latent_state)
            
            # Try to get global reward function
            try:
                from lzero.model import global_reward_network
                reward_function = global_reward_network.get_reward_function()
                
                for i in range(batch_size):
                    if eos_mask[i]:
                        selfies = current_selfies_list[i]
                        formula = formula_list[i]
                        spectrum_embed = spectrum_embeds[i] if spectrum_embeds is not None else torch.zeros(4096, device=device)
                        
                        try:
                            # Call global reward network
                            similarity_score = reward_function(selfies, spectrum_embed, formula)
                            
                            # Apply penalty for forced termination
                            if forced_eos_mask[i]:
                                similarity_score *= 0.8  # 20% penalty for forced termination
                                print(f"[INFO] Applied forced termination penalty: {similarity_score}")
                            
                            rewards[i] = similarity_score
                            
                        except Exception as e:
                            print(f"[WARN] Error computing reward with global network: {e}")
                            # Fallback reward
                            rewards[i] = -1.0 if forced_eos_mask[i] else 0.0
                            
            except ImportError:
                print("[WARN] Global reward network not available, using fallback rewards")
                # Fallback: penalty for forced termination, neutral for natural EOS
                for i in range(batch_size):
                    if eos_mask[i]:
                        rewards[i] = -1.0 if forced_eos_mask[i] else 0.0
        
        return rewards
    
    def _extract_spectrum_from_latent_state(self, latent_state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract spectrum embeddings from latent state.
        
        Args:
            latent_state: Current latent state
            
        Returns:
            Tensor: Spectrum embeddings or None if not available
        """
        # This is a placeholder - the actual implementation would depend on
        # how spectrum information is stored in the latent state
        # For now, return None and let the reward computation handle it
        return None

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
                    from lzero.model.pretrain.config import PretrainConfig
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
            valid_mask = (token_ids != self.tokenizer.pad_token_id) & (token_ids != self.tokenizer.sos_token_id)
            valid_ids = token_ids[valid_mask].tolist()
            
            # Convert token IDs back to SELFIES string
            try:
                selfies_str = self.tokenizer.decode_to_selfies(valid_ids, skip_special_tokens=True)
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
        formula_ids = latent_state[:, self.formula_start_idx:].long()
        
        # Import BERT tokenizer for decoding
        formula_strings = []
        for batch_idx in range(formula_ids.size(0)):
            token_ids = formula_ids[batch_idx].tolist()
            # Decode formula tokens
            formula_str = self.formula_tokenizer.decode(token_ids, skip_special_tokens=True)
            
            formula_strings.append(formula_str.strip())
                
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
            token_id = self.tokenizer.token_to_id(action_token)
            action_mask[:, token_id] = 0  # Allow this token
        
        # Apply base action mask
        masked_logits = full_logits + action_mask
        
        # Formula-based masking is now handled by the environment's action mask
        # The model just needs to provide full vocabulary logits
                
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
        
        # Simplified completion check - just check if molecules look complete
        for idx, current_selfies in enumerate(current_selfies_list):
            # Basic completion check - can be enhanced later
            is_complete = len(current_selfies) > 10  # Simple heuristic
            completion_status.append(is_complete)
                
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
                # Get the token ID of the EOS token
                eos_token_id = self.tokenizer.eos_token_id
                
                # Check if the current action is the end token (action contains token IDs directly)
                action_token_ids = action.squeeze().long()
                if action_token_ids.dim() == 0:  # Single action
                    action_token_ids = action_token_ids.unsqueeze(0)
                
                reward_function = global_reward_network.get_reward_function()
                batch_size = next_latent_state.size(0)
                reward_scores = []
                
                for batch_idx in range(batch_size):
                    current_token_id = action_token_ids[batch_idx].item() if batch_idx < len(action_token_ids) else action_token_ids[0].item()
                    
                    # Only use reward network if this is the EOS token action
                    if current_token_id == eos_token_id:
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
        vocab_size = len(self.tokenizer.get_vocab())
        selfies_ids_clamped = torch.clamp(selfies_ids.long(), 0, vocab_size - 1)
        
        mask = (selfies_ids_clamped != self.tokenizer.pad_token_id) & (selfies_ids_clamped != self.tokenizer.sos_token_id)

        assert not torch.isnan(spectrum).any(), f"spectrum has nan: {spectrum}"
        assert not torch.isnan(selfies_ids_clamped).any(), f"selfies_ids_clamped has nan: {selfies_ids_clamped}"
        assert not torch.isnan(mask).any(), f"mask has nan: {mask}"
        
        # Get full vocabulary logits - action_index = token_id
        policy_logits, value = self.transformer(spectrum, selfies_ids_clamped, mask)
        assert not torch.isnan(policy_logits).any(), f"policy_logits has nan: {policy_logits}"
        value = value.unsqueeze(-1)
        
        return MZNetworkOutput(
            value=value, 
            reward=reward, 
            policy_logits=policy_logits, 
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
        vocab_size = len(self.tokenizer.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1)
        
        # Create mask for transformer
        mask = (selfies_part_clamped != self.tokenizer.pad_token_id) & (selfies_part_clamped != self.tokenizer.sos_token_id)
        
        # Use transformer directly for batch processing - get full vocabulary logits
        policy_logits, value = self.transformer(spectrum, selfies_part_clamped, mask)
        
        val = value.unsqueeze(-1)
        pol = policy_logits
        rew = [0.0] * B

        return MZNetworkOutput(value=val, reward=rew, policy_logits=pol, latent_state=obs)

# Add the prediction functions for dynamic episode length
def parse_molecular_formula_for_atoms_transformer(formula):
    """
    Parse a molecular formula and count total atoms.
    
    Args:
        formula (str): Molecular formula like 'C6H12O6'
        
    Returns:
        int: Total number of atoms
    """
    if not formula or formula == '':
        return 0
    
    # Remove any charges, brackets, or other symbols for counting
    clean_formula = re.sub(r'[+\-\[\]()]', '', formula)
    
    # Find all element-count pairs
    # Pattern matches: Capital letter, optional lowercase, optional digits
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, clean_formula)
    
    total_atoms = 0
    for element, count_str in matches:
        count = int(count_str) if count_str else 1
        total_atoms += count
    
    return total_atoms

def predict_episode_length_from_formula_transformer(formula, safety_margin=1.1):
    """
    Predict episode length for a given molecular formula based on empirical analysis.
    
    Args:
        formula (str): Molecular formula
        safety_margin (float): Additional safety margin (default: 1.1 = 10% extra)
        
    Returns:
        dict: Prediction results including recommended max_steps
    """
    try:
        atom_count = parse_molecular_formula_for_atoms_transformer(formula)
        
        if atom_count == 0:
            return {
                'predicted_max_steps': 100,  # Default fallback
                'atom_count': 0,
                'predicted_tokens': 0,
                'upper_bound_tokens': 0,
                'error': 'Could not parse formula'
            }
        
        # Empirically derived linear relationship from analysis
        # y = 0.594x + 12.508 (mean prediction)
        # y < 0.594x + 47.774 (99% upper bound)
        slope = 0.594
        intercept = 12.508
        upper_bound_intercept = 47.774
        
        predicted = slope * atom_count + intercept
        upper_bound = slope * atom_count + upper_bound_intercept
        
        # Add safety margin and buffer
        safe_upper_bound = upper_bound * safety_margin
        recommended_max_steps = int(safe_upper_bound) + 5  # Additional buffer
        
        return {
            'predicted_max_steps': recommended_max_steps,
            'atom_count': atom_count,
            'predicted_tokens': predicted,
            'upper_bound_tokens': upper_bound,
            'safe_upper_bound': safe_upper_bound,
            'linear_equation': f"y = 0.594 * {atom_count} + 12.508 = {predicted:.1f}",
            'upper_bound_equation': f"y < 0.594 * {atom_count} + 47.774 = {upper_bound:.1f}"
        }
        
    except Exception as e:
        return {
            'predicted_max_steps': 100,  # Default fallback
            'atom_count': 0,
            'predicted_tokens': 0,
            'upper_bound_tokens': 0,
            'error': str(e)
        }

if __name__ == "__main__":
    # quick sanity check
    data   = torch.randn(4246)  # Updated to new observation dimension
    model  = MuZeroSelfiesTransformer()
    out = model.initial_inference(data)
    print([t.shape for t in (out.value, out.reward, out.policy_logits, out.latent_state)])
    print(out)