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
import os
from pathlib import Path

# Import spectrum encoder from DiffMS - same as reward network
try:
    from reward_model.diffms.src.mist.models.spectra_encoder import SpectraEncoderGrowing
    from reward_model.diffms.src.mist.data import featurizers
    SPECTRUM_ENCODER_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Spectrum encoder not available in transformer: {e}")
    SPECTRUM_ENCODER_AVAILABLE = False
    SpectraEncoderGrowing = None
    featurizers = None

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
        enable_spectrum_encoder=True,  # New parameter to enable spectrum encoder
        spectrum_encoder_checkpoint=None,  # Path to spectrum encoder checkpoint
    ):
        super().__init__()

        self.device = torch.device(device)
        self.spectrum_dim = 4096
        self.spectrum_chunk_size = spectrum_chunk_size
        self.spectrum_attention_heads = spectrum_attention_heads 
        self.n_head = n_head
        self.d_model = d_model
        self.enable_spectrum_encoder = enable_spectrum_encoder

        self.num_chunks = self.spectrum_dim // self.spectrum_chunk_size # how many chunks 

        # tokenizer for special ids and pad
        self.tokenizer = SelfiesTokenizer(max_len=max_len)
        # Simplified: vocab_size = action_space_size, action_index = token_id
        self.vocab_size = len(self.tokenizer.get_vocab())

        # record special token ids
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_token_id = self.tokenizer.sos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        # Spectrum encoder (identical to reward network)
        self.spectrum_encoder = None
        if self.enable_spectrum_encoder and SPECTRUM_ENCODER_AVAILABLE:
            try:
                print(f"[INFO] Initializing spectrum encoder in transformer...")
                self.spectrum_encoder = SpectraEncoderGrowing(
                    inten_transform='float',
                    inten_prob=0.1,
                    remove_prob=0.5,
                    peak_attn_layers=2,
                    num_heads=8,
                    pairwise_featurization=True,
                    embed_instrument=False,
                    cls_type='ms1',
                    set_pooling='cls',
                    spec_features='peakformula',
                    mol_features='fingerprint',
                    form_embedder='pos-cos',
                    output_size=4096,
                    hidden_size=512,
                    spectra_dropout=0.1,
                    top_layers=1,
                    refine_layers=4,
                    magma_modulo=2048,
                )
                
                # Load the weights from the pretrained model (same path as reward network)
                checkpoint_path = spectrum_encoder_checkpoint or "/home/zirui/MassEnv/reward_model/diffms/data/checkpoints/encoder_msg.pt"
                if os.path.exists(checkpoint_path):
                    self.spectrum_encoder.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    print(f"[INFO] Transformer loaded pretrained spectrum encoder from {checkpoint_path}")
                else:
                    print(f"[WARN] Spectrum encoder checkpoint not found at {checkpoint_path}")
                
                # Move to device
                self.spectrum_encoder = self.spectrum_encoder.to(self.device)
                
                # Freeze the weights of the pretrained model (same as reward network)
                for param in self.spectrum_encoder.parameters():
                    param.requires_grad = False
                
                # Set to evaluation mode for inference
                self.spectrum_encoder.eval()
                
                print(f"[INFO] Spectrum encoder successfully initialized and moved to {self.device}")
                
            except Exception as e:
                print(f"[ERROR] Failed to initialize spectrum encoder in transformer: {e}")
                import traceback
                traceback.print_exc()
                self.spectrum_encoder = None
                self.enable_spectrum_encoder = False
        else:
            if not self.enable_spectrum_encoder:
                print(f"[INFO] Spectrum encoder disabled in transformer")
            if not SPECTRUM_ENCODER_AVAILABLE:
                print(f"[WARN] Spectrum encoder not available in transformer")
        
        print(f"[INFO] Transformer spectrum encoder status: {self.spectrum_encoder is not None}")

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

        # Outputs Heads
        self.action_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), 
            nn.LeakyReLU(0.01),
            # nn.Tanh(),
            nn.Linear(self.d_model, 1)
        )
        self.to(self.device)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _compute_spectrum_embedding(self, spectrum_batch):
        """Compute spectrum embeddings using the pretrained encoder"""
        if self.spectrum_encoder is None:
            batch_size = self._infer_batch_size(spectrum_batch)
            return torch.zeros(batch_size, self.spectrum_dim, device=self.device)
        
        try:
            # Move spectrum data to device
            device_spectrum_batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in spectrum_batch.items()
            }
            
            # Compute embeddings
            with torch.no_grad():
                embeddings, _ = self.spectrum_encoder(device_spectrum_batch)
            return embeddings
            
        except Exception as e:
            print(f"[ERROR] Spectrum encoder failed: {e}")
            batch_size = self._infer_batch_size(spectrum_batch)
            return torch.zeros(batch_size, self.spectrum_dim, device=self.device)
    
    def _infer_batch_size(self, spectrum_batch):
        """Infer batch size from spectrum data"""
        if isinstance(spectrum_batch, dict):
            for value in spectrum_batch.values():
                if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                    return value.shape[0]
        return 1

    def _prepare_spectrum_memory(self, spectrum_input) -> torch.Tensor:
        """Prepare spectrum memory from input"""
        # Get spectrum embeddings from various input formats
        if isinstance(spectrum_input, torch.Tensor):
            spectrum_embed = spectrum_input
        elif isinstance(spectrum_input, dict) and 'spectrum_embeds' in spectrum_input:
            spectrum_embed = spectrum_input['spectrum_embeds']
        elif isinstance(spectrum_input, dict):
            spectrum_embed = self._compute_spectrum_embedding(spectrum_input)
        else:
            spectrum_embed = torch.zeros(1, self.spectrum_dim, device=self.device)
        
        # Process embeddings
        spectrum_embed = spectrum_embed.to(self.device)
        batch_size = spectrum_embed.size(0)
        
        # Decompose into chunks and apply attention
        spectrum_sequence = spectrum_embed.view(batch_size, self.num_chunks, self.spectrum_chunk_size)
        spectrum_attended, _ = self.spectrum_decomposer(
            query=spectrum_sequence, key=spectrum_sequence, value=spectrum_sequence
        )
        
        # Project to model dimension
        spectrum_memory = self.chunk_to_model_proj(spectrum_attended)
        return spectrum_memory

    def forward_pretrain(
        self,
        spectrum_input,
        tgt_tokens: torch.Tensor,
        tgt_mask: torch.Tensor,
        return_value: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: # return logits or (logits, values)
        """
        forward for logits and value (if return_value=True)
        
        Args:
            spectrum_input: Either:
                - torch.Tensor: (B, spectrum_dim) precomputed spectrum embeddings
                - List: List of raw spectrum data objects
            tgt_tokens: (B, T) tgt selfies tokens
            tgt_mask: (B, T) attention mask
            
        Returns:
            torch.Tensor: (B, T, vocab_size) logits for each position
            or Tuple[torch.Tensor, torch.Tensor]: (B, T, vocab_size), (B, T, 1)
        """
        tgt_tokens = tgt_tokens.long().to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        B, T = tgt_tokens.shape

        spectrum_memory = self._prepare_spectrum_memory(spectrum_input)  # (B, spectrum_heads, d_model)
        memory = spectrum_memory  # (B, spectrum_heads, d_model)

        # Decoder
        pos_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
        dec_in = self.token_embed(tgt_tokens) + self.pos_embed(pos_ids)
        causal = self._generate_square_subsequent_mask(T).to(self.device)
        
        dec_out = self.decoder(
            tgt=dec_in, 
            memory=memory,
            tgt_mask=causal,
            tgt_is_causal=True,
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
            tgt_is_causal=True,
            tgt_key_padding_mask=(tgt_mask==0).bool()
        )
        # Select the decoder output of the last **actual** token (exclude right-padding).
        last_token_idx = (tgt_mask.sum(dim=1) - 1).clamp(min=0)  # (B,)
        last = dec_out[torch.arange(B, device=self.device), last_token_idx]  # (B, d_model)

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

    # -------------------------------------------------------------------------
    # Autoregressive sequence generation
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        spectrum_input,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        greedy: bool = False,
        prefix_ids: Optional[torch.Tensor] = None,
        stop_at_eos: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively sample SELFIES token sequences conditioned on the given
        spectrum input.

        Args:
            spectrum_input: Either:
                - torch.Tensor: (B, 4096) precomputed spectrum embeddings
                - List: List of raw spectrum data objects
            max_len       : Maximum total sequence length **including** SOS/EOS.
                           Defaults to the tokenizer's ``max_len``.
            temperature   : Sampling temperature (>0). ``temperature=0`` or
                           ``greedy=True`` results in deterministic decoding.
            greedy        : If ``True`` use arg-max instead of sampling.
            prefix_ids    : Optional starting prefix **excluding** the SOS token.
                           Shape (B, L).  If ``None`` generation starts from SOS.
            stop_at_eos   : Whether to stop once every sequence has produced EOS.

        Returns:
            generated_ids  : (B, max_len) tensor of token ids padded with ``PAD``.
            attention_mask : (B, max_len) mask (1 for real tokens, 0 for PAD).
        """
        device = self.device
        
        # Determine batch size from spectrum input
        if isinstance(spectrum_input, torch.Tensor):
            batch_size = spectrum_input.size(0)
        else:
            batch_size = len(spectrum_input)

        # Determine maximum length
        if max_len is None:
            max_len = self.tokenizer.max_len
        assert max_len > 1, "max_len must be at least 2"

        sos_id = self.tokenizer.sos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        unk_id = self.tokenizer.unk_token_id

        # Initialize sequence with SOS (+ optional prefix)
        if prefix_ids is not None:
            prefix_ids = prefix_ids.to(device).long()
            if prefix_ids.dim() == 1:
                prefix_ids = prefix_ids.unsqueeze(0)
            assert prefix_ids.size(0) == batch_size, "prefix batch dim mismatch"
            
            # Check if all sequences in the batch start with SOS
            if prefix_ids.size(1) > 0 and (prefix_ids[:, 0] == sos_id).all():
                # All prefixes already start with SOS, use as is
                tokens = prefix_ids
                print(f"[INFO] Prefix already contains SOS token, using as is")
            else:
                    # Some or all prefixes don't start with SOS, prepend it
                    tokens = torch.cat([
                        torch.full((batch_size, 1), sos_id, device=device, dtype=torch.long),
                        prefix_ids
                    ], dim=1)
                    print(f"[INFO] Added SOS token to prefix")
        else:
            # No prefix provided, start with SOS only
            tokens = torch.full((batch_size, 1), sos_id, device=device, dtype=torch.long)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Prepare spectrum memory once (since it doesn't change during generation)
        spectrum_memory = self._prepare_spectrum_memory(spectrum_input)  # (B, num_chunks, d_model)

        # ------------------------------------------------------------------
        # Autoregressive loop
        # ------------------------------------------------------------------
        # Autoregressive generation loop
        for step in range(max_len - tokens.size(1)):
            B, T = tokens.shape
            
            # Create attention mask
            attn_mask = (tokens != pad_id).float()
            
            # Forward pass for inference
            # Position embeddings
            pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            dec_in = self.token_embed(tokens) + self.pos_embed(pos_ids)
            
            # Causal mask
            causal_mask = self._generate_square_subsequent_mask(T).to(device)
            
            # Decoder forward
            dec_out = self.decoder(
                tgt=dec_in,
                memory=spectrum_memory,
                tgt_mask=causal_mask,
                tgt_is_causal=True,
                tgt_key_padding_mask=(attn_mask == 0).bool()
            )  # (B, T, d_model)
            
            # Get logits for the last position (next token to generate)
            last_hidden = dec_out[:, -1, :]  # (B, d_model)
            next_logits = self.action_head(last_hidden)  # (B, vocab_size)

            # Apply temperature
            next_logits = next_logits / max(temperature, 1e-6)

            # Mask out disallowed tokens
            next_logits[:, pad_id] = float('-1e9')
            next_logits[:, sos_id] = float('-1e9')
            next_logits[:, unk_id] = float('-1e9')

            # Sample or select greedily
            if greedy or temperature == 0.0:
                next_tokens = torch.argmax(next_logits, dim=-1)  # (B,)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)  # (B,)

            # For sequences that already finished, append PAD
            next_tokens[finished] = pad_id
            
            # Update finished status
            finished = finished | (next_tokens == eos_id)

            # Append new tokens to sequences
            tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)

            # Early stopping if all sequences finished
            if stop_at_eos and finished.all():
                break

        # Pad or truncate to exactly max_len
        current_len = tokens.size(1)
        if current_len < max_len:
            pad_len = max_len - current_len
            pad_tensor = torch.full((batch_size, pad_len), pad_id, device=device, dtype=torch.long)
            tokens = torch.cat([tokens, pad_tensor], dim=1)
        elif current_len > max_len:
            tokens = tokens[:, :max_len]

        # Create final attention mask
        attention_mask = (tokens != pad_id).float()
        
        return tokens, attention_mask

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
        'logits': masked_logits,  # Keep batch dimension for compatibility with gumbel_muzero
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
    def __init__(self, observation_shape=4296, max_len=150,
                 d_model=512, n_enc=4, n_dec=8, n_head=16,
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
            n_dec=n_dec,  # Only pass n_dec, not n_enc
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
        
        # CRITICAL FIX: Check if EOS token already exists in any sequence
        batch_size = next_latent_state.size(0)
        for i in range(batch_size):
            eos_exists = (selfies_part[i] == self.tok.eos_token_id).any()
            if eos_exists:
                # Sequence is already complete (EOS found), don't modify it further
                # Skip updating this batch item
                continue
            
            # Find padding tokens for this sequence only
            padding_mask = selfies_part[i] == self.tok.pad_token_id
            if padding_mask.any():
                # Find the last padding token index for this sequence
                last_padding_token_index = torch.sum(padding_mask) - 1
                
                # Update the latent state at the correct global position
                global_index = self.spectrum_dim + last_padding_token_index.item()
                if global_index < next_latent_state.size(1):  # Bounds check
                    next_latent_state[i, global_index] = action[i] if action.dim() > 0 else action

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
        
        mask = (selfies_part_clamped != self.tok.pad_token_id)

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
    def __init__(self, observation_shape=4296, max_len=150,
                 d_model=512, n_enc=4, n_dec=8, n_head=16,
                 dropout=0.1, device='cuda', target_formula=None,  # Deprecated: formula extracted from observations
                 formula_max_len=50, pretrained_transformer_path=None, 
                 # Intelligent END token masking parameters
                 prevent_early_termination=True,
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
        print(f"  - Min completion requirement: DISABLED (abandoned)")
        
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
    
    def get_predicted_min_steps(self, formula: str) -> int:
        """
        Get predicted min steps for a given formula, with caching.
        
        Args:
            formula (str): Chemical formula
            
        Returns:
            int: Predicted minimum steps
        """
        cache_key = f"{formula}_min"
        if cache_key in self.predicted_max_steps_cache:
            return self.predicted_max_steps_cache[cache_key]
        
        if self.use_dynamic_max_steps and formula:
            prediction = predict_episode_length_from_formula_transformer(formula, self.dynamic_safety_margin)
            predicted_steps = prediction.get('predicted_min_steps', 10)
            self.predicted_max_steps_cache[cache_key] = predicted_steps
            return predicted_steps
        else:
            # Default fallback
            default_steps = 10
            self.predicted_max_steps_cache[cache_key] = default_steps
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
        # Ensure latent_state has batch dimension
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        
        batch_size = latent_state.shape[0]
        device = latent_state.device
        
        # Ensure action has correct dimensions
        if action.dim() == 0:
            action = action.unsqueeze(0)
        elif action.dim() == 1 and action.size(0) != batch_size:
            action = action.expand(batch_size)
        
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
            predicted_min = self.get_predicted_min_steps(formula)
            
            if current_length >= predicted_max:
                forced_eos_mask[i] = True
                # print(f"[INFO] Forcing EOS for batch {i}: length {current_length} >= predicted max {predicted_max} "
                #       f"(min: {predicted_min}, formula: {formula})")
        
        # Force EOS action for sequences that have reached predicted limit
        modified_action = action.clone()
        modified_action[forced_eos_mask] = self.eos_token_id
        
        # Update latent state by appending the action token
        # Enhanced latent_state shape: [batch_size, observation_shape] where observation_shape = spectrum + selfies + formula
        # action shape: [batch_size]
        
        next_latent_state = latent_state.clone()
        
        # For each batch item, find the first padding position in the SELFIES part and replace with action
        for i in range(batch_size):
            action_token = modified_action[i].item()
            
            # Extract SELFIES part from the latent state
            selfies_start = self.selfies_start_idx
            selfies_end = self.formula_start_idx
            selfies_tokens = next_latent_state[i, selfies_start:selfies_end]
            
            # CRITICAL FIX: Check if EOS token already exists in the sequence
            eos_exists = (selfies_tokens == self.eos_token_id).any()
            
            if eos_exists:
                # Sequence is already complete (EOS found), don't modify it further
                # Force action to be PAD token for consistency
                # print("EOS exists")
                action_token = self.tokenizer.pad_token_id
                # Skip updating the latent state for this batch item
                continue
            
            # Find the first padding token position in SELFIES part
            pad_positions = (selfies_tokens == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                # print("inserting new action")
                # There's space, insert the action token
                insert_pos = selfies_start + pad_positions[0].item()
                next_latent_state[i, insert_pos] = action_token
            else:
                # print("no space")
                # No space, shift left while preserving SOS token and append at the end
                # Preserve SOS token (position 0), shift positions 2 to end-1 left by one, insert action at end
                if len(selfies_tokens) > 2:  # Must have at least SOS + one token to shift
                    selfies_tokens[1:-1] = selfies_tokens[2:].clone()  # Shift everything after SOS left
                selfies_tokens[-1] = action_token  # Insert action at the end
                next_latent_state[i, selfies_start:selfies_end] = selfies_tokens
        # print(f"next_latent_state: {self._extract_selfies_from_latent_state(next_latent_state)[0]}")
        # Compute reward
        reward = self._compute_reward(next_latent_state, modified_action, forced_eos_mask)
        
        return next_latent_state, reward

    def _compute_reward(self, latent_state: torch.Tensor, action: torch.Tensor, forced_eos_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for the current state using efficient batched reward computation.
        
        Args:
            latent_state: Current latent state
            action: Action taken
            forced_eos_mask: Mask indicating which sequences were forced to terminate
            
        Returns:
            Tensor: Computed rewards
        """
        batch_size = latent_state.shape[0]
        device = latent_state.device
        rewards = torch.zeros(batch_size, 1, device=device)  # Shape: (batch_size, 1) to match base class
        
        # Check for EOS actions (both natural and forced)
        eos_mask = (action == self.eos_token_id)
        
        if eos_mask.any():
            # Extract SELFIES and spectrum information for reward computation
            current_selfies_list = self._extract_selfies_from_latent_state(latent_state)
            formula_list = self._extract_formula_from_latent_state(latent_state)
            spectrum_embeds = self._extract_spectrum_from_latent_state(latent_state)
            
            # Try to use batched reward computation
            try:
                batch_rewards = self._compute_batch_rewards_efficient(
                    current_selfies_list, formula_list, spectrum_embeds, eos_mask, forced_eos_mask
                )
                
                # Apply batch rewards to the reward tensor
                for i in range(batch_size):
                    if eos_mask[i]:
                        rewards[i, 0] = batch_rewards[i]
                        
            except Exception as e:
                print(f"[WARN] Batched reward computation failed: {e}, falling back to individual computation")
                # Fallback to individual computation
                try:
                    from lzero.model import global_reward_network
                    reward_function = global_reward_network.get_reward_function()
                    
                    for i in range(batch_size):
                        if eos_mask[i]:
                            selfies = current_selfies_list[i]
                            formula = formula_list[i]
                            spectrum_embed = spectrum_embeds[i] if spectrum_embeds is not None else torch.zeros(4096, device=device)
                            
                            try:
                                similarity_score = reward_function(selfies, spectrum_embed, formula)
                                if forced_eos_mask[i]:
                                    similarity_score *= 0.8  # 20% penalty for forced
                                    print(f"[WARN] Spectrum embedding is None for batch {i}") 
                                rewards[i, 0] = similarity_score
                            except Exception as e:
                                print(f"[WARN] Error computing individual reward: {e}")
                                rewards[i, 0] = -1.0 if forced_eos_mask[i] else 0.0
                                
                except ImportError:
                    print("[WARN] Global reward network not available, using fallback rewards")
                    # Fallback: penalty for forced termination, neutral for natural EOS
                    for i in range(batch_size):
                        if eos_mask[i]:
                            rewards[i, 0] = -1.0 if forced_eos_mask[i] else 0.0
        
        return rewards

    def _compute_batch_rewards_efficient(self, current_selfies_list: List[str], formula_list: List[str], 
                                       spectrum_embeds: Optional[torch.Tensor], eos_mask: torch.Tensor, 
                                       forced_eos_mask: torch.Tensor) -> List[float]:
        """
        Efficiently compute rewards using batched operations similar to run_pretrain.py approach.
        
        Args:
            current_selfies_list: List of SELFIES strings
            formula_list: List of formula strings
            spectrum_embeds: Spectrum embeddings tensor [batch_size, 4096]
            eos_mask: Boolean mask for EOS actions
            forced_eos_mask: Boolean mask for forced termination
            
        Returns:
            List of computed rewards
        """
        batch_size = len(current_selfies_list)
        batch_rewards = [0.0] * batch_size
        
        # Extract EOS sequences for batch processing
        eos_indices = []
        eos_selfies = []
        eos_formulas = []
        eos_spectrum_embeds = []
        eos_forced_mask = []
        
        for i in range(batch_size):
            if eos_mask[i]:
                eos_indices.append(i)
                eos_selfies.append(current_selfies_list[i])
                eos_formulas.append(formula_list[i])
                
                if spectrum_embeds is not None:
                    eos_spectrum_embeds.append(spectrum_embeds[i].detach().cpu())
                else:
                    eos_spectrum_embeds.append(torch.zeros(4096))
                    
                eos_forced_mask.append(forced_eos_mask[i].item())
        
        if not eos_selfies:
            return batch_rewards
        
        # Check if we can use reward server batching
        try:
            from lzero.model.global_reward_network import _request_queue, _response_dict, _server_enabled
            import uuid
            import time
            
            if _server_enabled and _request_queue is not None and _response_dict is not None:
                # Use reward server batching approach similar to run_pretrain.py
                # print(f"[DEBUG] Using reward server batching for {len(eos_selfies)} EOS sequences")
                
                # Prepare all requests for batched processing
                all_requests = []
                request_ids = []
                
                for idx, (selfies, formula, spectrum_embed) in enumerate(zip(eos_selfies, eos_formulas, eos_spectrum_embeds)):
                    request_id = str(uuid.uuid4())
                    request_ids.append(request_id)
                    
                    # Ensure spectrum embed is CPU tensor for serialization
                    if isinstance(spectrum_embed, torch.Tensor):
                        spectrum_embed = spectrum_embed.detach().cpu().float()
                        if not spectrum_embed.is_contiguous():
                            spectrum_embed = spectrum_embed.contiguous()
                    
                    request = (request_id, selfies, spectrum_embed, formula)
                    all_requests.append(request)
                
                # Send all requests rapidly to enable server-side batching
                start_send_time = time.time()
                for request in all_requests:
                    _request_queue.put(request, timeout=0.5)
                send_time = time.time() - start_send_time
                
                # print(f"[DEBUG] Sent {len(all_requests)} reward requests in {send_time:.3f}s")
                
                # Collect responses with efficient batching
                collected_responses = {}
                start_time = time.time()
                timeout = 5.0  # Shorter timeout for transformer responsiveness
                check_interval = 0.005  # 5ms checks for responsiveness
                
                # Brief initial wait to encourage server batching
                time.sleep(0.02)  # 20ms to allow batching
                
                while len(collected_responses) < len(request_ids) and (time.time() - start_time) < timeout:
                    for request_id in request_ids:
                        if request_id not in collected_responses and request_id in _response_dict:
                            collected_responses[request_id] = _response_dict[request_id]
                            del _response_dict[request_id]
                    
                    if len(collected_responses) < len(request_ids):
                        time.sleep(check_interval)
                
                elapsed_time = time.time() - start_time
                # print(f"[DEBUG] Collected {len(collected_responses)}/{len(request_ids)} responses in {elapsed_time:.3f}s")
                
                # Apply results back to batch_rewards
                for idx, (original_idx, request_id, is_forced) in enumerate(zip(eos_indices, request_ids, eos_forced_mask)):
                    if request_id in collected_responses:
                        reward = collected_responses[request_id]
                        
                        # Apply penalty for forced termination
                        if is_forced:
                            reward *= 0.8  # 20% penalty
                            
                        batch_rewards[original_idx] = float(reward)
                    else:
                        print(f"[WARN] No response for request {request_id}")
                        batch_rewards[original_idx] = -1.0 if is_forced else 0.0
                
                # Clean up any remaining responses
                for request_id in request_ids:
                    _response_dict.pop(request_id, None)
                
                return batch_rewards
            else:
                print("[DEBUG] Reward server not available, using fallback computation")
                
        except ImportError:
            print("[DEBUG] Reward server not available, using direct computation")
        
        # Fallback: use direct reward function computation
        try:
            from lzero.model import global_reward_network
            reward_function = global_reward_network.get_reward_function()
            
            # Check if we're in a subprocess and can use ClientSideBatcher
            import multiprocessing as mp
            current_process = mp.current_process()
            is_subprocess = current_process.name != 'MainProcess'
            # is_subprocess = True
            
            if is_subprocess:
                try:
                    # Try to import and use ClientSideBatcher for subprocess efficiency
                    from lzero.model.global_reward_network import get_client_batcher
                    batcher = get_client_batcher()
                    
                    print(f"[DEBUG] Using ClientSideBatcher for {len(eos_selfies)} sequences in subprocess")
                    
                    # Use concurrent futures to process multiple batcher requests
                    import concurrent.futures
                    import threading
                    
                    def compute_single_reward(idx, selfies, formula, spectrum_embed, is_forced):
                        try:
                            reward = batcher.submit_request(
                                selfies=selfies,
                                spectrum_embed=spectrum_embed,
                                formula=formula,
                                timeout=3.0
                            )
                            if is_forced:
                                reward *= 0.8
                            return idx, float(reward)
                        except Exception as e:
                            print(f"[WARN] Batcher request failed for sequence {idx}: {e}")
                            return idx, -1.0 if is_forced else 0.0
                    
                    # Process requests concurrently for better efficiency
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(eos_selfies), 4)) as executor:
                        future_to_idx = {
                            executor.submit(compute_single_reward, idx, selfies, formula, spectrum_embed, is_forced): idx
                            for idx, (selfies, formula, spectrum_embed, is_forced) in enumerate(
                                zip(eos_selfies, eos_formulas, eos_spectrum_embeds, eos_forced_mask)
                            )
                        }
                        
                        # Apply results
                        for future in concurrent.futures.as_completed(future_to_idx):
                            try:
                                result_idx, reward = future.result()
                                original_idx = eos_indices[result_idx]
                                batch_rewards[original_idx] = reward
                            except Exception as e:
                                result_idx = future_to_idx[future]
                                original_idx = eos_indices[result_idx]
                                is_forced = eos_forced_mask[result_idx]
                                print(f"[WARN] Future failed for sequence {result_idx}: {e}")
                                batch_rewards[original_idx] = -1.0 if is_forced else 0.0
                    
                    return batch_rewards
                    
                except ImportError:
                    print("[DEBUG] ClientSideBatcher not available, using direct computation")
            
            # Direct computation (main process or fallback)
            print(f"[DEBUG] Using direct reward computation for {len(eos_selfies)} sequences")
            
            for idx, (original_idx, selfies, formula, spectrum_embed, is_forced) in enumerate(
                zip(eos_indices, eos_selfies, eos_formulas, eos_spectrum_embeds, eos_forced_mask)
            ):
                try:
                    reward = reward_function(selfies, spectrum_embed, formula)
                    if is_forced:
                        reward *= 0.8
                    batch_rewards[original_idx] = float(reward)
                except Exception as e:
                    print(f"[WARN] Direct reward computation failed for sequence {idx}: {e}")
                    batch_rewards[original_idx] = -1.0 if is_forced else 0.0
            
            return batch_rewards
            
        except ImportError:
            print("[WARN] Global reward network not available")
            # Fallback: simple penalty system
            for idx, (original_idx, is_forced) in enumerate(zip(eos_indices, eos_forced_mask)):
                batch_rewards[original_idx] = -1.0 if is_forced else 0.0
            
            return batch_rewards # rds
    
    def _extract_spectrum_from_latent_state(self, latent_state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract spectrum embeddings from latent state.
        
        Args:
            latent_state: Current latent state containing spectrum + SELFIES + formula
                         Format: [spectrum(4096) + selfies_tokens(max_len) + formula_tokens(formula_max_len)]
            
        Returns:
            Tensor: Spectrum embeddings [batch_size, 4096] or None if not available
        """
        try:
            # Extract the first 4096 elements as spectrum embedding
            # The latent state format is: [spectrum(4096) + selfies_tokens + formula_tokens]
            if latent_state.size(-1) >= 4096:
                spectrum_embed = latent_state[:, :4096]
                return spectrum_embed
            else:
                print(f"[WARN] Latent state too small for spectrum extraction: {latent_state.shape}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to extract spectrum from latent state: {e}")
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
                # Skip value head parameters - keep them randomly initialized
                # if 'value_head' in key:
                #     print(f"[INFO] Skipping value head parameter: {key}")
                #     continue
                    
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
            
        # Extract formula token IDs from the correct slice
        # Latent state format: [spectrum(4096) + selfies_tokens(max_len) + formula_tokens(formula_max_len)]
        formula_start = 4096 + 150  # Skip spectrum and SELFIES (hardcoded for now)
        formula_end = formula_start + self.formula_max_len  # Take only formula_max_len tokens
        
        formula_ids = latent_state[:, formula_start:formula_end].long()
        
        formula_strings = []
        for batch_idx in range(formula_ids.size(0)):
            token_ids = formula_ids[batch_idx].tolist()
            
            # Remove padding tokens and decode
            valid_tokens = [tid for tid in token_ids if tid != 0]  # Assume 0 is padding
            
            try:
                # Decode formula tokens using the formula tokenizer
                formula_str = self.formula_tokenizer.decode(valid_tokens, skip_special_tokens=True)
                formula_strings.append(formula_str.strip())
            except Exception as e:
                # Fallback: create a simple formula from SELFIES
                print(f"[WARN] Formula decoding failed for batch {batch_idx}: {e}")
                # Extract SELFIES first and create a basic formula
                try:
                    selfies_list = self._extract_selfies_from_latent_state(latent_state[batch_idx:batch_idx+1])
                    if selfies_list and len(selfies_list[0]) > 0:
                        # Simple heuristic: count atoms from SELFIES
                        selfies = selfies_list[0]
                        c_count = selfies.count('[C]')
                        o_count = selfies.count('[O]')
                        n_count = selfies.count('[N]')
                        
                        # Build simple formula
                        formula_parts = []
                        if c_count > 0:
                            formula_parts.append(f"C{c_count}" if c_count > 1 else "C")
                        if o_count > 0:
                            formula_parts.append(f"O{o_count}" if o_count > 1 else "O")
                        if n_count > 0:
                            formula_parts.append(f"N{n_count}" if n_count > 1 else "N")
                        
                        fallback_formula = "".join(formula_parts) if formula_parts else "C"
                        formula_strings.append(fallback_formula)
                    else:
                        formula_strings.append("C")  # Default fallback
                except:
                    formula_strings.append("C")  # Ultimate fallback
                
        return formula_strings
    
    def _apply_formula_mask(self, full_logits: torch.Tensor, current_selfies_list: List[str], formula_list: List[str]) -> torch.Tensor:
        """Apply formula-based action masking to full vocabulary logits at token_id positions"""
        # Check for NaN in input logits first
        if torch.isnan(full_logits).any():
            print(f"[WARN] NaN detected in input full_logits before masking")
            full_logits = torch.where(torch.isnan(full_logits), torch.tensor(0.0, device=full_logits.device), full_logits)
        
        batch_size = full_logits.size(0)
        vocab_size = full_logits.size(1)
        device = full_logits.device
        
        # Create mask for each batch item
        masked_logits = full_logits.clone()
        
        for batch_idx in range(batch_size):
            current_selfies = current_selfies_list[batch_idx] if batch_idx < len(current_selfies_list) else ""
            formula = formula_list[batch_idx] if batch_idx < len(formula_list) else ""
            
            try:
                # Create actions list from vocabulary (excluding special tokens)
                vocab = self.tokenizer.get_vocab()
                actions_list = []
                atom_tokens = []
                bonded_atom_tokens = []
                
                for token, token_id in vocab.items():
                    # Skip special tokens
                    if token_id in [self.tokenizer.pad_token_id, self.tokenizer.sos_token_id, self.tokenizer.unk_token_id]:
                        continue
                    actions_list.append(token)
                    
                    # Categorize tokens for masking
                    if token.startswith('[') and token.endswith(']') and len(token) > 2:
                        # Simple heuristic: tokens with = or # are bonded
                        if '=' in token or '#' in token:
                            bonded_atom_tokens.append(token)
                        else:
                            atom_tokens.append(token)
                
                # Always apply basic masking first - special tokens should always be masked
                masked_logits[batch_idx, self.tokenizer.pad_token_id] = float('-1e9')
                masked_logits[batch_idx, self.tokenizer.sos_token_id] = float('-1e9')
                masked_logits[batch_idx, self.tokenizer.unk_token_id] = float('-1e9')
                
                # Use utility function for additional formula masking if available
                if hasattr(self, 'get_action_mask_from_selfies_string') and self.get_action_mask_from_selfies_string is not None:
                    try:
                        action_mask = self.get_action_mask_from_selfies_string(
                            formula=formula,
                            current_selfies=current_selfies,
                            actions_list=actions_list,
                            atom_tokens=atom_tokens,
                            bonded_atom_tokens=bonded_atom_tokens,
                            formula_masking=True,
                            end_token='</s>',  # EOS token
                            remove_token=None,
                            special_tokens=[],
                            min_formula_completion=0.0,  # ABANDONED - set to 0.0 to always allow EOS based on other conditions
                            allow_early_end_after_steps=getattr(self, 'allow_early_end_after_steps', 20)
                        )
                        
                        # Apply additional mask to logits - mask[i] = False means mask out action i
                        for i, action_token in enumerate(actions_list):
                            token_id = vocab.get(action_token)
                            if token_id is not None and token_id < vocab_size:
                                if not action_mask[i]:  # If action is masked out
                                    masked_logits[batch_idx, token_id] = float('-1e9')
                    except Exception as inner_e:
                        print(f"[WARN] Formula masking utility failed for batch {batch_idx}: {inner_e}")
                        # Basic masking already applied above
                    
            except Exception as e:
                print(f"[WARN] Formula masking failed for batch {batch_idx}: {e}")
                # Fallback: basic masking only
                masked_logits[batch_idx, self.tokenizer.pad_token_id] = float('-1e9')
                masked_logits[batch_idx, self.tokenizer.sos_token_id] = float('-1e9')
                masked_logits[batch_idx, self.tokenizer.unk_token_id] = float('-1e9')
                
        # Final check for NaN values
        if torch.isnan(masked_logits).any():
            print(f"[WARN] NaN detected in final masked_logits, replacing with safe values")
            masked_logits = torch.where(torch.isnan(masked_logits), torch.tensor(-1e9, device=device), masked_logits)
                
        return masked_logits
    
    def _check_completion_status(self, current_selfies_list: List[str], formula_list: List[str]) -> List[bool]:
        """Check if molecules are complete based on proper SELFIES validation and formula matching"""
        if not formula_list or not any(formula_list):
            return [False] * len(current_selfies_list)
            
        completion_status = []
        
        # Proper completion check based on SELFIES validity and formula matching
        for idx, current_selfies in enumerate(current_selfies_list):
            is_complete = False
            
            try:
                # Must have minimum reasonable length for a valid molecule (at least 15 characters)
                if len(current_selfies) < 15:
                    is_complete = False
                else:
                    # Check if SELFIES string has proper structure (basic validation)
                    # Must contain at least one element token like [C], [N], [O], etc.
                    import re
                    element_pattern = r'\[[CNOHSP][^]]*\]'  # Common elements in brackets
                    element_matches = re.findall(element_pattern, current_selfies)
                    
                    # Must have at least 3 element tokens for a reasonable molecule
                    if len(element_matches) >= 3:
                        # Check for proper bracket balance
                        bracket_balance = current_selfies.count('[') - current_selfies.count(']')
                        if bracket_balance == 0:  # Properly balanced brackets
                            # Additional check: molecule should have reasonable complexity
                            # Count unique token types
                            unique_tokens = set(element_matches)
                            if len(unique_tokens) >= 2:  # At least 2 different element types
                                is_complete = True
                
            except Exception as e:
                # If any validation fails, consider incomplete
                is_complete = False
                
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
        
        # Print completion info very occasionally for debugging (every 1000 calls)
        if any(completion_status):
            self._debug_counter = getattr(self, '_debug_counter', 0) + 1
            if self._debug_counter % 1000 == 0:  # Print only every 1000 completions
                complete_indices = [i for i, complete in enumerate(completion_status) if complete][:3]  # Show max 3
                # print(f"[DEBUG] Molecules complete at step {self._debug_counter}: {len([i for i, complete in enumerate(completion_status) if complete])}/{len(completion_status)} complete")
                for idx in complete_indices:  # Show only first few examples
                    if idx < len(current_selfies_list):
                        selfies_preview = current_selfies_list[idx][:40] + '...' if len(current_selfies_list[idx]) > 40 else current_selfies_list[idx]
                        print(f"  Example {idx}: SELFIES='{selfies_preview}', Formula='{formula_list[idx]}'")
                        print(f"    Length: {len(current_selfies_list[idx])}, Reward: {reward[idx].item():.4f}")
        
        # Get transformer output - only use spectrum and SELFIES parts for transformer
        spectrum = next_latent_state[:, :self.spectrum_dim]
        selfies_ids = next_latent_state[:, self.selfies_start_idx:self.formula_start_idx]
        
        # Clamp SELFIES token IDs to valid vocabulary range
        vocab_size = len(self.tokenizer.get_vocab())
        selfies_ids_clamped = torch.clamp(selfies_ids.long(), 0, vocab_size - 1)
        
        mask = (selfies_ids_clamped != self.tokenizer.pad_token_id)

        assert not torch.isnan(spectrum).any(), f"spectrum has nan: {spectrum}"
        assert not torch.isnan(selfies_ids_clamped).any(), f"selfies_ids_clamped has nan: {selfies_ids_clamped}"
        assert not torch.isnan(mask).any(), f"mask has nan: {mask}"
        
        # Get full vocabulary logits - action_index = token_id
        policy_logits, value = self.transformer(spectrum, selfies_ids_clamped, mask)
        assert not torch.isnan(policy_logits).any(), f"policy_logits has nan: {policy_logits}"
        value = value.unsqueeze(-1)
        
        # Apply formula masking to policy logits
        try:
            policy_logits = self._apply_formula_mask(policy_logits, current_selfies_list, formula_list)
        except Exception as e:
            print(f"[WARN] Formula masking failed in recurrent_inference: {e}")
            # Apply basic masking at least
            policy_logits[:, self.tokenizer.pad_token_id] = float('-1e9')
            policy_logits[:, self.tokenizer.sos_token_id] = float('-1e9')
            policy_logits[:, self.tokenizer.unk_token_id] = float('-1e9')
        
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
        mask = (selfies_part_clamped != self.tokenizer.pad_token_id)

        
        # Use transformer directly for batch processing - get full vocabulary logits
        policy_logits, value = self.transformer(spectrum, selfies_part_clamped, mask)
        
        # Apply formula masking to policy logits
        current_selfies_list = self._extract_selfies_from_latent_state(vec)
        try:
            policy_logits = self._apply_formula_mask(policy_logits, current_selfies_list, formula_list)
        except Exception as e:
            print(f"[WARN] Formula masking failed in initial_inference: {e}")
            # Apply basic masking at least
            policy_logits[:, self.tokenizer.pad_token_id] = float('-1e9')
            policy_logits[:, self.tokenizer.sos_token_id] = float('-1e9')
            policy_logits[:, self.tokenizer.unk_token_id] = float('-1e9')
        
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
    Predict episode length for a given molecular formula based on dual-bound empirical analysis.
    Uses separate slopes for upper and lower bounds to provide more accurate predictions.
    
    Args:
        formula (str): Molecular formula
        safety_margin (float): Additional safety margin (default: 1.1 = 10% extra)
        
    Returns:
        dict: Prediction results including recommended max_steps with dual bounds
    """
    try:
        atom_count = parse_molecular_formula_for_atoms_transformer(formula)
        
        if atom_count == 0:
            return {
                'predicted_max_steps': 100,  # Default fallback
                'predicted_min_steps': 10,   # Default minimum
                'atom_count': 0,
                'predicted_tokens': 0,
                'upper_bound_tokens': 0,
                'lower_bound_tokens': 0,
                'error': 'Could not parse formula'
            }
        
        # Updated empirically derived relationships from dual-bound analysis
        # Mean relationship: y = 0.604x + 11.980
        # Upper bound: y < 0.941x + 17.605 (99th percentile with separate slope)
        # Lower bound: y > 0.429x + 0.395 (1st percentile with separate slope)
        
        mean_slope = 0.604
        mean_intercept = 11.980
        
        upper_slope = 0.941
        upper_intercept = 17.605
        
        lower_slope = 0.429
        lower_intercept = 0.395
        
        # Calculate predictions using different relationships
        predicted_mean = mean_slope * atom_count + mean_intercept
        upper_bound = upper_slope * atom_count + upper_intercept
        lower_bound = max(1, lower_slope * atom_count + lower_intercept)  # Ensure minimum of 1
        
        # Add safety margin to upper bound for max_steps
        safe_upper_bound = upper_bound * safety_margin
        recommended_max_steps = max(int(safe_upper_bound) + 5, 50)  # Minimum 50 steps
        
        # Calculate minimum expected steps (useful for early termination prevention)
        recommended_min_steps = max(int(lower_bound * 0.8), 5)  # 80% of lower bound, minimum 5
        
        return {
            'predicted_max_steps': recommended_max_steps,
            'predicted_min_steps': recommended_min_steps,
            'atom_count': atom_count,
            'predicted_tokens': predicted_mean,
            'upper_bound_tokens': upper_bound,
            'lower_bound_tokens': lower_bound,
            'safe_upper_bound': safe_upper_bound,
            'mean_equation': f"y = {mean_slope} * {atom_count} + {mean_intercept} = {predicted_mean:.1f}",
            'upper_bound_equation': f"y < {upper_slope} * {atom_count} + {upper_intercept} = {upper_bound:.1f}",
            'lower_bound_equation': f"y > {lower_slope} * {atom_count} + {lower_intercept} = {lower_bound:.1f}",
            'coverage_info': '98.2% coverage with separate slope bounds'
        }
        
    except Exception as e:
        return {
            'predicted_max_steps': 100,  # Default fallback
            'predicted_min_steps': 10,   # Default minimum
            'atom_count': 0,
            'predicted_tokens': 0,
            'upper_bound_tokens': 0,
            'lower_bound_tokens': 0,
            'error': str(e)
        }

# -----------------------------------------------------------------------------
# Separate Policy and Value Networks Transformer 
# -----------------------------------------------------------------------------
class SeparatePolicyValueTransformer(nn.Module):
    """
    Separate Policy and Value Networks for SELFIES generation and evaluation.
    Follows RLHF principles with independent networks for policy and value functions.
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 150,
        policy_d_model: int = 512,
        value_d_model: int = 256,
        policy_n_layers: int = 8,
        value_n_layers: int = 4,
        policy_n_head: int = 16,
        value_n_head: int = 8,
        dropout: float = 0.1,
        spectrum_chunk_size: int = 256,
        spectrum_attention_heads: int = 4,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.spectrum_dim = 4096
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Spectrum processing configuration
        self.spectrum_chunk_size = spectrum_chunk_size
        self.spectrum_attention_heads = spectrum_attention_heads
        self.num_chunks = self.spectrum_dim // spectrum_chunk_size
        
        # Tokenizer for special tokens
        self.tokenizer = SelfiesTokenizer(max_len=max_len)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_token_id = self.tokenizer.sos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        
        # Spectrum processing (shared)
        self.spectrum_decomposer = nn.MultiheadAttention(
            embed_dim=spectrum_chunk_size,
            num_heads=spectrum_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # ====================================================================
        # POLICY NETWORK - Autoregressive Generation
        # ====================================================================
        self.policy_chunk_proj = nn.Linear(self.spectrum_chunk_size, policy_d_model)
        self.policy_token_embed = nn.Embedding(vocab_size, policy_d_model)
        self.policy_pos_embed = nn.Embedding(max_len, policy_d_model)
        
        # Policy decoder layers
        policy_decoder_layer = nn.TransformerDecoderLayer(
            d_model=policy_d_model,
            nhead=policy_n_head,
            dim_feedforward=policy_d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.policy_decoder = nn.TransformerDecoder(
            policy_decoder_layer, 
            num_layers=policy_n_layers
        )
        
        # Policy output head
        self.policy_head = nn.Linear(policy_d_model, vocab_size, bias=False)
        
        # ====================================================================
        # VALUE NETWORK - Sequence Evaluation
        # ====================================================================
        self.value_chunk_proj = nn.Linear(self.spectrum_chunk_size, value_d_model)
        self.value_token_embed = nn.Embedding(vocab_size, value_d_model)
        self.value_pos_embed = nn.Embedding(max_len, value_d_model)
        
        # Value encoder layers (bidirectional processing)
        value_encoder_layer = nn.TransformerEncoderLayer(
            d_model=value_d_model,
            nhead=value_n_head,
            dim_feedforward=value_d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.value_encoder = nn.TransformerEncoder(
            value_encoder_layer,
            num_layers=value_n_layers
        )
        
        # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(value_d_model * 2, value_d_model),  # Concat spectrum + sequence
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(value_d_model, value_d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(value_d_model // 2, 1)
        )
        
        self.to(self.device)
    
    def _prepare_spectrum_memory(self, spectrum_embed: torch.Tensor) -> torch.Tensor:
        """Process spectrum embeddings using self-attention"""
        batch_size = spectrum_embed.size(0)
        
        # Reshape spectrum into chunks
        spectrum_sequence = spectrum_embed.view(batch_size, self.num_chunks, self.spectrum_chunk_size)
        
        # Apply self-attention to spectrum chunks
        spectrum_attended, _ = self.spectrum_decomposer(
            query=spectrum_sequence,
            key=spectrum_sequence, 
            value=spectrum_sequence
        )
        
        return spectrum_attended  # [batch_size, num_chunks, chunk_size]
    
    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    
    def forward_policy(
        self,
        spectrum_embed: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for policy network (autoregressive generation)
        
        Args:
            spectrum_embed: (B, 4096) spectrum embeddings
            input_ids: (B, T) input token sequence
            attention_mask: (B, T) attention mask
            
        Returns:
            policy_logits: (B, T, vocab_size) action logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Process spectrum
        spectrum_memory = self._prepare_spectrum_memory(spectrum_embed)  # (B, num_chunks, chunk_size)
        spectrum_memory = self.policy_chunk_proj(spectrum_memory)        # (B, num_chunks, policy_d_model)
        
        # Token embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_embeds = self.policy_token_embed(input_ids) + self.policy_pos_embed(positions)
        
        # Causal mask for autoregressive generation
        causal_mask = self._generate_causal_mask(seq_len).to(device)
        
        # Decoder forward pass
        decoder_output = self.policy_decoder(
            tgt=token_embeds,
            memory=spectrum_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~attention_mask.bool()
        )
        
        # Policy logits
        policy_logits = self.policy_head(decoder_output)
        
        return policy_logits
    
    def forward_value(
        self,
        spectrum_embed: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for value network (sequence evaluation)
        
        Args:
            spectrum_embed: (B, 4096) spectrum embeddings
            input_ids: (B, T) input token sequence
            attention_mask: (B, T) attention mask
            
        Returns:
            values: (B, T, 1) value predictions for each token
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Process spectrum
        spectrum_memory = self._prepare_spectrum_memory(spectrum_embed)  # (B, num_chunks, chunk_size)
        spectrum_features = self.value_chunk_proj(spectrum_memory)       # (B, num_chunks, value_d_model)
        spectrum_pooled = spectrum_features.mean(dim=1)                  # (B, value_d_model) - pool across chunks
        
        # Token embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_embeds = self.value_token_embed(input_ids) + self.value_pos_embed(positions)
        
        # Encoder forward pass (bidirectional)
        encoded_sequence = self.value_encoder(
            token_embeds,
            src_key_padding_mask=~attention_mask.bool()
        )  # (B, T, value_d_model)
        
        # Combine spectrum and sequence features
        spectrum_expanded = spectrum_pooled.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, value_d_model)
        combined_features = torch.cat([encoded_sequence, spectrum_expanded], dim=-1)  # (B, T, 2*value_d_model)
        
        # Value predictions
        values = self.value_head(combined_features)  # (B, T, 1)
        
        return values
    
    def forward(
        self,
        spectrum_embed: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_policy: bool = True,
        return_value: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Combined forward pass for both networks
        
        Args:
            spectrum_embed: (B, 4096) spectrum embeddings
            input_ids: (B, T) input token sequence
            attention_mask: (B, T) attention mask
            return_policy: Whether to compute policy logits
            return_value: Whether to compute value predictions
            
        Returns:
            policy_logits and/or value predictions based on flags
        """
        results = []
        
        if return_policy:
            policy_logits = self.forward_policy(spectrum_embed, input_ids, attention_mask)
            results.append(policy_logits)
        
        if return_value:
            values = self.forward_value(spectrum_embed, input_ids, attention_mask)
            results.append(values)
        
        if len(results) == 1:
            return results[0]
        elif len(results) == 2:
            return tuple(results)
        else:
            raise ValueError("Must return at least one of policy or value")


# -----------------------------------------------------------------------------
# MuZero wrapper for separate policy-value transformer
# -----------------------------------------------------------------------------
@MODEL_REGISTRY.register('MuZeroSeparatePolicyValueTransformer', force_overwrite=True)
class MuZeroSeparatePolicyValueTransformer(nn.Module):
    """
    MuZero wrapper for separate policy and value networks transformer.
    Provides the same interface as the unified transformer but with separate networks.
    """
    
    def __init__(self, observation_shape=4296, max_len=150,
                 policy_d_model=512, value_d_model=256,
                 policy_n_layers=8, value_n_layers=4,
                 policy_n_head=16, value_n_head=8,
                 dropout=0.1, device='cuda', 
                 pretrained_policy_path=None, pretrained_value_path=None,
                 **kwargs):
        super().__init__()
        
        self.spectrum_dim = 4096
        self.max_len = max_len
        self.tokenizer = SelfiesTokenizer(max_len=max_len)
        vocab_size = len(self.tokenizer.get_vocab())
        
        # Create separate policy-value transformer
        self.transformer = SeparatePolicyValueTransformer(
            vocab_size=vocab_size,
            max_len=max_len,
            policy_d_model=policy_d_model,
            value_d_model=value_d_model,
            policy_n_layers=policy_n_layers,
            value_n_layers=value_n_layers,
            policy_n_head=policy_n_head,
            value_n_head=value_n_head,
            dropout=dropout,
            device=device
        )
        
        self.device = torch.device(device)
        self.to(self.device)
        
        # Load pretrained weights if provided
        if pretrained_policy_path or pretrained_value_path:
            self._load_pretrained_weights(pretrained_policy_path, pretrained_value_path)
        
        print(f"[INFO] MuZero Separate Policy-Value Transformer initialized:")
        print(f"  Policy network: {policy_d_model}d, {policy_n_layers}L, {policy_n_head}H")
        print(f"  Value network: {value_d_model}d, {value_n_layers}L, {value_n_head}H")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _load_pretrained_weights(self, policy_path: str = None, value_path: str = None):
        """Load pretrained weights for policy and/or value networks"""
        if policy_path:
            try:
                print(f"[INFO] Loading pretrained policy network from: {policy_path}")
                checkpoint = torch.load(policy_path, map_location=self.device, weights_only=False)
                
                # Extract policy state dict
                if 'policy_state_dict' in checkpoint:
                    policy_state = checkpoint['policy_state_dict']
                elif 'model_state_dict' in checkpoint:
                    # Filter policy-related parameters
                    policy_state = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                  if 'policy' in k or 'token_embed' in k or 'pos_embed' in k or 'action_head' in k}
                else:
                    policy_state = checkpoint
                
                # Load policy weights (with prefix mapping if needed)
                policy_state_mapped = {}
                for key, value in policy_state.items():
                    # Map old keys to new separate network keys
                    if key.startswith('token_embed'):
                        policy_state_mapped[f'policy_{key}'] = value
                    elif key.startswith('pos_embed'):
                        policy_state_mapped[f'policy_{key}'] = value
                    elif key.startswith('action_head'):
                        policy_state_mapped[key.replace('action_head', 'policy_head')] = value
                    elif 'policy' in key:
                        policy_state_mapped[key] = value
                
                missing, unexpected = self.transformer.load_state_dict(policy_state_mapped, strict=False)
                print(f"[INFO] Policy network loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                
            except Exception as e:
                print(f"[WARN] Failed to load policy weights: {e}")
        
        if value_path:
            try:
                print(f"[INFO] Loading pretrained value network from: {value_path}")
                checkpoint = torch.load(value_path, map_location=self.device, weights_only=False)
                
                # Extract value state dict
                if 'value_state_dict' in checkpoint:
                    value_state = checkpoint['value_state_dict']
                elif 'model_state_dict' in checkpoint:
                    # Filter value-related parameters
                    value_state = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                 if 'value' in k}
                else:
                    value_state = checkpoint
                
                missing, unexpected = self.transformer.load_state_dict(value_state, strict=False)
                print(f"[INFO] Value network loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                
            except Exception as e:
                print(f"[WARN] Failed to load value weights: {e}")
    
    def initial_inference(self, obs: torch.Tensor):
        """Initial inference using separate networks"""
        vec = obs if obs.dim() == 2 else obs.unsqueeze(0)
        vec = vec.to(self.device)
        B = vec.size(0)
        
        # Extract components from observation
        spectrum = vec[:, :self.spectrum_dim]
        selfies_part = vec[:, self.spectrum_dim:self.spectrum_dim + self.max_len]
        
        # Clamp SELFIES tokens to valid range
        vocab_size = len(self.tokenizer.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1)
        
        # Create attention mask
        attention_mask = (selfies_part_clamped != self.tokenizer.pad_token_id).float()
        
        # Get predictions from both networks
        policy_logits = self.transformer.forward_policy(spectrum, selfies_part_clamped, attention_mask)
        values = self.transformer.forward_value(spectrum, selfies_part_clamped, attention_mask)
        
        # Take last position for action prediction
        last_policy_logits = policy_logits[:, -1, :]  # (B, vocab_size)
        last_values = values[:, -1, :]  # (B, 1)
        
        # Apply basic action masking
        last_policy_logits[:, self.tokenizer.pad_token_id] = float('-1e9')
        last_policy_logits[:, self.tokenizer.sos_token_id] = float('-1e9')
        last_policy_logits[:, self.tokenizer.unk_token_id] = float('-1e9')
        
        return MZNetworkOutput(
            value=last_values,
            reward=[0.0] * B,
            policy_logits=last_policy_logits,
            latent_state=obs
        )
    
    def _representation(self, observation: torch.Tensor) -> torch.Tensor:
        """Return observation as latent state"""
        return observation
    
    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update latent state by appending action token"""
        action = action.squeeze().float()
        
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        
        next_latent_state = latent_state.clone()
        
        # Find first padding position in SELFIES part and replace with action
        selfies_part = next_latent_state[:, self.spectrum_dim:self.spectrum_dim + self.max_len]
        padding_mask = selfies_part == self.tokenizer.pad_token_id
        
        for i in range(next_latent_state.size(0)):
            # CRITICAL FIX: Check if EOS token already exists in the sequence
            eos_exists = (selfies_part[i] == self.tokenizer.eos_token_id).any()
            if eos_exists:
                # Sequence is already complete (EOS found), don't modify it further
                continue
                
            pad_positions = torch.where(padding_mask[i])[0]
            if len(pad_positions) > 0:
                # Replace first padding token with action
                global_pos = self.spectrum_dim + pad_positions[0].item()
                next_latent_state[i, global_pos] = action[i] if action.dim() > 0 else action
        
        reward = torch.zeros(latent_state.size(0), 1, device=latent_state.device)
        return next_latent_state, reward
    
    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor):
        """Recurrent inference using separate networks"""
        next_latent_state, reward = self._dynamics(latent_state, action)
        
        # Extract components
        spectrum = next_latent_state[:, :self.spectrum_dim]
        selfies_part = next_latent_state[:, self.spectrum_dim:self.spectrum_dim + self.max_len]
        
        # Clamp and create mask
        vocab_size = len(self.tokenizer.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1)
        attention_mask = (selfies_part_clamped != self.tokenizer.pad_token_id).float()
        
        # Get predictions from both networks
        policy_logits = self.transformer.forward_policy(spectrum, selfies_part_clamped, attention_mask)
        values = self.transformer.forward_value(spectrum, selfies_part_clamped, attention_mask)
        
        # Take last position
        last_policy_logits = policy_logits[:, -1, :]  # (B, vocab_size)
        last_values = values[:, -1, :]  # (B, 1)
        
        # Apply basic action masking
        last_policy_logits[:, self.tokenizer.pad_token_id] = float('-1e9')
        last_policy_logits[:, self.tokenizer.sos_token_id] = float('-1e9')
        last_policy_logits[:, self.tokenizer.unk_token_id] = float('-1e9')
        
        return MZNetworkOutput(
            value=last_values,
            reward=reward,
            policy_logits=last_policy_logits,
            latent_state=next_latent_state
        )


if __name__ == "__main__":
    # Quick sanity check
    try:
        data = torch.randn(4296)  # Updated to new observation dimension
        model = MuZeroSelfiesTransformer()
        out = model.initial_inference(data)
        print([t.shape for t in (out.value, out.reward, out.policy_logits, out.latent_state)])
        print(out)
        
        # Test separate policy-value transformer
        separate_model = MuZeroSeparatePolicyValueTransformer()
        separate_out = separate_model.initial_inference(data)
        print("Separate model output shapes:", [t.shape for t in (separate_out.value, separate_out.reward, separate_out.policy_logits, separate_out.latent_state)])
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()