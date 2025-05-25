import torch
import torch.nn as nn
import torch.nn.functional as F
import selfies as sf
from tokenizers import Tokenizer, processors, models
from tokenizers.implementations import BaseTokenizer
from typing import List, Tuple, Optional, Union
from ding.utils import MODEL_REGISTRY, SequenceType
from lzero.model.common import MZNetworkOutput
from zoo.masspecgym.envs.massgymenv import MassGymEnv
from transformers import BertTokenizer
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
        # Fallback actions list for testing
        return ['[C]', '[H]', '[O]', '[N]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]', 
                '[=C]', '[=N]', '[=O]', '[=S]', '[#C]', '[#N]', '[Ring1]', '[Ring2]', '[Ring3]',
                '[Branch1]', '[Branch2]', '[Branch3]', '<END>', '<REMOVE>'] + ['[UNK]'] * 47

# Get actions list (lazy loading)
actions_list = None

# -----------------------------------------------------------------------------
# Special-token constants
# -----------------------------------------------------------------------------
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

class SpecialTokensBaseTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: Tokenizer, max_len: int):
        super().__init__(tokenizer)
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        self.max_length = max_len
        # Ensure REMOVE and END are in vocab
        self._tokenizer.add_tokens(["<REMOVE>", "<END>"])
        # Add special tokens
        self.add_special_tokens([self.pad_token, self.sos_token, self.eos_token, self.unk_token])
        # Record token IDs
        self.pad_token_id = self.token_to_id(self.pad_token)
        self.sos_token_id = self.token_to_id(self.sos_token)
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.unk_token_id = self.token_to_id(self.unk_token)
        self.remove_token_id = self.token_to_id("<REMOVE>")
        self.end_token_id    = self.token_to_id("<END>")
        # Enable padding and truncation
        self.enable_padding(direction="right", pad_token=self.pad_token, pad_id=self.pad_token_id, length=max_len)
        self.enable_truncation(max_len)
        # Post processor for adding SOS/EOS
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.sos_token} $A {self.eos_token}",
            pair=f"{self.sos_token} $A {self.eos_token} {self.sos_token} $B {self.eos_token}",
            special_tokens=[(self.sos_token, self.sos_token_id), (self.eos_token, self.eos_token_id)],
        )

class SelfiesTokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, max_len: int):
        alphabet = list(sorted(sf.get_semantic_robust_alphabet()))
        vocab = {symbol: i for i, symbol in enumerate(alphabet)}
        vocab[UNK_TOKEN] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
        super().__init__(tokenizer, max_len)

    def encode_selfies(self, selfies_str: str, add_special_tokens: bool = True) -> List[int]:
        tokens = list(sf.split_selfies(selfies_str))
        return super().encode(tokens, is_pretokenized=True, add_special_tokens=add_special_tokens).ids

    def decode_to_selfies(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        text = super().decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text.replace(" ", "")

# -----------------------------------------------------------------------------
# Utility: pad sequence to max_len
# -----------------------------------------------------------------------------
def pad_to_maxlen(ids: List[int], max_len: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(ids) > max_len:
        raise ValueError(f"Sequence too long: {len(ids)} > {max_len}")
    padded = ids + [pad_id] * (max_len - len(ids))
    mask   = [1] * len(ids) + [0] * (max_len - len(ids))
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

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

        # Encoder for spectrum
        self.spec_proj = nn.Sequential(
            nn.Linear(self.spectrum_dim, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
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

        # Encoder
        enc_feat = self.spec_proj(spectrum_embed)       # (B,d)
        mem      = self.encoder(enc_feat.unsqueeze(1))  # (B,1,d)

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
    """Single-step greedy prediction"""
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
    # forward
    logits, value = model(vec, ids, mask)
    next_id = torch.argmax(logits, dim=-1)[0].item()
    return {
        'logits': logits.squeeze(0),
        'value':  value, 
        'probs':  F.softmax(logits, dim=-1).squeeze(0).cpu().numpy(),
        'current_prefix': ids[0].tolist(),
        'next_token_id': next_id,
    }

# -----------------------------------------------------------------------------
# MuZero transformer wrapper
# -----------------------------------------------------------------------------
@MODEL_REGISTRY.register('MuZeroSelfiesTransformer')
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
        pol = pred['logits']
        rew = [0.0] * B

        return MZNetworkOutput(value=val, reward=rew, policy_logits=pol, latent_state=obs)

    def _representation(self, observation: torch.Tensor) -> torch.Tensor:
        """Simply return the prefix as the latent state representation"""
        return observation

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update latent state by replacing last padding token with action"""
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
        # action is a tensor of shape (B, 1)
        # check if any of the actions is the end token
        if (action == self.tok.end_token_id).any():
            print("end token found in recurrent inference")

        # For base class, extract only spectrum and SELFIES parts for transformer
        spectrum = next_latent_state[:, :self.spectrum_dim]
        selfies_part = next_latent_state[:, self.spectrum_dim:self.spectrum_dim + self.tok.max_length]
        
        # Clamp SELFIES token IDs to valid vocabulary range
        vocab_size = len(self.tok.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1)
        
        mask = (selfies_part_clamped != self.tok.pad_token_id) & (selfies_part_clamped != self.tok.end_token_id)

        logits, value = self.transformer(spectrum, selfies_part_clamped, mask)
        value = value.unsqueeze(-1)

        return MZNetworkOutput(value=value, reward=reward, policy_logits=logits, latent_state=next_latent_state)

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
@MODEL_REGISTRY.register('MuZeroSelfiesTransformerEnhanced')
class MuZeroSelfiesTransformerEnhanced(MuZeroSelfiesTransformer):
    def __init__(self, observation_shape=4246, max_len=100,
                 d_model=512, n_enc=4, n_dec=6, n_head=8,
                 dropout=0.1, device='cuda', target_formula=None,  # Deprecated: formula extracted from observations
                 formula_max_len=50, **kwargs):
        super().__init__(observation_shape, max_len, d_model, n_enc, n_dec, 
                        n_head, dropout, device, **kwargs)
        
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
        
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
            formula_str = self.bert_tokenizer.decode(token_ids, skip_special_tokens=True)
            formula_strings.append(formula_str.strip())
            # except:
            #     formula_strings.append("")
                
        return formula_strings
    
    def _apply_formula_mask(self, logits: torch.Tensor, current_selfies_list: List[str], formula_list: List[str]) -> torch.Tensor:
        """Apply formula-based action masking to logits"""
        # Check for NaN in input logits first
        if torch.isnan(logits).any():
            print(f"[WARN] NaN detected in input logits before masking")
            logits = torch.where(torch.isnan(logits), torch.tensor(0.0, device=logits.device), logits)
        
        if not formula_list or not any(formula_list):
            return logits
            
        masked_logits = logits.clone()
        batch_size = logits.size(0)
        
        for batch_idx in range(batch_size):
            current_selfies = current_selfies_list[batch_idx]
            target_formula = formula_list[batch_idx] if batch_idx < len(formula_list) else ""
            
            if not target_formula:
                continue
                
            try:
                # Get action mask using the utility function
                action_mask = self.get_action_mask_from_selfies_string(
                    formula=target_formula,
                    current_selfies=current_selfies,
                    actions_list=actions_list,
                    atom_tokens=self.atom_tokens,
                    bonded_atom_tokens=self.bonded_atom_tokens,
                    formula_masking=True,
                    end_token="<END>",
                    remove_token="<REMOVE>",
                    special_tokens=[]
                )
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
                
                # Use a safer masking approach
                # Find the minimum logit value and subtract a reasonable amount
                min_logit = masked_logits[batch_idx].min().item()
                mask_value = min_logit - 10.0  # Subtract 10 from minimum to ensure masked actions have very low probability
                
                masked_logits[batch_idx][~mask_tensor] = mask_value
                
            except Exception as e:
                print(f"[WARN] Failed to apply formula mask for batch {batch_idx}: {e}")
                # Continue without masking for this batch
                
        # Final check for NaN values
        if torch.isnan(masked_logits).any():
            print(f"[WARN] NaN detected in final masked_logits, replacing with safe values")
            masked_logits = torch.where(torch.isnan(masked_logits), torch.tensor(-10.0, device=logits.device), masked_logits)
                
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
                    special_tokens=[]
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
        
        # Print completion info if any molecule is complete
        if any(completion_status):
            complete_indices = [i for i, complete in enumerate(completion_status) if complete]
            print(f"Molecules complete at indices {complete_indices}")
            for idx in complete_indices:
                print(f"  Batch {idx}: SELFIES='{current_selfies_list[idx]}', Formula='{formula_list[idx]}'")
        
        # Get transformer output - only use spectrum and SELFIES parts for transformer
        spectrum = next_latent_state[:, :self.spectrum_dim]
        selfies_ids = next_latent_state[:, self.selfies_start_idx:self.formula_start_idx]
        
        # Clamp SELFIES token IDs to valid vocabulary range
        vocab_size = len(self.tok.get_vocab())
        selfies_ids_clamped = torch.clamp(selfies_ids.long(), 0, vocab_size - 1)
        
        mask = (selfies_ids_clamped != self.tok.pad_token_id) & (selfies_ids_clamped != self.tok.end_token_id)

        logits, value = self.transformer(spectrum, selfies_ids_clamped, mask)
        
        # Apply formula-based action masking
        masked_logits = self._apply_formula_mask(logits, current_selfies_list, formula_list)
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
        # if formula_list and formula_list[0]:
        #     print(f"Using target formula from observation: '{formula_list[0]}'")

        # Get initial prediction - only use spectrum and SELFIES parts
        spectrum = vec[:, :self.spectrum_dim]
        selfies_part = vec[:, self.selfies_start_idx:self.formula_start_idx]
        
        # Clamp SELFIES token IDs to valid vocabulary range to handle random data
        vocab_size = len(self.tok.get_vocab())
        selfies_part_clamped = torch.clamp(selfies_part.long(), 0, vocab_size - 1)
        
        # Create mask for transformer
        mask = (selfies_part_clamped != self.tok.pad_token_id) & (selfies_part_clamped != self.tok.end_token_id)
        
        # Use transformer directly for batch processing
        logits, value = self.transformer(spectrum, selfies_part_clamped, mask)
        
        # Extract initial SELFIES (should be empty)
        current_selfies_list = self._extract_selfies_from_latent_state(vec)
        
        # Apply formula masking to logits
        masked_logits = self._apply_formula_mask(logits, current_selfies_list, formula_list)
        
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