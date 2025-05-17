import torch
import torch.nn as nn
import torch.nn.functional as F
import selfies as sf
from tokenizers import Tokenizer, processors, models
from tokenizers.implementations import BaseTokenizer
from typing import List, Tuple, Optional, Union
from ding.utils import MODEL_REGISTRY, SequenceType
from .common import MZNetworkOutput
from zoo.masspecgym.envs.massgymenv import MassGymEnv

# -----------------------------------------------------------------------------
# Env for get the action list 
# -----------------------------------------------------------------------------
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
actions_list = env.actions_list  # length 71

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

    # rebuild token sequence
    ids_list = [[tokenizer.sos_token_id] + row.tolist() for row in prefix_ids]
    # pad and mask
    max_len = tokenizer.max_length
    ids_batch = []
    mask_batch = []
    for seq in ids_list:
        ids_pad, m = pad_to_maxlen(seq, max_len, tokenizer.pad_token_id)
        ids_batch.append(ids_pad)
        mask_batch.append(m)
    ids = torch.stack(ids_batch,  dim=0).to(dev)  # (B,T)
    mask = torch.stack(mask_batch, dim=0).to(dev)  # (B,T)
    # forward
    logits, value = model(vec, ids, mask)
    next_id = torch.argmax(logits, dim=-1)[0].item()
    return {
        'logits': logits.squeeze(0),
        'value':  value, 
        'probs':  F.softmax(logits, dim=-1).squeeze(0).cpu().numpy(),
        'current_prefix': ids_list[0],
        'next_token_id': next_id,
    }

# -----------------------------------------------------------------------------
# MuZero transformer wrapper
# -----------------------------------------------------------------------------
@MODEL_REGISTRY.register('MuZeroSelfiesTransformer')
class MuZeroSelfiesTransformer(nn.Module):
    def __init__(self, observation_shape=4096, max_len=128,
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

        pred = step_prediction(self.transformer, self.tok, vec, device=self.device)
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

        # clone for help gradient computation
        next_latent_state = latent_state.clone()

        # print("latent_state shape: ", latent_state.shape)
        padding_mask = next_latent_state == self.tok.pad_token_id
        last_padding_token_index = torch.sum(padding_mask, dim=1) - 1

        next_latent_state[torch.arange(next_latent_state.size(0)), last_padding_token_index] = action

        reward = torch.zeros(latent_state.size(0), 1, device=latent_state.device)

        return next_latent_state, reward

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor):
        """Perform recurrent inference step"""
        next_latent_state, reward = self._dynamics(latent_state, action)
        
        spectrum = next_latent_state[:, :self.spectrum_dim]
        ids = next_latent_state[:, self.spectrum_dim:]
        mask = ids != self.tok.pad_token_id

        logits, value = self.transformer(spectrum, ids, mask)
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

if __name__ == "__main__":
    # quick sanity check
    data   = torch.randn(4196)
    model  = MuZeroSelfiesTransformer()
    out = model.initial_inference(data)
    print([t.shape for t in (out.value, out.reward, out.policy_logits, out.latent_state)])
    print(out)