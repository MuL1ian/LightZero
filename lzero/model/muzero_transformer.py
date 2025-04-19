from typing import Optional, Tuple

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY, SequenceType
from .mass_selfies_ed import MassSelfiesED, SelfiesTokenizer 
from .common import MZNetworkOutput, RepresentationNetworkMLP, PredictionNetworkMLP, MLP_V2
from .utils import renormalize, get_params_mean, get_dynamic_mean, get_reward_mean



@MODEL_REGISTRY.register('MuZeroSelfiesTransformer')
class MuZeroSelfiesTransformer(nn.Module):
    def __init__(
        self,
        tokenizer: SelfiesTokenizer,
        model_cfg: dict,
        max_len: int = 128,
        categorical_distribution: bool = False,        # value/reward 均为实数 → False
        state_norm: bool = False,       
    ):
        super().__init__()
        
        super().__init__()
        self.tok = tokenizer
        self.transformer = MassSelfiesED(
            vocab_size=len(self.tok._tokenizer.get_vocab()), **model_cfg
        )
        self.max_len = max_len
        self.action_space_size = len(self.tok._tokenizer.get_vocab())
        self.categorical_distribution = categorical_distribution
        self.state_norm = state_norm


    def initial_inference(self, obs: torch.Tensor) -> MZNetworkOutput:
        input_ids, attn_mask = self._pad_batch_prefix(obs['prefix'])
        logits, value = self.transformer(
            spectrum_embed=obs['spectrum'],
            tgt_tokens=input_ids,
            tgt_mask=attn_mask,
        )
        # latent_state 就保存当前 prefix 的 token 序列（后续复用）
        latent_state = input_ids
        return MZNetworkOutput(
            value,                             # (B,)
            torch.zeros_like(value),           # reward_prefix=0
            logits,                            # policy_logits
            latent_state,                      # 保存 prefix
        )

    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor) -> MZNetworkOutput:
        new_prefix = torch.cat(
            [latent_state, action.unsqueeze(1)], dim=1
        )
        # 2. 重新计算 logits / value
        input_ids, attn_mask = self._pad_batch_prefix(new_prefix)
        spectrum = self.cached_spectrum           # 上一步放进缓存（见下）
        logits, value = self.transformer(spectrum, input_ids, attn_mask)

        return MZNetworkOutput(
            value,
            torch.zeros_like(value),              # 每步 reward=0
            logits,
            new_prefix,                           # 更新 latent_state
        )

    def _pad_batch_prefix(self, prefix_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        prefix_ids : (B,<=T) LongTensor
        回传 (B,max_len) 的 padded input_ids 与 attention_mask
        """
        B, L = prefix_ids.shape
        pad_len = self.max_len - L
        if pad_len < 0:
            raise ValueError(f"prefix length {L} exceeds max_len {self.max_len}")
        pad = prefix_ids.new_full((B, pad_len), self.tok.pad_token_id)
        input_ids = torch.cat([prefix_ids, pad], dim=1)
        mask      = torch.cat([torch.ones(B, L, dtype=torch.bool, device=prefix_ids.device),
                               torch.zeros(B, pad_len, dtype=torch.bool, device=prefix_ids.device)], dim=1)
        return input_ids, mask