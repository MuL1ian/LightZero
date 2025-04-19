import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import selfies as sf
from tokenizers import Tokenizer, processors, models
from tokenizers.implementations import BaseTokenizer

# -----------------------------------------------------------------------------
# Special‑token constants
# -----------------------------------------------------------------------------
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# -----------------------------------------------------------------------------
# SelfiesTokenizer
# -----------------------------------------------------------------------------
class SpecialTokensBaseTokenizer(BaseTokenizer):
    """Wrap any HuggingFace tokenizers.Tokenizer with special token handling."""

    def __init__(self, tokenizer: Tokenizer, max_len: int):
        super().__init__(tokenizer)

        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        self.max_length = max_len

        self.add_special_tokens(
            [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        )

        self.pad_token_id = self.token_to_id(self.pad_token)
        self.sos_token_id = self.token_to_id(self.sos_token)
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.unk_token_id = self.token_to_id(self.unk_token)

        # Enable automatic pad / truncation
        self.enable_padding(
            direction="right",
            pad_token=self.pad_token,
            pad_id=self.pad_token_id,
            length=max_len,
        )
        self.enable_truncation(max_len)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.sos_token} $A {self.eos_token}",
            pair=f"{self.sos_token} $A {self.eos_token} {self.sos_token} $B {self.eos_token}",
            special_tokens=[
                (self.sos_token, self.sos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )


class SelfiesTokenizer(SpecialTokensBaseTokenizer):

    def __init__(self, max_len: int):
        alphabet = list(sorted(sf.get_semantic_robust_alphabet()))
        vocab = {symbol: i for i, symbol in enumerate(alphabet)}
        vocab[UNK_TOKEN] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
        super().__init__(tokenizer, max_len=max_len)

    # ---------------------------------------------------------------------
    # encode / decode of the selfies
    # ---------------------------------------------------------------------
    def encode_selfies(self, selfies_str: str, add_special_tokens: bool = True) -> List[int]:
        selfies_tokens = list(sf.split_selfies(selfies_str))
        return super().encode(
            selfies_tokens, is_pretokenized=True, add_special_tokens=add_special_tokens
        ).ids

    def decode_to_selfies(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        text = super().decode(token_ids, skip_special_tokens=skip_special_tokens)
        return self._decode_wordlevel_str_to_selfies(text)

    def _decode_wordlevel_str_to_selfies(self, text: str) -> str:
        # WordLevel joins by space – simply remove them
        tokens = text.strip().split()
        return "".join(tokens)



# -----------------------------------------------------------------------------
# 2. Utility helpers
# -----------------------------------------------------------------------------

def pad_to_maxlen(ids: List[int], max_len: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(ids) > max_len:
        raise ValueError(f"Sequence too long: {len(ids)} > {max_len}")
    padded = ids + [pad_id] * (max_len - len(ids))
    mask = [1] * len(ids) + [0] * (max_len - len(ids))
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


# Collate function for DataLoader ------------------------------------------------
class Collator:
    def __init__(self, tokenizer: SelfiesTokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Tuple[torch.Tensor, str]]):
        spectra, selfies_list = zip(*batch)  # spectra: (B,4096)
        spectra = torch.stack(spectra)

        input_ids, attn_mask, target_ids = [], [], []
        for s in selfies_list:
            ids = self.tok.encode_selfies(s)  # [SOS] ... [EOS]
            # Teacher‑forcing shift
            inp = ids[:-1]
            tgt = ids[1:]
            inp_pad, mask = pad_to_maxlen(inp, self.max_len, self.tok.pad_token_id)
            tgt_pad, _ = pad_to_maxlen(tgt, self.max_len, self.tok.pad_token_id)
            input_ids.append(inp_pad)
            attn_mask.append(mask)
            target_ids.append(tgt_pad)

        return (
            spectra,
            torch.stack(input_ids),
            torch.stack(attn_mask),
            torch.stack(target_ids),
        )

# -----------------------------------------------------------------------------
# 3. Encoder‑Decoder Transformer
# -----------------------------------------------------------------------------
class MassSelfiesED(nn.Module):
    """Encoder Decoder Transformer producing (action_logits, value)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_enc: int = 4,
        n_dec: int = 6,
        n_head: int = 8,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        
        # -------- Encoder -------------
        self.spec_proj = nn.Sequential(
            nn.Linear(4096, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc)

        # -------- Decoder -------------
        self.token_embed = nn.Embedding(vocab_size, d_model)

        self.pos_embed = nn.Embedding(1024, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)

        # -------- Heads ---------------
        self.action_head = nn.Linear(d_model, vocab_size, bias=False)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        
        self.to(self.device)

    # ----------------------------------------------------------------------
    def forward(
        self,
        spectrum_embed: torch.Tensor,  # (B,4096)
        tgt_tokens: torch.Tensor,      # (B,T)
        tgt_mask: torch.Tensor,        # (B,T) 1=real,0=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = tgt_tokens.shape

        # ---- Encoder ----
        enc_in = self.spec_proj(spectrum_embed).unsqueeze(1)  # (B,1,d)
        mem = self.encoder(enc_in)  # (B,1,d)

        # ---- Decoder ----
        pos_ids = torch.arange(T, device=tgt_tokens.device).unsqueeze(0)
        dec_in = self.token_embed(tgt_tokens) + self.pos_embed(pos_ids)

        key_padding = tgt_mask == 0  # pad→True
        

        causal_mask = self._generate_square_subsequent_mask(T).to(tgt_tokens.device)
        
        dec_out = self.decoder(
            tgt=dec_in,
            memory=mem,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding,
            memory_key_padding_mask=None,
        )

        dec_last = dec_out[:, -1, :]  # (B,d)
        print('last token embedding:')
        print("dec_last", dec_last[0, :5].tolist())
        print("dec_last", dec_last.shape)
        logits = self.action_head(dec_last)  # (B,|V|)
        value   = self.value_head(dec_last).squeeze(-1)  # (B)
        return logits, value
        
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# -----------------------------------------------------------------------------
# 4. Simple Trainer
# -----------------------------------------------------------------------------
# class Trainer:
#     def __init__(self, model: MassSelfiesED, tokenizer: SelfiesTokenizer, lr: float = 3e-4, device: str = "cpu"):
#         self.model = model
#         self.tok = tokenizer
#         self.device = torch.device(device)
#         self.optim = torch.optim.AdamW(model.parameters(), lr=lr)
#         self.criterion = nn.CrossEntropyLoss(ignore_index=self.tok.pad_token_id)

#     def train_epoch(self, loader: DataLoader):
#         self.model.train()
#         total, n = 0.0, 0
#         for spec, inp_ids, mask, tgt_ids in loader:
#             spec = spec.to(self.device)
#             inp_ids = inp_ids.to(self.device)
#             mask = mask.to(self.device)
#             tgt_ids = tgt_ids.to(self.device)
#             logits, _ = self.model(spec, inp_ids, mask)
#             loss = self.criterion(logits, tgt_ids[:, -1])  # 只对最后 token 计算 CE
#             self.optim.zero_grad(); loss.backward(); self.optim.step()
#             total += loss.item() * spec.size(0); n += spec.size(0)
#         return total / n

# -----------------------------------------------------------------------------
# 5. Greedy decoding helper
# -----------------------------------------------------------------------------

@torch.no_grad()
def step_prediction(model: MassSelfiesED, tokenizer: SelfiesTokenizer, spectrum_vec: torch.Tensor, prefix_selfies: str = None, device: str = None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    
    spec = spectrum_vec.unsqueeze(0).to(device)
    
    if prefix_selfies:
        prefix_ids = [
            tokenizer.token_to_id(sym)
            for sym in sf.split_selfies(prefix_selfies)
        ]
        current_ids = [tokenizer.sos_token_id] + prefix_ids
    else:
        current_ids = [tokenizer.sos_token_id]
    
    max_len = tokenizer.max_length if hasattr(tokenizer, 'max_length') else 64
    ids, mask = pad_to_maxlen(current_ids, max_len, tokenizer.pad_token_id)
    ids = ids.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    

    logits, value = model(spec, ids, mask)
    
    return {
        'logits': logits.squeeze(0), 
        'value': value.item(),
        'probs': F.softmax(logits, dim=-1).squeeze(0).cpu().numpy(),
        'current_prefix': current_ids
    }


# -----------------------------------------------------------------------------
# 6. Dummy dataset (replace with real one)
# -----------------------------------------------------------------------------
data = torch.load("debug_spectrum_embeds.pt")

observartion = {
    "embeds": data["embeds"][1],
    "partial_formulas": '[C][O][H][Branch1]'
}

class SingleObsDataset(Dataset):
    def __init__(self, obs: Dict[str, Any]):
        self.spec = torch.tensor(obs["embeds"], dtype=torch.float32)
        self.selfies = obs["partial_formulas"]      #

    def __len__(self):   
        return 1

    def __getitem__(self, idx):
        return self.spec, self.selfies  
    

# -----------------------------------------------------------------------------
# 7. Main entry
# -----------------------------------------------------------------------------
def main(args):
    # 固定随机种子（可选）
    torch.manual_seed(42)

    # 选择设备
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # 1. 初始化 Tokenizer 和 Model
    tok = SelfiesTokenizer(max_len=args.max_len)

    vocab_size = len(tok._tokenizer.get_vocab())
    model = MassSelfiesED(vocab_size=vocab_size, d_model=args.d_model, device=device)

    model.eval()

    # 2. 读取示例数据
    data = torch.load("debug_spectrum_embeds.pt", map_location="cpu")
    spec_vec = torch.tensor(data["embeds"][1], dtype=torch.float32).to(device)
    prefix   = '[C][O][H][Branch1]'

    # 3. 单步预测下一个 token
    out = step_prediction(model, tok, spec_vec, prefix_selfies=prefix, device=device)
    next_id  = out['logits'].argmax().item()
    next_tok = tok.decode_to_selfies([next_id], skip_special_tokens=True)

    print(f"prefix: {prefix}")
    print(f"spec_vec: {spec_vec.shape}")
    print(f"\nSELFIES '{prefix}' next token:")
    print(f"  Value: {out['value']:.4f}")
    print(f"  predicted token: {next_tok} (ID: {next_id})")

    prefix = prefix + next_tok

    out = step_prediction(model, tok, spec_vec, prefix_selfies=prefix, device=device)
    next_id  = out['logits'].argmax().item()
    next_tok = tok.decode_to_selfies([next_id], skip_special_tokens=True)

    print(f"prefix: {prefix}")
    print(f"spec_vec: {spec_vec.shape}")
    print(f"\nSELFIES '{prefix}' next token:")
    print(f"  Value: {out['value']:.4f}")
    print(f"  predicted token: {next_tok} (ID: {next_id})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    main(args)
