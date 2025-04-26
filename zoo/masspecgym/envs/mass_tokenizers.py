import typing as T
import selfies as sf
from tokenizers import Tokenizer, processors, models
from tokenizers.implementations import BaseTokenizer
import torch
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

class SpecialTokensBaseTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_len: T.Optional[int] = None,
    ):
        super().__init__(tokenizer)

        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        self.max_length = max_len
        self._tokenizer.add_tokens([ "<REMOVE>", "<END>" ])

        self.add_special_tokens([self.pad_token, self.sos_token, self.eos_token, self.unk_token])

        self.pad_token_id = self.token_to_id(self.pad_token)
        self.sos_token_id = self.token_to_id(self.sos_token)
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.unk_token_id = self.token_to_id(self.unk_token)
        self.remove_token_id = self.token_to_id("<REMOVE>")
        self.end_token_id    = self.token_to_id("<END>")

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
    def __init__(
        self,
        selfies_train: T.Optional[T.Union[str, T.List[str]]] = None,
        **kwargs
    ):
        alphabet = list(sorted(sf.get_semantic_robust_alphabet()))
        vocab = {symbol: i for i, symbol in enumerate(alphabet)}
        vocab[UNK_TOKEN] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))

        super().__init__(tokenizer, **kwargs)


    def encode_selfies(self, selfies_str: str, add_special_tokens: bool = True) -> T.List[int]:
        selfies_tokens = list(sf.split_selfies(selfies_str))
        return super().encode(
            selfies_tokens, is_pretokenized=True, add_special_tokens=add_special_tokens
        ).ids

    def decode_to_selfies(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        text = super().decode(token_ids, skip_special_tokens=skip_special_tokens)
        return self._decode_wordlevel_str_to_selfies(text)

    def _decode_wordlevel_str_to_selfies(self, text: str) -> str:
        """Converts a WordLevel string back to a SELFIES string."""
        return text.replace(" ", "")


if __name__ == "__main__":
    tokenizer = SelfiesTokenizer(max_len=100)

    # 输入一个SELFIES字符串
    s = "[C][C][C][H][H]"

    print(tokenizer.token_to_id("[C]"))
    print(s)
    # 编码为 token ID 列表
    token_ids = tokenizer.encode_selfies(s)
    print(torch.tensor(token_ids, dtype=torch.long))
    print(torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).shape)
    print(torch.tensor(token_ids, dtype=torch.long).shape)
    # print("Token IDs:", token_ids)

    # 解码回 SELFIES
    # recovered_selfies = tokenizer.decode_to_selfies(token_ids)
    # print("Recovered SELFIES:", recovered_selfies)
