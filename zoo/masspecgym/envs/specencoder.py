import torch
import torch.nn as nn
from torch_geometric.nn import MLP

class SpectrumEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,           
        d_model: int,            
        nhead: int,             
        num_encoder_layers: int,  
        dropout: float = 0.1, 
        pre_norm: bool = False,  
        use_formula: bool = False,  
        *args,
        **kwargs
    ):
        super().__init__()
        

        self.src_encoder = nn.Linear(input_dim, d_model)
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            norm_first=pre_norm,
            batch_first=True  
        )
        

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.d_model = d_model
        self.use_formula = use_formula
        
        if self.use_formula:
            self.formula_mlp = MLP(
                in_channels=input_dim,  
                hidden_channels=d_model,
                out_channels=d_model,
                num_layers=1,
                dropout=dropout,
                norm=None
            )

    def forward(self, batch):
        """
        Returns:
            encoded: (batch_size, seq_len, d_model) 编码后的向量
        """
        spec = batch["spec"]  # (batch_size, seq_len, input_dim)
        
        src_key_padding_mask = self.generate_padding_mask(spec)
        
        src = self.src_encoder(spec)  # (batch_size, seq_len, d_model)
        src = src * (self.d_model ** 0.5) 
        
        if self.use_formula and "formula" in batch:
            formula_emb = self.formula_mlp(batch["formula"])  # (batch_size, d_model)
            src = src + formula_emb.unsqueeze(1) 
            
        encoded = self.transformer_encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return encoded

    def generate_padding_mask(self, spec):
        return spec.sum(-1) == 0  # (batch_size, seq_len)

    def get_latent_vector(self, batch):
        encoded = self.forward(batch)  # (batch_size, seq_len, d_model)
        
        latent = encoded.mean(dim=1)  # (batch_size, d_model)
        
        return latent


if __name__ == "__main__":
    # 初始化模型
    encoder = SpectrumEncoder(
        input_dim=2,          # 2D输入
        d_model=256,         # 编码维度
        nhead=8,             # 8个注意力头
        num_encoder_layers=6, # 6层编码器
        dropout=0.1,
        pre_norm=True,
        use_formula=False    # 是否使用化学式信息
    )

    # 准备输入数据
    batch = {
        "spec": torch.randn(32, 100, 2)  # (batch_size=32, seq_len=100, input_dim=2)
    }

    # 获取编码结果
    encoded = encoder(batch)  # (32, 100, 256)

    # 获取latent vector
    latent = encoder.get_latent_vector(batch)  # (32, 256)
    print(latent)