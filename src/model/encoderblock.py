import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MultiHeadAttention, VisualEncoder, MLP

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, block_size=16, dropout=0.1):
        super().__init__()
        self.encoder = VisualEncoder(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.pre_norm = nn.LayerNorm(out_channels)
        self.mha = MultiHeadAttention(
            d_model=out_channels,
            n_heads=n_heads,
            block_size=block_size
        )
        self.post_norm = nn.LayerNorm(out_channels)
        self.mlp = MLP(out_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor):
        out = self.encoder(x)  # -> (B, num_patches**2, d_model)
        residual = out
        out = self.pre_norm(out)
        attn = self.mha(out)  # -> (B, num_patches**2, d_model)
        attn = attn + residual
        residual = attn
        out = self.post_norm(attn)
        mlp = self.mlp(out)     # -> (B, num_patches**2, d_model)
        out = mlp + residual
        return out 

if __name__=="__main__":
    B, H, W, C = 32, 224, 224, 3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    x = torch.rand(B, C, H, W).to(device)
    encoder = EncoderBlock(C, 8, 4).to(device)
    out = encoder(x)
    print("Shape of output:", out.shape)
