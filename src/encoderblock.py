import torch
import torch.nn as nn
import torch.nn.functional as F

from .common.attention import MultiHeadAttention
from .common.visual_encoder import VisualEncoder

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

        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_channels, out_channels),
            nn.Dropout(dropout)
    )

    def forward(self, x: torch.Tensor):
        out = self.encoder(x)  # -> (B, num_patches, d_model)
        residual = out
        out = self.pre_norm(out)
        attn = self.mha(out)  # -> (B, num_patches, d_model)
        attn = attn + residual
        residual = attn
        out = self.post_norm(attn)
        mlp = self.mlp(out)     
        out = mlp + residual
        return out 
