import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .mlp import MLP


class EncoderBlock(nn.Module):
    """Transformer Encoder Block following Pre-LayerNorm formulation.

    Structure:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        """Initialize EncoderBlock.

        Args:
            d_model: Hidden dimensionality of the token representations.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.pre_norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            is_causal=False,
        )

        self.pre_norm_mlp = nn.LayerNorm(d_model)
        self.mlp = MLP(
            in_features=d_model,
            out_features=d_model,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the EncoderBlock.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional attention mask tensor.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Pre-Norm Multi-Head Self-Attention with residual connection
        normed_attn = self.pre_norm_attn(x)
        x = x + self.attn(normed_attn, attention_mask=attention_mask)

        # Pre-Norm MLP Feed-Forward with residual connection
        normed_mlp = self.pre_norm_mlp(x)
        x = x + self.mlp(normed_mlp)

        return x
