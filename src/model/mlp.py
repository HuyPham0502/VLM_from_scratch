import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer Feed-Forward Network with GELU activation and Dropout.

    Commonly used as the dense feed-forward sublayer in standard Transformers
    and as individual expert networks in Mixture-of-Experts (MoE) architectures.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        expansion_factor: int = 4,
    ):
        """Initialize the MLP.

        Args:
            in_features: Dimensionality of the input representation.
            out_features: Dimensionality of the output representation.
            dropout: Dropout probability applied after activation and projection.
            expansion_factor: Multiplier for the hidden intermediate layer (typically 4).
        """
        super().__init__()
        hidden_features = in_features * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
                or (num_tokens, in_features).

        Returns:
            torch.Tensor: Projected tensor of shape (..., out_features).
        """
        return self.net(x)
