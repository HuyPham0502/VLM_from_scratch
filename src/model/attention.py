import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Attention module.

    Allows the model to jointly attend to information from different representation
    subspaces at different sequence positions. Supports both bidirectional attention
    (for vision encoders) and causal attention (for autoregressive decoders).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        is_causal: bool = False,
    ):
        """Initialize MultiHeadAttention.

        Args:
            d_model: Total hidden dimensionality of the model.
            n_heads: Number of parallel attention heads (must divide d_model).
            dropout: Dropout probability applied to attention weights.
            is_causal: If True, applies a lower-triangular causal mask.
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.is_causal = is_causal

        # Joint linear projection for Query, Key, and Value
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional mask tensor of shape (batch_size, 1, seq_len, seq_len)
                or (batch_size, seq_len) where 0 indicates masked positions.

        Returns:
            torch.Tensor: Attended output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.shape

        # Step 1: Project input to Query, Key, Value and split into 3 tensors of shape (B, T, C)
        q, k, v = self.qkv_proj(x).split(C, dim=2)

        # Step 2: Reshape for multi-head attention -> (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Compute scaled dot-product attention scores -> (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Step 4: Apply causal mask if required
        if self.is_causal:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Step 5: Apply optional external attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Shape: (B, 1, 1, T)
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 6: Softmax normalization, dropout, and matrix multiplication with Value
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v  # Shape: (B, n_heads, T, d_k)

        # Step 7: Concatenate heads and project output -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
