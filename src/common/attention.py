import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1, is_decoded=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible of n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_model // self.n_heads

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask', torch.tril(torch.ones(block_size, block_size))
        )

        self.is_decoded = is_decoded

    def forward(self, x):
        B, T, C = x.shape
        # Project to q, k, v matrices and split them from (B, T, 3*C) to 3*(B ,T, C )
        q, k, v = self.qkv(x).split(C, dim=2)
        # Reshape them from (B, T, C) to (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if self.is_decoded:
            attn = attn.masked_fill(self.mask[:T, :T]==0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # Compute values
        attn = attn @ v
        # Merge n_heads into one
        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)