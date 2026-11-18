import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 256, device: str = "cuda") -> torch.Tensor:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedder
        self.embedder = nn.Embedding(vocab_size, d_model, device=device)
        