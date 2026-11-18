import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 4 * in_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * in_features, out_features),
            nn.Dropout(dropout)
    )

    def forward(self, x):
        return self.net(x)