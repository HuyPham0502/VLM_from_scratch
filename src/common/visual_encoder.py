import torch 
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 patch_size: int = 16):
        super().__init__()
        self.patches = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor):
        # Check whether input data does not have shape (B, C, H, W)
        if len(x.shape) != 4:   
            x = x.unsquezze(0)

        return self.patches(x).flatten(2).transpose(1, 2)

    