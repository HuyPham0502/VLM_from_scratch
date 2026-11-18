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
            x = x.unsqueeze(0)

        # Shape -> (B, out_channels, num_patch, num_patch) -> (B, out_channels, num_patch * num_patch) -> (B, num_patch * num_patch, out_channels)
        return self.patches(x).flatten(2).transpose(1, 2)


if __name__=="__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    x = torch.rand(3, 336, 336).to(device)
    encoder = VisualEncoder(3, 128).to(device)
    out = encoder(x)
    print("Shape of output: ", out.shape)