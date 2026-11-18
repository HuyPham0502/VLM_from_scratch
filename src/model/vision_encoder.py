import torch
import torch.nn as nn
from .encoder_block import EncoderBlock


class VisionEncoder(nn.Module):
    """Vision Transformer Encoder that converts raw images into visual tokens.

    Pipeline:
        1. 2D Patchify via Strided Convolution (Conv2d).
        2. Flatten spatial grid into sequence of patch tokens.
        3. Add learnable Positional Embeddings.
        4. Pass through N stacked Transformer EncoderBlocks.
        5. Final LayerNorm normalization.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize VisionEncoder.

        Args:
            image_size: Resolution of square input image (H = W).
            patch_size: Resolution of each square patch (P = P).
            in_channels: Number of image channels (typically 3 for RGB).
            d_model: Dimensionality of the visual token embeddings.
            num_layers: Number of stacked Transformer EncoderBlocks.
            num_heads: Number of attention heads in each block.
            dropout: Dropout probability.
        """
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # E.g., (224 // 16)**2 = 196
        self.d_model = d_model

        # Step 1: Patchify and linearly project using strided 2D convolution
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Step 2: Learnable 1D spatial positional embedding for each patch position
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Step 3: Stack of Transformer Encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                n_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Step 4: Final LayerNorm normalization
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual token representations from image tensor.

        Args:
            x: Input image tensor of shape (batch_size, 3, image_size, image_size)
               or (3, image_size, image_size).

        Returns:
            torch.Tensor: Visual token sequence of shape (batch_size, num_patches, d_model).
        """
        # Ensure 4D batch shape (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        B = x.shape[0]

        # Patchify: (B, C, H, W) -> (B, d_model, H/P, W/P)
        patches = self.patch_embed(x)

        # Flatten spatial dimensions and transpose: (B, d_model, num_patches) -> (B, num_patches, d_model)
        tokens = patches.flatten(2).transpose(1, 2)

        # Add spatial positional embedding
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)

        # Pass through stacked Transformer Encoder blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Final LayerNorm
        return self.final_norm(tokens)
