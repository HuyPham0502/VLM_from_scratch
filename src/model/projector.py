import torch
import torch.nn as nn


class MultimodalProjector(nn.Module):
    """Two-layer MLP projector bridging the Vision Encoder and Language Model.

    Maps visual token representations from dimensionality d_vision to d_llm,
    allowing visual tokens to be directly ingested by the Language Model decoder.
    """

    def __init__(self, d_vision: int = 512, d_llm: int = 896):
        """Initialize MultimodalProjector.

        Args:
            d_vision: Hidden dimensionality of the Vision Transformer.
            d_llm: Hidden dimensionality of the Pretrained Language Model.
        """
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_vision, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Project visual tokens into LLM embedding space.

        Args:
            visual_tokens: Tensor of shape (batch_size, num_patches, d_vision).

        Returns:
            torch.Tensor: Projected tokens of shape (batch_size, num_patches, d_llm).
        """
        target_dtype = self.projector[0].weight.dtype
        if visual_tokens.dtype != target_dtype:
            visual_tokens = visual_tokens.to(dtype=target_dtype)
        return self.projector(visual_tokens)
