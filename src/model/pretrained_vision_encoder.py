import torch
import torch.nn as nn
from transformers import AutoModel, SiglipVisionModel, CLIPVisionModel


class PretrainedVisionEncoder(nn.Module):
    """Pretrained Vision Transformer Encoder wrapper (SigLIP or CLIP).

    Extracts dense visual patch representations from a pretrained vision backbone.
    Outputs a tensor of shape (B, num_patches, d_vision), matching the input
    requirements of the Multimodal Projector.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        cache_dir: str = "./hf_cache",
        freeze: bool = True,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """Initialize PretrainedVisionEncoder.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Local cache directory for model weights.
            freeze: Whether to freeze encoder parameters.
            torch_dtype: Floating point precision for model weights.
        """
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.is_frozen = freeze

        name_lower = model_name.lower()
        if "siglip" in name_lower:
            try:
                self.model = SiglipVisionModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    local_files_only=True,
                )
            except Exception:
                self.model = SiglipVisionModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                )
            self.is_clip = False
        elif "clip" in name_lower:
            try:
                self.model = CLIPVisionModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    local_files_only=True,
                )
            except Exception:
                self.model = CLIPVisionModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                )
            self.is_clip = True
        else:
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    local_files_only=True,
                )
            except Exception:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                )
            self.is_clip = False

        # Determine visual hidden dimension
        config = self.model.config
        if hasattr(config, "vision_config"):
            self.d_model = config.vision_config.hidden_size
        elif hasattr(config, "hidden_size"):
            self.d_model = config.hidden_size
        else:
            self.d_model = 768

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        """Freeze all parameters in the vision encoder."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.is_frozen = True

    def unfreeze(self) -> None:
        """Unfreeze all parameters in the vision encoder."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        self.is_frozen = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual patch embeddings.

        Args:
            images: Tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Visual token sequence of shape (B, num_patches, d_vision).
        """
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        images = images.to(dtype=model_dtype, device=device)

        if self.is_frozen:
            with torch.no_grad():
                outputs = self.model(pixel_values=images)
        else:
            outputs = self.model(pixel_values=images)

        hidden_states = outputs.last_hidden_state

        # For CLIP: token 0 is [CLS], tokens 1:197 are spatial patch tokens
        if self.is_clip and hidden_states.shape[1] > 196:
            hidden_states = hidden_states[:, 1:, :]

        return hidden_states
