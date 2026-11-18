import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from .vision_encoder import VisionEncoder
from .pretrained_vision_encoder import PretrainedVisionEncoder
from .projector import MultimodalProjector
from .moe import MultimodalMoELayer


class VisionLanguageModel(nn.Module):
    """Hybrid Vietnamese Vision-Language Model (ViVLM-MoE).

    Architecture:
        1. Custom Vision Transformer: Extracts visual tokens from raw images.
        2. Multimodal Projector: Projects visual tokens to language embedding dimension.
        3. Pretrained Tokenizer & Embeddings: Projects Vietnamese text prompt to embeddings.
        4. Multimodal Fusion: Concatenates visual tokens and text tokens into a unified sequence.
        5. Multimodal MoE Adapter (Optional / Stage 2): Routes multimodal tokens to expert MLPs.
        6. Pretrained LLM Decoder (Qwen2.5-0.5B-Instruct): Autoregressive generation.
    """

    def __init__(self, config: dict):
        """Initialize VisionLanguageModel from configuration dictionary.

        Args:
            config: Full project configuration dictionary.
        """
        super().__init__()
        self.config = config

        model_cfg = config["model"]
        moe_cfg = config.get("moe", {})

        self.d_vision = model_cfg.get("d_vision", 512)
        self.d_llm = model_cfg.get("llm_hidden_size", 896)
        self.llm_name = model_cfg.get("llm_name", "Qwen/Qwen2.5-0.5B-Instruct")
        self.use_moe = moe_cfg.get("enabled", False)
        self.aux_loss_weight = moe_cfg.get("aux_loss_weight", 0.01)

        self.cache_dir = config.get("paths", {}).get("cache_dir", "./hf_cache")

        # Component 1: Vision Encoder (Option 1: scratch, Option 2: pretrained)
        self.vision_encoder_type = str(model_cfg.get("vision_encoder_type", "scratch")).lower()
        self.pretrained_vision_name = model_cfg.get("pretrained_vision_name", "google/siglip-base-patch16-224")
        self.freeze_vision_encoder = model_cfg.get("freeze_vision_encoder", True)

        if self.vision_encoder_type == "pretrained":
            dtype = torch.float16 if config.get("training", {}).get("fp16", True) else torch.float32
            self.vision_encoder = PretrainedVisionEncoder(
                model_name=self.pretrained_vision_name,
                cache_dir=self.cache_dir,
                freeze=self.freeze_vision_encoder,
                torch_dtype=dtype,
            )
            self.d_vision = self.vision_encoder.d_model
            print(f"[Model] Initialized Pretrained Vision Encoder: {self.pretrained_vision_name} (d_vision={self.d_vision})")
        else:
            self.d_vision = model_cfg.get("d_vision", 512)
            self.vision_encoder = VisionEncoder(
                image_size=model_cfg.get("image_size", 224),
                patch_size=model_cfg.get("patch_size", 16),
                in_channels=3,
                d_model=self.d_vision,
                num_layers=model_cfg.get("num_encoder_layers", 6),
                num_heads=model_cfg.get("num_heads", 8),
                dropout=model_cfg.get("dropout", 0.1),
            )
            print(f"[Model] Initialized Custom ViT from scratch (d_vision={self.d_vision})")

        # Component 2: Multimodal Projector (built from scratch, auto-adapted to d_vision)
        self.projector = MultimodalProjector(
            d_vision=self.d_vision,
            d_llm=self.d_llm,
        )

        # Component 3: Multimodal MoE Adapter (built from scratch)
        self.moe_adapter = MultimodalMoELayer(
            d_model=self.d_llm,
            num_experts=moe_cfg.get("num_experts", 4),
            top_k=moe_cfg.get("top_k", 2),
            dropout=model_cfg.get("dropout", 0.1),
        )

        # Component 4 & 5: Pretrained Tokenizer and Language Model
        print(f"[Model] Loading Pretrained LLM & Tokenizer: {self.llm_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LLM with FP16 precision
        dtype = torch.float16 if config.get("training", {}).get("fp16", True) else torch.float32
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                cache_dir=self.cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                cache_dir=self.cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        # Freeze LLM weights by default
        if model_cfg.get("freeze_llm", True):
            self.freeze_llm()

    def freeze_llm(self) -> None:
        """Freeze all parameters in the pretrained LLM decoder."""
        for param in self.llm.parameters():
            param.requires_grad = False
        print("[Model] Pretrained LLM weights have been FROZEN.")

    def set_moe(self, enabled: bool, verbose: bool = False) -> None:
        """Enable or disable the Multimodal MoE Adapter.

        Args:
            enabled: Boolean flag to activate MoE routing.
            verbose: Whether to print status updates.
        """
        changed = (self.use_moe != enabled)
        self.use_moe = enabled
        if verbose and changed:
            status = "ENABLED" if enabled else "DISABLED"
            print(f"[Model] Multimodal MoE Adapter is now {status}.")

    def get_trainable_parameters(self, stage: int = 1) -> list:
        """Retrieve trainable parameter groups tailored for each training stage.

        Args:
            stage: Training stage (1: Base ViVLM, 2: MoE Adapter).

        Returns:
            list: List of parameter tensors requiring gradients.
        """
        if stage == 1:
            # Stage 1: Train Projector (and Vision Encoder if trained from scratch)
            self.set_moe(False, verbose=False)
            trainable = []
            if self.vision_encoder_type == "pretrained" and self.freeze_vision_encoder:
                for p in self.vision_encoder.parameters():
                    p.requires_grad = False
            else:
                for p in self.vision_encoder.parameters():
                    p.requires_grad = True
                    trainable.append(p)

            for p in self.projector.parameters():
                p.requires_grad = True
                trainable.append(p)

            for p in self.moe_adapter.parameters():
                p.requires_grad = False
            return trainable

        elif stage == 2:
            # Stage 2: Train only the MoE Adapter (freeze Vision, Projector, and LLM)
            self.set_moe(True, verbose=False)
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.projector.parameters():
                p.requires_grad = False
            trainable = []
            for p in self.moe_adapter.parameters():
                p.requires_grad = True
                trainable.append(p)
            return trainable

        else:
            raise ValueError(f"Unknown training stage: {stage}. Expected 1 or 2.")

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """End-to-end forward pass with Multimodal Fusion, MoE, and Loss Masking.

        Args:
            images: Batch of image tensors of shape (batch_size, 3, H, W).
            input_ids: Text token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
            labels: Target token IDs for Cross-Entropy loss of shape (batch_size, seq_len).

        Returns:
            dict containing:
                - "loss": Total scalar loss (Cross-Entropy + MoE Aux Loss).
                - "logits": Model output prediction logits.
                - "aux_loss": Auxiliary load balancing loss scalar.
        """
        B = images.shape[0]

        # Step 1: Extract visual tokens and project to d_llm -> (B, N_p, d_llm)
        visual_tokens = self.vision_encoder(images)
        projected_visual = self.projector(visual_tokens).to(dtype=self.llm.dtype)
        num_visual_tokens = projected_visual.shape[1]

        # Step 2: Retrieve text embeddings from Pretrained LLM embedding layer -> (B, N_t, d_llm)
        text_embeds = self.llm.get_input_embeddings()(input_ids).to(dtype=self.llm.dtype)

        # Step 3: Multimodal Fusion by concatenating along sequence dimension -> (B, N_p + N_t, d_llm)
        fused_embeds = torch.cat([projected_visual, text_embeds], dim=1)

        # Step 4: Construct combined attention mask
        if attention_mask is not None:
            # Ones for all visual tokens since they are fully visible
            visual_mask = torch.ones((B, num_visual_tokens), device=images.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.ones((B, fused_embeds.shape[1]), device=images.device, dtype=torch.long)

        # Step 5: Route through Multimodal MoE Adapter if enabled
        aux_loss = torch.tensor(0.0, device=images.device)
        if self.use_moe:
            fused_embeds, aux_loss = self.moe_adapter(fused_embeds)

        # Step 6: Prepare labels with Loss Masking (set -100 for all visual positions)
        full_labels = None
        if labels is not None:
            visual_labels = torch.full(
                (B, num_visual_tokens),
                fill_value=-100,
                device=labels.device,
                dtype=labels.dtype,
            )
            full_labels = torch.cat([visual_labels, labels], dim=1)

        # Step 7: Pass multimodal embeddings to Pretrained LLM Decoder
        outputs = self.llm(
            inputs_embeds=fused_embeds,
            attention_mask=combined_mask,
            labels=full_labels,
            return_dict=True,
        )

        total_loss = outputs.loss
        if total_loss is not None and self.use_moe:
            total_loss = total_loss + (self.aux_loss_weight * aux_loss)

        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "aux_loss": aux_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Autoregressive generation for a single image-question pair in Vietnamese.

        Args:
            image: Preprocessed image tensor of shape (1, 3, H, W) or (3, H, W).
            question: Vietnamese question string.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (lower is more deterministic).
            top_p: Nucleus sampling probability threshold.

        Returns:
            str: Generated Vietnamese answer string.
        """
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)

        device = next(self.parameters()).device
        image = image.to(device)

        # Step 1: Format prompt using simple instruction template
        prompt_text = f"Hình ảnh: <image>\nCâu hỏi: {question}\nTrả lời:"
        tokenized = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # Step 2: Project visual tokens and text embeddings
        visual_tokens = self.vision_encoder(image)
        projected_visual = self.projector(visual_tokens).to(dtype=self.llm.dtype)  # (1, N_p, d_llm)
        text_embeds = self.llm.get_input_embeddings()(input_ids).to(dtype=self.llm.dtype)  # (1, N_t, d_llm)

        # Step 3: Multimodal Fusion
        fused_embeds = torch.cat([projected_visual, text_embeds], dim=1)

        # Step 4: Apply MoE if enabled
        if self.use_moe:
            fused_embeds, _ = self.moe_adapter(fused_embeds)

        visual_mask = torch.ones((1, projected_visual.shape[1]), device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # Step 5: Autoregressive decoding loop
        generated_ids = []
        current_embeds = fused_embeds
        current_mask = combined_mask

        for _ in range(max_new_tokens):
            outputs = self.llm(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                # Apply top-p (nucleus) filtering
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled_logits[:, indices_to_remove] = float("-inf")
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            if token_id in (self.tokenizer.eos_token_id, getattr(self.tokenizer, "eot_id", None)):
                break

            generated_ids.append(token_id)
            next_embed = self.llm.get_input_embeddings()(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            next_mask = torch.ones((1, 1), device=device, dtype=current_mask.dtype)
            current_mask = torch.cat([current_mask, next_mask], dim=1)

        # Decode generated token IDs back to text
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
