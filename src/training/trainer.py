import os
import math
import torch
from pathlib import Path
from tqdm import tqdm
import wandb

from .evaluator import VLMEvaluator


class EarlyStopping:
    """Early stopping utility to monitor loss and halt training when validation plateaus."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = "min"):
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait for improvement before halting.
            min_delta: Minimum reduction required to qualify as improvement.
            mode: 'min' for loss, 'max' for metrics.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def step(self, current_score: float) -> bool:
        """Evaluate current score against best score.

        Args:
            current_score: Monitored metric value.

        Returns:
            bool: True if current score is the new best, False otherwise.
        """
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class VLMTrainer:
    """Multi-stage trainer for the Vietnamese Vision-Language Model (ViVLM-MoE).

    Features:
        - Support for Stage 1 (Base ViVLM) and Stage 2 (Multimodal MoE Adapter).
        - FP16 Automatic Mixed Precision (AMP) with GradScaler.
        - Gradient accumulation for memory-efficient training.
        - Automatic Weights & Biases (W&B) experiment tracking.
        - Periodic validation and best-checkpoint saving.
    """

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        config: dict = None,
        device=None,
        stage: int = 1,
    ):
        """Initialize VLMTrainer.

        Args:
            model: VisionLanguageModel instance.
            train_dataloader: DataLoader for training split.
            val_dataloader: Optional DataLoader for validation split.
            config: Full project configuration dictionary.
            device: Compute device.
            stage: Training stage (1: Base ViVLM, 2: MoE Adapter).
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.device = device or next(model.parameters()).device
        self.stage = stage

        train_cfg = self.config.get("training", {})
        self.num_epochs = train_cfg.get("num_epochs", 5)
        self.learning_rate = float(train_cfg.get("learning_rate", 1.0e-3 if stage == 1 else 5.0e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.gradient_accumulation_steps = max(1, train_cfg.get("gradient_accumulation_steps", 2))
        self.use_fp16 = train_cfg.get("fp16", True) and (self.device.type == "cuda")
        self.logging_steps = train_cfg.get("logging_steps", 10)

        # Setup Early Stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 3),
            min_delta=float(es_cfg.get("min_delta", 0.001)),
            mode="min",
        ) if self.early_stopping_enabled else None

        paths_cfg = self.config.get("paths", {})
        self.checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve stage-specific trainable parameters
        self.trainable_params = self.model.get_trainable_parameters(stage=self.stage)
        num_trainable = sum(p.numel() for p in self.trainable_params if p.requires_grad)
        enc_type = getattr(self.model, "vision_encoder_type", "scratch")
        enc_info = getattr(self.model, 'pretrained_vision_name', 'Custom ViT') if enc_type == 'pretrained' else 'From Scratch'
        print(f"\n[Trainer] Stage {self.stage} Initialized:")
        print(f"  - Vision Tower: {enc_type.upper()} ({enc_info})")
        print(f"  - Trainable parameters: {num_trainable / 1e6:.2f} M")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Mixed Precision (FP16): {self.use_fp16}")

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Setup GradScaler for FP16 training
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)

        # Setup evaluator
        self.evaluator = VLMEvaluator(
            model=self.model,
            tokenizer=self.model.tokenizer,
            device=self.device,
        )

        # Setup Weights & Biases
        self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases run if API key is provided."""
        wandb_cfg = self.config.get("wandb", {})
        api_key = os.environ.get("WANDB_API_KEY", "").strip()

        if api_key:
            try:
                wandb.login(key=api_key, relogin=False)
                run_name = wandb_cfg.get("run_name") or f"vivlm-stage{self.stage}"
                self.wandb_run = wandb.init(
                    project=wandb_cfg.get("project", "vivlm-moe"),
                    entity=wandb_cfg.get("entity"),
                    name=run_name,
                    config=self.config,
                    reinit=True,
                )
                if self.wandb_run:
                    wandb.define_metric("step")
                    wandb.define_metric(f"stage{self.stage}/*", step_metric="step")
                    wandb.define_metric("epoch")
                print("[W&B] Initialized run successfully.")
            except Exception as e:
                print(f"[W&B] Failed to initialize: {e}. Running without online logging.")
                self.wandb_run = None
        else:
            print("[W&B] No WANDB_API_KEY found. Proceeding without W&B tracking.")
            self.wandb_run = None

    def train(self, max_epochs: int = None, max_steps: int = None) -> None:
        """Execute full training loop with logging, gradient accumulation, and evaluation.

        Args:
            max_epochs: Optional epoch override (e.g. set to 1 for quick test).
            max_steps: Optional step limit for debugging.
        """
        epochs = max_epochs or self.num_epochs
        total_batches = len(self.train_dataloader)
        est_total_steps = max(1, total_batches * epochs // self.gradient_accumulation_steps)
        global_step = 0
        best_val_score = -1.0

        # Dynamically adapt logging interval if dataset or steps are small
        if est_total_steps <= 20:
            effective_logging_steps = max(1, min(self.logging_steps, 2))
        else:
            effective_logging_steps = self.logging_steps

        print(f"\n[Trainer] Starting training for {epochs} epoch(s)...")
        print(f"  - Batches per epoch: {total_batches}")
        print(f"  - Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"  - Estimated total optimization steps: {est_total_steps}")
        print(f"  - Effective logging interval: every {effective_logging_steps} step(s)")
        self.model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            accumulated_loss = 0.0
            accumulated_batches = 0

            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}/{epochs} (Stage {self.stage})",
                unit="batch",
            )

            for step, batch in enumerate(pbar):
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass under Automatic Mixed Precision (AMP)
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    outputs = self.model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"] / self.gradient_accumulation_steps

                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
                raw_loss = loss.item() * self.gradient_accumulation_steps
                accumulated_loss += raw_loss
                accumulated_batches += 1
                epoch_loss += raw_loss

                # Perform optimizer step every gradient_accumulation_steps
                is_accum_step = (step + 1) % self.gradient_accumulation_steps == 0
                is_last_batch = (step + 1) == total_batches

                if is_accum_step or is_last_batch:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params,
                        self.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    global_step += 1

                    # Logging trigger
                    is_log_step = (
                        (global_step % effective_logging_steps == 0)
                        or (global_step == 1)
                        or is_last_batch
                    )

                    if is_log_step and (accumulated_batches > 0):
                        avg_loss = accumulated_loss / accumulated_batches
                        aux_loss_val = outputs.get("aux_loss", 0.0)
                        if isinstance(aux_loss_val, torch.Tensor):
                            aux_loss_val = aux_loss_val.item()

                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "aux_loss": f"{aux_loss_val:.4f}",
                        })

                        if self.wandb_run:
                            wandb.log({
                                f"stage{self.stage}/train_loss": avg_loss,
                                f"stage{self.stage}/aux_loss": aux_loss_val,
                                f"stage{self.stage}/learning_rate": self.optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                                "step": global_step,
                            }, step=global_step)

                        accumulated_loss = 0.0
                        accumulated_batches = 0

                    if max_steps and global_step >= max_steps:
                        print(f"[Trainer] Reached maximum steps limit ({max_steps}). Exiting training loop.")
                        break

            pbar.close()
            epoch_avg_loss = epoch_loss / max(1, total_batches)
            print(f"\n[Trainer] Epoch {epoch}/{epochs} complete. Train Loss: {epoch_avg_loss:.4f}")

            log_dict = {
                f"stage{self.stage}/epoch_loss": epoch_avg_loss,
                "epoch": epoch,
                "step": global_step,
            }

            # Evaluate validation loss if validation split is provided
            if self.val_dataloader:
                val_loss = self.evaluate_loss(self.val_dataloader)
                print(f"[Trainer] Validation Loss: {val_loss:.4f}")
                log_dict[f"stage{self.stage}/val_loss"] = val_loss
                monitored_score = val_loss
            else:
                monitored_score = epoch_avg_loss

            if self.wandb_run:
                wandb.log(log_dict, step=global_step)

            # Checkpoint saving: strictly best and last
            is_best = False
            if self.early_stopping:
                is_best = self.early_stopping.step(monitored_score)
            elif epoch == 1 or monitored_score < best_val_score:
                best_val_score = monitored_score
                is_best = True

            if is_best:
                print(f"[Checkpoint] Loss improved to {monitored_score:.4f}. Saving best model...")
                self.save_checkpoint(f"stage{self.stage}_best.pt")

            # Always save current state as 'last'
            self.save_checkpoint(f"stage{self.stage}_last.pt")

            # Check early stopping condition
            if self.early_stopping and self.early_stopping.early_stop:
                print(
                    f"\n[EarlyStopping] Early stopping triggered! "
                    f"No improvement for {self.early_stopping.patience} consecutive epochs. "
                    f"Best monitored score: {self.early_stopping.best_score:.4f}."
                )
                break

        print(f"\n[Trainer] Training for Stage {self.stage} finished successfully!")
        self.save_checkpoint(f"stage{self.stage}_last.pt")
        # Keep stage_final.pt for backward compatibility
        self.save_checkpoint(f"stage{self.stage}_final.pt")

        if self.wandb_run:
            self.wandb_run.finish()

    @torch.no_grad()
    def evaluate_loss(self, dataloader) -> float:
        """Compute average loss across the validation split without tracking gradients.

        Args:
            dataloader: Validation DataLoader.

        Returns:
            float: Mean validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in dataloader:
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]

            total_loss += loss.item()
            count += 1

        self.model.train()
        return total_loss / max(1, count)

    def save_checkpoint(self, filename: str) -> None:
        """Save model trainable weights and configuration.

        Args:
            filename: Target file name (e.g. 'stage1_best.pt').
        """
        save_path = self.checkpoint_dir / filename
        state_dict = {
            "stage": self.stage,
            "vision_encoder": self.model.vision_encoder.state_dict(),
            "projector": self.model.projector.state_dict(),
            "moe_adapter": self.model.moe_adapter.state_dict(),
            "config": self.config,
        }
        torch.save(state_dict, save_path)
        print(f"[Checkpoint] Saved checkpoint to: {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained checkpoint weights into model.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.vision_encoder.load_state_dict(checkpoint["vision_encoder"], strict=False)
        self.model.projector.load_state_dict(checkpoint["projector"], strict=False)
        if "moe_adapter" in checkpoint:
            self.model.moe_adapter.load_state_dict(checkpoint["moe_adapter"], strict=False)
        print(f"[Checkpoint] Successfully loaded weights from: {checkpoint_path}")
