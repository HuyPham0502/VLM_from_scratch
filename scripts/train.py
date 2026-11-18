import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, setup_device, set_seed, setup_environment
from src.model.vlm import VisionLanguageModel
from src.data.dataset import ViVQADataset, VLMCollator
from src.training.trainer import VLMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vietnamese Vision-Language Model (ViVLM-MoE)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2], help="Training stage (1: Base, 2: MoE; defaults to config)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load before training")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for quick test runs")
    parser.add_argument("--logging-steps", type=int, default=None, help="Logging interval in steps")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (default: 0 for fast Windows execution)")
    parser.add_argument("--enable-moe", action="store_true", help="Explicitly enable Multimodal MoE Adapter")
    parser.add_argument("--vision-encoder", type=str, default=None, choices=["scratch", "pretrained"], help="Override vision encoder type")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.vision_encoder:
        config["model"]["vision_encoder_type"] = args.vision_encoder
    stage = args.stage if args.stage is not None else config.get("training", {}).get("stage", 1)
    if args.enable_moe or stage == 2:
        config["moe"]["enabled"] = True
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.logging_steps:
        config["training"]["logging_steps"] = args.logging_steps

    # Setup environment & caches on project directory
    setup_environment(config)
    set_seed(config["training"].get("seed", 42))
    device = setup_device()

    # Initialize model
    print(f"\n[Main] Initializing ViVLM-MoE for Stage {stage}...")
    model = VisionLanguageModel(config).to(device)

    # Determine checkpoint to load
    checkpoint_to_load = args.checkpoint
    curr_enc = config.get("model", {}).get("vision_encoder_type", "scratch")
    if not checkpoint_to_load and stage == 2:
        # Auto-detect Stage 1 checkpoint if entering Stage 2 with matching encoder
        chk_dir = Path(config.get("paths", {}).get("checkpoint_dir", "./checkpoints"))
        for candidate in ["stage1_best.pt", "stage1_last.pt", "stage1_final.pt"]:
            cand_path = chk_dir / candidate
            if cand_path.exists():
                try:
                    chk_meta = torch.load(cand_path, map_location="cpu")
                    chk_enc = chk_meta.get("config", {}).get("model", {}).get("vision_encoder_type", "scratch")
                    if chk_enc == curr_enc:
                        checkpoint_to_load = str(cand_path)
                        print(f"[Main] Auto-selected Stage 1 checkpoint matching '{curr_enc}': {checkpoint_to_load}")
                        break
                except Exception:
                    pass

    if checkpoint_to_load:
        print(f"[Main] Loading checkpoint: {checkpoint_to_load}")
        chk = torch.load(checkpoint_to_load, map_location=device)
        try:
            model.vision_encoder.load_state_dict(chk.get("vision_encoder", {}), strict=False)
        except Exception as e:
            print(f"[Main] Notice: Vision encoder weights skipped ({e}).")
        try:
            model.projector.load_state_dict(chk.get("projector", {}), strict=False)
        except Exception as e:
            print(f"[Main] Notice: Projector weights skipped ({e}).")
        if "moe_adapter" in chk and config["moe"]["enabled"]:
            try:
                model.moe_adapter.load_state_dict(chk["moe_adapter"], strict=False)
            except Exception as e:
                print(f"[Main] Notice: MoE adapter weights skipped ({e}).")

    # Build dataset & dataloader
    data_cfg = config.get("data", {})
    cache_dir = config.get("paths", {}).get("cache_dir", "./hf_cache")
    max_samples = args.max_samples or data_cfg.get("max_samples")

    train_dataset = ViVQADataset(
        dataset_name=data_cfg.get("dataset_name", "ThucPD/ViVQA"),
        split="train",
        tokenizer=model.tokenizer,
        image_size=data_cfg.get("image_size", 224),
        max_length=config["training"].get("max_length", 256),
        max_samples=max_samples,
        cache_dir=cache_dir,
    )

    collator = VLMCollator(pad_token_id=model.tokenizer.pad_token_id)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 8),
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Build validation dataloader for loss monitoring and early stopping
    val_dataloader = None
    try:
        val_max_samples = data_cfg.get("val_max_samples", 200)
        val_dataset = ViVQADataset(
            dataset_name=data_cfg.get("dataset_name", "ThucPD/ViVQA"),
            split="test",
            tokenizer=model.tokenizer,
            image_size=data_cfg.get("image_size", 224),
            max_length=config["training"].get("max_length", 256),
            max_samples=val_max_samples,
            cache_dir=cache_dir,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["training"].get("batch_size", 8),
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
        )
        print(f"[Main] Validation dataloader ready ({len(val_dataset)} samples).")
    except Exception as e:
        print(f"[Main] Notice: Validation split unavailable ({e}). Trainer will monitor training loss.")
        val_dataloader = None

    # Initialize trainer and start training
    trainer = VLMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device,
        stage=stage,
    )

    trainer.train(max_epochs=args.epochs)


if __name__ == "__main__":
    main()
