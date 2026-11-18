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
from src.training.evaluator import VLMEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese Vision-Language Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt file (auto-detects if None)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--max-samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--enable-moe", action="store_true", help="Enable MoE during evaluation")
    parser.add_argument("--skip-bert-score", action="store_true", help="Skip BERTScore calculation")
    parser.add_argument("--vision-encoder", type=str, default=None, choices=["scratch", "pretrained"], help="Override vision encoder type")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.vision_encoder:
        config["model"]["vision_encoder_type"] = args.vision_encoder
    setup_environment(config)
    set_seed(config["training"].get("seed", 42))
    device = setup_device()

    # Auto-detect checkpoint if not explicitly provided
    checkpoint_path = args.checkpoint
    curr_enc = config.get("model", {}).get("vision_encoder_type", "scratch")
    if not checkpoint_path:
        chk_dir = Path(config.get("paths", {}).get("checkpoint_dir", "./checkpoints"))
        for candidate in ["stage2_best.pt", "stage2_last.pt", "stage2_final.pt", "stage1_best.pt", "stage1_last.pt", "stage1_final.pt"]:
            cand = chk_dir / candidate
            if cand.exists():
                try:
                    chk_meta = torch.load(cand, map_location="cpu")
                    chk_enc = chk_meta.get("config", {}).get("model", {}).get("vision_encoder_type", "scratch")
                    if chk_enc == curr_enc:
                        checkpoint_path = str(cand)
                        print(f"[Main] Auto-selected checkpoint for evaluation: {checkpoint_path}")
                        break
                except Exception:
                    pass

    if checkpoint_path and "stage2" in checkpoint_path:
        config["moe"]["enabled"] = True
    if args.enable_moe:
        config["moe"]["enabled"] = True

    print("\n[Main] Loading ViVLM-MoE model for evaluation...")
    model = VisionLanguageModel(config).to(device)

    if checkpoint_path:
        print(f"[Main] Loading weights from: {checkpoint_path}")
        chk = torch.load(checkpoint_path, map_location=device)
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
    else:
        print("[Main] Evaluating initial uncalibrated weights for current architecture.")

    data_cfg = config.get("data", {})
    cache_dir = config.get("paths", {}).get("cache_dir", "./hf_cache")

    test_dataset = ViVQADataset(
        dataset_name=data_cfg.get("dataset_name", "ThucPD/ViVQA"),
        split=args.split,
        tokenizer=model.tokenizer,
        image_size=data_cfg.get("image_size", 224),
        max_length=config["training"].get("max_length", 256),
        max_samples=args.max_samples,
        cache_dir=cache_dir,
    )

    collator = VLMCollator(pad_token_id=model.tokenizer.pad_token_id)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["training"].get("batch_size", 8),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    evaluator = VLMEvaluator(
        model=model,
        tokenizer=model.tokenizer,
        device=device,
    )

    results = evaluator.evaluate(
        dataloader=test_dataloader,
        max_samples=args.max_samples,
        use_bert_score=not args.skip_bert_score,
    )


if __name__ == "__main__":
    main()
