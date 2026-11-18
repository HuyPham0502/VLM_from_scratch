import os
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Disable Gradio telemetry to prevent network hanging in isolated environments
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import torch
from PIL import Image
from torchvision import transforms
import gradio as gr

from src.utils.config import load_config, setup_device, setup_environment
from src.model.vlm import VisionLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="ViVLM-MoE Gradio Web Interface")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    parser.add_argument("--dry-run", action="store_true", help="Test initialization and exit without launching UI")
    parser.add_argument("--vision-encoder", type=str, default=None, choices=["scratch", "pretrained"], help="Override vision encoder type")
    return parser.parse_args()


def create_ui(model, device, image_size=224):
    """Build Gradio Blocks application interface."""

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def answer_question(image, question, enable_moe, temperature, max_tokens):
        if image is None:
            return "Vui lòng tải lên một hình ảnh trước khi hỏi."
        if not question or not question.strip():
            return "Vui lòng nhập câu hỏi về bức ảnh."

        # Set MoE state
        model.set_moe(enable_moe)

        # Preprocess PIL image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        image_tensor = transform(image).to(device)

        # Generate Vietnamese answer
        response = model.generate(
            image=image_tensor,
            question=question.strip(),
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        return response

    with gr.Blocks(title="ViVLM-MoE: Vietnamese Vision-Language Model") as demo:
        gr.Markdown(
            """
            # 🇻🇳 ViVLM-MoE: Vietnamese Vision-Language Model
            **Mô hình thị giác - ngôn ngữ tiếng Việt tích hợp Multimodal Mixture-of-Experts (MoE) xây dựng từ đầu.**
            - *Vision Encoder & Projector:* Tự xây dựng từ đầu (From Scratch).
            - *Multimodal MoE Adapter:* Router Top-K + Expert MLPs sau tầng Fusion.
            - *Language Decoder:* Qwen2.5-0.5B-Instruct (Pretrained & Frozen).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="1. Tải ảnh lên")
                question_input = gr.Textbox(
                    label="2. Nhập câu hỏi tiếng Việt",
                    placeholder="Ví dụ: Bức ảnh chụp cảnh gì? Có bao nhiêu người?",
                    lines=2,
                )

                with gr.Accordion("⚙️ Cấu hình suy luận & MoE", open=False):
                    enable_moe = gr.Checkbox(label="Bật Multimodal MoE Adapter", value=False)
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature")
                    max_tokens = gr.Slider(minimum=16, maximum=128, value=64, step=8, label="Max New Tokens")

                submit_btn = gr.Button("🚀 Trả lời câu hỏi", variant="primary")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(label="3. Câu trả lời của mô hình", lines=4)

        submit_btn.click(
            fn=answer_question,
            inputs=[image_input, question_input, enable_moe, temperature, max_tokens],
            outputs=[answer_output],
        )

        gr.Markdown(
            """
            ### 💡 Gợi ý câu hỏi mẫu:
            - *Bức ảnh chụp cảnh gì?*
            - *Có bao nhiêu người / đồ vật trong ảnh?*
            - *Màu sắc chủ đạo của bức ảnh là gì?*
            - *Mô tả chi tiết những gì bạn nhìn thấy.*
            """
        )

    return demo


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.vision_encoder:
        config["model"]["vision_encoder_type"] = args.vision_encoder
    setup_environment(config)
    device = setup_device()

    print("[Main] Initializing ViVLM-MoE model...")
    model = VisionLanguageModel(config).to(device)

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
                        print(f"[Main] Auto-selected checkpoint for UI: {checkpoint_path}")
                        break
                except Exception:
                    pass

    if checkpoint_path:
        print(f"[Main] Loading trained checkpoint from: {checkpoint_path}")
        chk = torch.load(checkpoint_path, map_location=device)
        try:
            model.vision_encoder.load_state_dict(chk.get("vision_encoder", {}), strict=False)
        except Exception as e:
            print(f"[Main] Notice: Vision encoder weights skipped ({e}).")
        try:
            model.projector.load_state_dict(chk.get("projector", {}), strict=False)
        except Exception as e:
            print(f"[Main] Notice: Projector weights skipped ({e}).")
        if "moe_adapter" in chk:
            try:
                model.moe_adapter.load_state_dict(chk["moe_adapter"], strict=False)
            except Exception as e:
                print(f"[Main] Notice: MoE adapter weights skipped ({e}).")

    image_size = config.get("model", {}).get("image_size", 224)
    demo = create_ui(model, device, image_size=image_size)

    if args.dry_run:
        print("[Main] Dry-run test passed successfully. UI initialized without errors.")
        return

    print(f"\n[Gradio] Launching Web Interface on port {args.port}...")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
