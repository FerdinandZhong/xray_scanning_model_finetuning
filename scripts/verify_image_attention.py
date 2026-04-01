#!/usr/bin/env python3
"""
Verification gate: prove that VLM training actually attends to images.

Two checks:
1. Pre-training: verify collate_fn output contains vision tokens + image_grid_thw
2. Post-training: feed same prompt with different images, assert outputs differ

Usage:
  # Pre-training check (no model needed):
  python scripts/verify_image_attention.py --check-collate \
      --data data/stcray_vlm/stcray_vlm_train.jsonl

  # Post-training check (needs fine-tuned model):
  python scripts/verify_image_attention.py --check-model \
      --model checkpoints/qwen3vl-2b-xray-qlora/final \
      --data data/stcray_vlm/stcray_vlm_test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# Allow imports from project root
sys.path.append(str(Path(__file__).parent.parent))


def check_collate(data_path: str, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
    """Verify collate_fn produces correct multimodal outputs."""
    from training.vqa_dataset import XrayVQADataset, collate_fn

    print("=" * 60)
    print("VERIFICATION GATE: Collate Function Check")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    dataset = XrayVQADataset(
        jsonl_file=data_path,
        processor=processor,
        max_seq_length=2048,
    )

    # Grab 2 samples
    samples = [dataset[0], dataset[1]]
    batch = collate_fn(samples, processor, max_seq_length=2048)

    errors = []

    # Check 1: image_grid_thw present
    if "image_grid_thw" not in batch:
        errors.append("image_grid_thw MISSING from collate output")
    else:
        thw = batch["image_grid_thw"]
        if thw.dim() != 2 or thw.shape[1] != 3:
            errors.append(f"image_grid_thw has unexpected shape {thw.shape}, expected (N, 3)")
        else:
            print(f"  [PASS] image_grid_thw present, shape={thw.shape}")

    # Check 2: pixel_values present
    if "pixel_values" not in batch:
        errors.append("pixel_values MISSING from collate output")
    else:
        print(f"  [PASS] pixel_values present, shape={batch['pixel_values'].shape}")

    # Check 3: vision tokens in input_ids
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    has_vision = (batch["input_ids"] == image_pad_id).any().item()
    if not has_vision:
        errors.append("No <|image_pad|> tokens found in input_ids")
    else:
        count = (batch["input_ids"] == image_pad_id).sum().item()
        print(f"  [PASS] {count} vision tokens found in input_ids")

    # Check 4: labels have masking (not all tokens are loss targets)
    masked = (batch["labels"] == -100).sum().item()
    total = batch["labels"].numel()
    if masked == 0:
        errors.append("Labels have no masking -- all tokens in loss (prompt should be masked)")
    elif masked == total:
        errors.append("Labels are fully masked -- no tokens in loss (response template not found)")
    else:
        print(f"  [PASS] Labels: {masked}/{total} tokens masked ({masked/total:.0%} prompt, {(total-masked)/total:.0%} response)")

    print()
    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        print("\nVERIFICATION FAILED")
        return False
    else:
        print("ALL CHECKS PASSED: images are correctly integrated into the token stream")
        return True


def check_model(
    data_path: str,
    model_path: str,
    base_model: str = "Qwen/Qwen3-VL-2B-Instruct",
):
    """Verify model outputs differ for different images with the same prompt."""
    print("=" * 60)
    print("VERIFICATION GATE: Image-Dependent Output Check")
    print("=" * 60)

    # Load test data -- pick 2 samples with different categories
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
            if len(samples) >= 10:
                break

    # Find 2 samples with different images
    img1_path = samples[0]["image_path"]
    img2_path = None
    for s in samples[1:]:
        if s["image_path"] != img1_path:
            img2_path = s["image_path"]
            break

    if img2_path is None:
        print("  [SKIP] Could not find 2 different images in first 10 samples")
        return True

    print(f"  Image 1: {Path(img1_path).name}")
    print(f"  Image 2: {Path(img2_path).name}")

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if Path(model_path).exists():
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # Same prompt for both images
    prompt = "Detect and list all prohibited items in this X-ray baggage scan with their bounding boxes."
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = []
    for img_path in [img1_path, img2_path]:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
        # Extract assistant response
        response = decoded.split("assistant")[-1].strip() if "assistant" in decoded else decoded
        outputs.append(response)

    print(f"\n  Output 1: {outputs[0][:120]}...")
    print(f"  Output 2: {outputs[1][:120]}...")

    if outputs[0] == outputs[1]:
        print("\n  [FAIL] Both images produced identical output -- model may not attend to images")
        return False
    else:
        print("\n  [PASS] Different images produce different outputs -- model attends to images")
        return True


def main():
    parser = argparse.ArgumentParser(description="Verify VLM training attends to images")
    parser.add_argument("--check-collate", action="store_true", help="Check collate_fn output")
    parser.add_argument("--check-model", action="store_true", help="Check model image-dependence")
    parser.add_argument("--data", required=True, help="Path to JSONL data file")
    parser.add_argument("--model", default=None, help="Path to fine-tuned LoRA adapters")
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3-VL-2B-Instruct", help="Base model name"
    )
    args = parser.parse_args()

    if not args.check_collate and not args.check_model:
        parser.error("Specify --check-collate and/or --check-model")

    all_passed = True

    if args.check_collate:
        if not check_collate(args.data, args.base_model):
            all_passed = False

    if args.check_model:
        if not args.model:
            parser.error("--model is required for --check-model")
        if not check_model(args.data, args.model, args.base_model):
            all_passed = False

    print()
    if all_passed:
        print("VERIFICATION GATE: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("VERIFICATION GATE: FAILED -- do NOT proceed with full training")
        sys.exit(1)


if __name__ == "__main__":
    main()
