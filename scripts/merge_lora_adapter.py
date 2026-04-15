#!/usr/bin/env python3
"""
Merge QLoRA adapter into base Qwen3-VL model for deployment.

Loads the base model, applies the fine-tuned LoRA adapter, merges weights,
and saves a standalone model that can be served directly by vLLM.

Usage:
  python scripts/merge_lora_adapter.py \
      --base-model Qwen/Qwen3-VL-2B-Instruct \
      --adapter-path /home/cdsw/checkpoints/qwen3vl-2b-xray-qlora/final \
      --output-dir /home/cdsw/models/qwen3vl-2b-xray-merged

Environment Variables:
  BASE_MODEL:   Base model name or path (default: Qwen/Qwen3-VL-2B-Instruct)
  ADAPTER_PATH: Path to fine-tuned LoRA adapter (default: checkpoints/qwen3vl-2b-xray-qlora/final)
  OUTPUT_DIR:   Path for merged model output (default: models/qwen3vl-2b-xray-merged)
"""

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge QLoRA adapter into base model")
    parser.add_argument(
        "--base-model", type=str,
        default=os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct"),
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter-path", type=str,
        default=os.getenv("ADAPTER_PATH", "checkpoints/qwen3vl-2b-xray-qlora/final"),
        help="Path to fine-tuned LoRA adapter",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.getenv("OUTPUT_DIR", "models/qwen3vl-2b-xray-merged"),
        help="Output directory for merged model",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Merge QLoRA Adapter into Base Model")
    print("=" * 60)
    print(f"  Base model:   {args.base_model}")
    print(f"  Adapter path: {args.adapter_path}")
    print(f"  Output dir:   {args.output_dir}")
    print()

    # Validate adapter path
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"Error: Adapter path not found: {adapter_path}")
        sys.exit(1)

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"Error: No adapter_config.json found in {adapter_path}")
        print("  This doesn't look like a valid LoRA adapter directory")
        sys.exit(1)

    print(f"  Adapter config found: {adapter_config}")

    # Import heavy dependencies after validation
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    # Step 1: Load base model
    print("Step 1/4: Loading base model...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
    print()

    # Step 2: Load and apply LoRA adapter
    print("Step 2/4: Loading LoRA adapter...")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Print adapter info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable / 1e6:.1f}M ({trainable / total * 100:.2f}%)")
    print()

    # Step 3: Merge weights
    print("Step 3/4: Merging LoRA weights into base model...")
    t0 = time.time()
    model = model.merge_and_unload()
    print(f"  Merged in {time.time() - t0:.1f}s")
    print()

    # Step 4: Save merged model + processor
    print("Step 4/4: Saving merged model...")
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))

    # Save processor (tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True,
    )
    processor.save_pretrained(str(output_dir))
    save_time = time.time() - t0

    # Report size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"  Saved in {save_time:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"  Size: {total_size / 1e9:.2f} GB")
    print(f"  Files: {len(list(output_dir.iterdir()))}")
    print()

    print("=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print()
    print("Deploy with vLLM:")
    print(f'  model_source: "{output_dir}"')
    print()


if __name__ == "__main__":
    main()
