#!/usr/bin/env python3
"""
Merge LoRA adapter into base model for deployment (CAI wrapper).

Environment Variables:
- BASE_MODEL:   Base model name (default: Qwen/Qwen3-VL-2B-Instruct)
- ADAPTER_PATH: Path to LoRA adapter (default: /home/cdsw/checkpoints/qwen3vl-2b-pgrav-qlora/final)
- OUTPUT_DIR:   Path for merged model (default: /home/cdsw/models/qwen3vl-2b-pgrav-merged)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, check_output_exists, validate_script_exists


def main():
    print("=" * 60)
    print("Merge LoRA Adapter into Base Model")
    print("=" * 60)

    base_model = os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    adapter_path = os.getenv("ADAPTER_PATH", "/home/cdsw/checkpoints/qwen3vl-2b-pgrav-qlora/final")
    output_dir = os.getenv("OUTPUT_DIR", "/home/cdsw/models/qwen3vl-2b-pgrav-merged")

    print(f"  Base model:   {base_model}")
    print(f"  Adapter path: {adapter_path}")
    print(f"  Output dir:   {output_dir}")
    print()

    # Idempotency: check for model.safetensors in output
    safetensors = Path(output_dir) / "model.safetensors"
    if not os.getenv("FORCE_REPROCESS", "false").lower() == "true":
        if check_output_exists(str(safetensors), min_size_mb=100):
            print(f"  Merged model already exists: {safetensors}")
            print("  Skipping. Set FORCE_REPROCESS=true to re-merge.")
            return

    # Validate adapter
    adapter_config = Path(adapter_path) / "adapter_config.json"
    if not adapter_config.exists():
        print(f"Error: Adapter not found at {adapter_path}")
        print("   Ensure train_vlm_phase_b job has completed")
        sys.exit(1)

    script = validate_script_exists("scripts/merge_lora_adapter.py")
    venv_python = get_venv_python()

    cmd = [
        venv_python, "-u", str(script),
        "--base-model", base_model,
        "--adapter-path", adapter_path,
        "--output-dir", output_dir,
    ]
    run_in_venv(cmd)

    # Verify
    safetensors_files = list(Path(output_dir).glob("*.safetensors"))
    if safetensors_files:
        total_size = sum(f.stat().st_size for f in safetensors_files) / 1e9
        print(f"  Merged model: {output_dir} ({total_size:.2f} GB)")
        print("Merge completed successfully")
    else:
        print(f"Error: No safetensors files found in {output_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
