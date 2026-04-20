#!/usr/bin/env python3
"""
Evaluate fine-tuned VLM vs base model on STCray test set (CAI wrapper).

This script wraps evaluation/eval_vlm_qlora.py for CAI job execution.
All configuration is read from environment variables.

Environment Variables:
- BASE_MODEL:      Base model name (default: Qwen/Qwen3-VL-2B-Instruct)
- FINETUNED_MODEL: Path to fine-tuned LoRA adapters (default: /home/cdsw/checkpoints/qwen3vl-2b-xray-qlora/final)
- TEST_DATA:       Test JSONL path (default: data/stcray_vlm/stcray_vlm_test.jsonl)
- OUTPUT_DIR:      Evaluation results dir (default: /home/cdsw/test_results/vlm_qlora_eval)
- IOU_THRESHOLD:   IoU threshold for bbox matching (default: 0.5)
- NUM_SAMPLES:     Number of test samples (default: "" = all)
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/cdsw/cai_integration")
from utils import get_venv_env


def main():
    """Execute VLM evaluation."""
    print("=" * 60)
    print("VLM Evaluation Job (Base vs Fine-tuned)")
    print("=" * 60)

    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"

    # Verify venv exists
    if not venv_python.exists():
        print(f"Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)

    # Get configuration from environment
    base_model = os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    finetuned_model = os.getenv("FINETUNED_MODEL", "/home/cdsw/checkpoints/qwen3vl-2b-xray-qlora/final")
    test_data = os.getenv("TEST_DATA", "data/stcray_vlm/stcray_vlm_test.jsonl")
    output_dir = os.getenv("OUTPUT_DIR", "/home/cdsw/test_results/vlm_qlora_eval")
    iou_threshold = os.getenv("IOU_THRESHOLD", "0.5")
    num_samples = os.getenv("NUM_SAMPLES", "")

    print(f"  Using Python: {venv_python}")
    print(f"  Working directory: {project_root}")
    print()
    print("Evaluation Configuration:")
    print(f"  Base model:       {base_model}")
    print(f"  Fine-tuned model: {finetuned_model}")
    print(f"  Test data:        {test_data}")
    print(f"  Output dir:       {output_dir}")
    print(f"  IoU threshold:    {iou_threshold}")
    print(f"  Num samples:      {num_samples if num_samples else 'all'}")
    print()

    # Verify test data exists
    test_path = project_root / test_data
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("   Ensure convert_stcray_vlm job has completed successfully")
        sys.exit(1)

    with open(test_path) as f:
        n_test = sum(1 for _ in f)
    print(f"  Test samples: {n_test}")

    # Verify fine-tuned model exists
    ft_path = Path(finetuned_model)
    if not ft_path.exists():
        print(f"Warning: Fine-tuned model not found at {ft_path}")
        print("   Will evaluate base model only")

    # Check GPU availability
    venv_env = get_venv_env(project_root)
    try:
        gpu_check = subprocess.run(
            [str(venv_python), "-c",
             "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); "
             "print(f'GPU count: {torch.cuda.device_count()}')"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            env=venv_env,
        )
        print(gpu_check.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not check GPU availability: {e}")

    print()
    print("Starting evaluation...")
    print("-" * 60)

    # Build command
    cmd = [
        str(venv_python),
        "evaluation/eval_vlm_qlora.py",
        "--base-model", base_model,
        "--finetuned-model", finetuned_model,
        "--test-data", test_data,
        "--output-dir", output_dir,
        "--iou-threshold", iou_threshold,
    ]

    # Add num_samples if specified
    if num_samples:
        cmd.extend(["--num-samples", num_samples])

    print(f"  Command: {' '.join(cmd)}")
    print()

    # Execute evaluation with venv activated
    result = subprocess.run(cmd, cwd=str(project_root), env=venv_env)

    if result.returncode != 0:
        print(f"Error: Evaluation failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Report output location
    print()
    print("-" * 60)
    output_path = Path(output_dir)
    if output_path.exists():
        result_files = list(output_path.glob("*"))
        print(f"  Results saved to: {output_path}")
        for f in result_files:
            print(f"    {f.name}")
    else:
        print(f"  Note: Output directory not created at {output_path}")

    print()
    print("Evaluation job completed successfully")


if __name__ == "__main__":
    main()
