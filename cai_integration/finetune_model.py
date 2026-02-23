#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-VL with checkpoint resumption support.

This script wraps training/train_local.py for CAI job execution.
It supports resuming from checkpoints via RESUME_FROM_CHECKPOINT env var.

Environment Variables:
- CONFIG_FILE: Training config path (default: cai_integration/config/cai_train_config.yaml)
- RESUME_FROM_CHECKPOINT: Checkpoint path to resume from (optional)
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Execute model fine-tuning."""
    print("=" * 60)
    print("Model Fine-tuning Job")
    print("=" * 60)
    
    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"
    
    # Verify venv exists
    if not venv_python.exists():
        print(f"❌ Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)
    
    # Get configuration
    config_file = os.getenv("CONFIG_FILE", "cai_integration/config/cai_train_config.yaml")
    resume_checkpoint = os.getenv("RESUME_FROM_CHECKPOINT", "")
    
    print(f"✓ Using Python: {venv_python}")
    print(f"✓ Working directory: {project_root}")
    print()
    print(f"Fine-tuning Configuration:")
    print(f"  Config: {config_file}")
    print(f"  Resume from: {resume_checkpoint if resume_checkpoint else 'None (fresh start)'}")
    print()
    
    # Verify config exists
    config_path = project_root / config_file
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        sys.exit(1)
    
    # Verify VQA datasets exist
    train_vqa = project_root / "data/stcray_vqa_train.jsonl"
    val_vqa = project_root / "data/stcray_vqa_val.jsonl"
    
    if not train_vqa.exists():
        print(f"❌ Error: Training VQA dataset not found at {train_vqa}")
        print("   Ensure generate_vqa job has completed successfully")
        sys.exit(1)
    
    if not val_vqa.exists():
        print(f"❌ Error: Validation VQA dataset not found at {val_vqa}")
        print("   Ensure generate_vqa job has completed successfully")
        sys.exit(1)
    
    print(f"✓ Training data verified: {train_vqa}")
    print(f"✓ Validation data verified: {val_vqa}")
    print()
    
    # Check GPU availability
    try:
        gpu_check = subprocess.run(
            [str(venv_python), "-c", "import torch; print(f'GPUs: {torch.cuda.device_count()}')"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        print(gpu_check.stdout.strip())
    except Exception as e:
        print(f"⚠ Warning: Could not check GPU availability: {e}")
    
    print()
    print("=" * 60)
    print("Starting Fine-tuning")
    print("=" * 60)
    print("This may take 6-12 hours depending on GPU configuration...")
    print("Checkpoints will be saved every 500 steps")
    print()
    
    # Build command
    cmd = [
        str(venv_python),
        "training/train_local.py",
        "--config", config_file,
    ]
    
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        cmd.extend(["--resume-from-checkpoint", resume_checkpoint])
    
    # Run training
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        print()
        print("=" * 60)
        print(f"❌ Fine-tuning failed with exit code {result.returncode}")
        print("=" * 60)
        print()
        print("To resume from last checkpoint:")
        print("  1. Find latest checkpoint:")
        print("     ls -d /home/cdsw/outputs/qwen25vl_stcray_lora/checkpoint-*")
        print()
        print("  2. Set RESUME_FROM_CHECKPOINT and re-trigger job:")
        print("     RESUME_FROM_CHECKPOINT=/home/cdsw/outputs/qwen25vl_stcray_lora/checkpoint-XXXX")
        sys.exit(result.returncode)
    
    print()
    print("=" * 60)
    print("✓ Fine-tuning Complete")
    print("=" * 60)
    print()
    print(f"Model saved to: /home/cdsw/outputs/qwen25vl_stcray_lora")
    print()
    print("Next steps:")
    print("  1. Evaluate model performance:")
    print("     python evaluation/eval_vqa.py --model outputs/qwen25vl_stcray_lora --test-file data/stcray_vqa_val.jsonl")
    print()
    print("  2. Push model to HuggingFace (optional):")
    print("     huggingface-cli upload your-org/qwen25vl-xray-finetuned outputs/qwen25vl_stcray_lora")
    # Success - exit normally without sys.exit(0) for CAI compatibility


if __name__ == "__main__":
    main()
