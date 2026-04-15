#!/usr/bin/env python3
"""
Train Qwen3-VL-2B with QLoRA on STCray VQA dataset (CAI wrapper).

This script wraps training/train_vlm_qlora.py for CAI job execution.
All configuration is read from environment variables.

Environment Variables:
- MODEL_NAME:                    Model to fine-tune (default: Qwen/Qwen3-VL-2B-Instruct)
- TRAIN_DATA:                    Training JSONL path (default: data/stcray_vlm/stcray_vlm_train.jsonl)
- EVAL_DATA:                     Evaluation JSONL path (default: data/stcray_vlm/stcray_vlm_val.jsonl)
- OUTPUT_DIR:                    Checkpoint output dir (default: /home/cdsw/checkpoints/qwen3vl-2b-xray-qlora)
- BATCH_SIZE:                    Per-device batch size (default: 2)
- EPOCHS:                        Training epochs (default: 3)
- GRADIENT_ACCUMULATION_STEPS:   Gradient accumulation (default: 4)
- LEARNING_RATE:                 Learning rate (default: 2e-4)
- WARMUP_STEPS:                  Warmup steps (default: 100)
- MAX_SEQ_LENGTH:                Max sequence length (default: 2048)
- LOGGING_STEPS:                 Log every N steps (default: 10)
- SAVE_STEPS:                    Save checkpoint every N steps (default: 500)
- EVAL_STEPS:                    Evaluate every N steps (default: 500)
- LORA_R:                        LoRA rank (default: 16)
- LORA_ALPHA:                    LoRA alpha (default: 32)
- LORA_DROPOUT:                  LoRA dropout (default: 0.05)
- SEED:                          Random seed (default: 42)
- RESUME_FROM_CHECKPOINT:        Checkpoint path to resume (default: "" = fresh start)
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_venv_env


def main():
    """Execute VLM QLoRA training."""
    print("=" * 60)
    print("VLM QLoRA Training Job (Phase A: Domain Adaptation)")
    print("=" * 60)

    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"

    # Verify venv exists
    if not venv_python.exists():
        print(f"Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)

    # Get configuration from environment
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
    train_data = os.getenv("TRAIN_DATA", "data/stcray_vlm/stcray_vlm_train.jsonl")
    eval_data = os.getenv("EVAL_DATA", "data/stcray_vlm/stcray_vlm_val.jsonl")
    output_dir = os.getenv("OUTPUT_DIR", "/home/cdsw/checkpoints/qwen3vl-2b-xray-qlora")
    batch_size = os.getenv("BATCH_SIZE", "2")
    epochs = os.getenv("EPOCHS", "3")
    grad_accum = os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")
    learning_rate = os.getenv("LEARNING_RATE", "2e-4")
    warmup_steps = os.getenv("WARMUP_STEPS", "100")
    max_seq_length = os.getenv("MAX_SEQ_LENGTH", "2048")
    logging_steps = os.getenv("LOGGING_STEPS", "10")
    save_steps = os.getenv("SAVE_STEPS", "500")
    eval_steps = os.getenv("EVAL_STEPS", "500")
    lora_r = os.getenv("LORA_R", "16")
    lora_alpha = os.getenv("LORA_ALPHA", "32")
    lora_dropout = os.getenv("LORA_DROPOUT", "0.05")
    seed = os.getenv("SEED", "42")
    resume_checkpoint = os.getenv("RESUME_FROM_CHECKPOINT", "")

    print(f"  Using Python: {venv_python}")
    print(f"  Working directory: {project_root}")
    print()
    print("Training Configuration:")
    print(f"  Model:           {model_name}")
    print(f"  Train data:      {train_data}")
    print(f"  Eval data:       {eval_data}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Epochs:          {epochs}")
    print(f"  Grad accum:      {grad_accum}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  LoRA r/alpha:    {lora_r}/{lora_alpha}")
    print(f"  Max seq length:  {max_seq_length}")
    print(f"  Save/eval steps: {save_steps}/{eval_steps}")
    print(f"  Resume from:     {resume_checkpoint if resume_checkpoint else 'None (fresh start)'}")
    print()

    # Verify training data exists
    train_path = project_root / train_data
    eval_path = project_root / eval_data
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("   Ensure convert_stcray_vlm job has completed successfully")
        sys.exit(1)
    if not eval_path.exists():
        print(f"Error: Evaluation data not found at {eval_path}")
        print("   Ensure convert_stcray_vlm job has completed successfully")
        sys.exit(1)

    # Count training samples
    with open(train_path) as f:
        n_train = sum(1 for _ in f)
    with open(eval_path) as f:
        n_eval = sum(1 for _ in f)
    print(f"  Training samples:   {n_train}")
    print(f"  Evaluation samples: {n_eval}")

    # Check GPU availability
    venv_env = get_venv_env(project_root)
    try:
        gpu_check = subprocess.run(
            [str(venv_python), "-c",
             "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); "
             "print(f'GPU count: {torch.cuda.device_count()}'); "
             "[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)') "
             "for i in range(torch.cuda.device_count())]"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            env=venv_env,
        )
        print(gpu_check.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not check GPU availability: {e}")

    print()
    print("Starting training...")
    print("-" * 60)

    # Build command
    cmd = [
        str(venv_python), "-u",
        "training/train_vlm_qlora.py",
        "--model-name", model_name,
        "--train-data", train_data,
        "--eval-data", eval_data,
        "--output-dir", output_dir,
        "--num-train-epochs", epochs,
        "--per-device-train-batch-size", batch_size,
        "--per-device-eval-batch-size", batch_size,
        "--gradient-accumulation-steps", grad_accum,
        "--learning-rate", learning_rate,
        "--warmup-steps", warmup_steps,
        "--max-seq-length", max_seq_length,
        "--logging-steps", logging_steps,
        "--save-steps", save_steps,
        "--eval-steps", eval_steps,
        "--lora-r", lora_r,
        "--lora-alpha", lora_alpha,
        "--lora-dropout", lora_dropout,
        "--seed", seed,
    ]

    # Add checkpoint resumption if specified
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            cmd.extend(["--resume-from-checkpoint", resume_checkpoint])
            print(f"  Resuming from checkpoint: {resume_checkpoint}")
        else:
            print(f"Warning: Checkpoint path not found: {resume_checkpoint}")
            print("  Starting fresh training instead")

    print(f"  Command: {' '.join(cmd)}")
    print()

    # Execute training with venv activated
    result = subprocess.run(cmd, cwd=str(project_root), env=venv_env)

    if result.returncode != 0:
        print(f"Error: Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Verify output
    print()
    print("-" * 60)
    print("Verifying training output...")
    output_path = Path(output_dir)
    if output_path.exists():
        checkpoints = sorted(output_path.glob("checkpoint-*"))
        if checkpoints:
            print(f"  Checkpoints saved: {len(checkpoints)}")
            print(f"  Latest: {checkpoints[-1].name}")
        final_dir = output_path / "final"
        if final_dir.exists():
            print(f"  Final model saved: {final_dir}")
        else:
            print("  Note: No 'final' directory found (check if training completed all epochs)")
    else:
        print(f"Warning: Output directory not found: {output_path}")

    print()
    print("Training job completed successfully")


if __name__ == "__main__":
    main()
