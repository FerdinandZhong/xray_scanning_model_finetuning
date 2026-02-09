#!/usr/bin/env python3
"""
Generate VQA dataset using external vLLM server.

This script wraps data/llm_vqa_generator.py for CAI job execution.
It connects to an external Qwen2.5-VL vLLM server to generate VQA pairs.

Environment Variables:
- VLLM_API_BASE: External vLLM server endpoint (required)
- MODEL_NAME: Model name (default: Qwen/Qwen2.5-VL-7B-Instruct)
- SAMPLES_PER_IMAGE: Number of VQA pairs per image (default: 3)
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Execute VQA dataset generation."""
    print("=" * 60)
    print("VQA Dataset Generation Job")
    print("=" * 60)
    
    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"
    
    # Verify venv exists
    if not venv_python.exists():
        print(f"❌ Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)
    
    # Get configuration from environment
    api_base = os.getenv("VLLM_API_BASE")
    model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    samples_per_image = int(os.getenv("SAMPLES_PER_IMAGE", "3"))
    api_key = os.getenv("API_KEY", "")
    
    if not api_base:
        print("❌ Error: VLLM_API_BASE environment variable not set")
        print("   Configure this in jobs_config.yaml:")
        print("   environment:")
        print('     VLLM_API_BASE: "http://your-vllm-server:8000/v1"')
        sys.exit(1)
    
    print(f"✓ Using Python: {venv_python}")
    print(f"✓ Working directory: {project_root}")
    print()
    print(f"VQA Generation Configuration:")
    print(f"  vLLM API Base: {api_base}")
    print(f"  Model: {model}")
    print(f"  Samples per image: {samples_per_image}")
    print(f"  API Key: {'<configured>' if api_key else '<not set>'}")
    print()
    
    # Verify dataset exists
    train_annotations = project_root / "data/stcray/train/annotations.json"
    test_annotations = project_root / "data/stcray/test/annotations.json"
    
    if not train_annotations.exists():
        print(f"❌ Error: Training annotations not found at {train_annotations}")
        print("   Ensure download_dataset job has completed successfully")
        sys.exit(1)
    
    print(f"✓ Dataset verified")
    print()
    
    # Set OpenAI API environment variables for vLLM/OpenAI/Claude
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = api_base
    # Use provided API key or "EMPTY" for public vLLM servers
    env["OPENAI_API_KEY"] = api_key if api_key else "EMPTY"
    
    # Also set for Anthropic if using Claude
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    
    # Generate training set
    print("=" * 60)
    print("Step 1: Generating Training VQA Dataset")
    print("=" * 60)
    print("This may take 2-3 hours for ~30k images...")
    print()
    
    result = subprocess.run(
        [
            str(venv_python),
            "data/llm_vqa_generator.py",
            "--annotations", "data/stcray/train/annotations.json",
            "--images-dir", "data/stcray/train/images",
            "--output", "data/stcray_vqa_train.jsonl",
            "--model", model,
            "--samples-per-image", str(samples_per_image),
            "--rate-limit-delay", "0.1",
            "--batch-save", "100",
        ],
        cwd=str(project_root),
        env=env
    )
    
    if result.returncode != 0:
        print(f"❌ Training VQA generation failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print()
    print("✓ Training VQA dataset generated")
    print()
    
    # Generate validation set
    print("=" * 60)
    print("Step 2: Generating Validation VQA Dataset")
    print("=" * 60)
    print("This may take 1-2 hours for ~16k images...")
    print()
    
    result = subprocess.run(
        [
            str(venv_python),
            "data/llm_vqa_generator.py",
            "--annotations", "data/stcray/test/annotations.json",
            "--images-dir", "data/stcray/test/images",
            "--output", "data/stcray_vqa_val.jsonl",
            "--model", model,
            "--samples-per-image", str(samples_per_image),
            "--rate-limit-delay", "0.1",
            "--batch-save", "100",
        ],
        cwd=str(project_root),
        env=env
    )
    
    if result.returncode != 0:
        print(f"❌ Validation VQA generation failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print()
    print("=" * 60)
    print("✓ VQA Dataset Generation Complete")
    print("=" * 60)
    print()
    print("Files created:")
    print(f"  - Training: /home/cdsw/data/stcray_vqa_train.jsonl")
    print(f"  - Validation: /home/cdsw/data/stcray_vqa_val.jsonl")
    print()
    print("Next job: finetune_model")


if __name__ == "__main__":
    main()
