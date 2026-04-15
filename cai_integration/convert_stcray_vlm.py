#!/usr/bin/env python3
"""
Convert STCray annotations to VQA-style JSONL for VLM training (CAI wrapper).

This script wraps data/convert_stcray_to_vlm.py for CAI job execution.
All configuration is read from environment variables.

Environment Variables:
- INPUT_DIR:  Path to processed STCray data (default: data/stcray_processed)
- OUTPUT_DIR: Path for VQA JSONL output (default: data/stcray_vlm)
- VAL_RATIO:  Validation split ratio (default: 0.1)
- SEED:       Random seed (default: 42)
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("/home/cdsw/cai_integration")))
from utils import get_venv_env


def main():
    """Execute STCray to VLM format conversion."""
    print("=" * 60)
    print("STCray to VLM Format Conversion Job")
    print("=" * 60)

    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"

    # Verify venv exists
    if not venv_python.exists():
        print(f"Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)

    # Get configuration from environment
    input_dir = os.getenv("INPUT_DIR", "data/stcray_processed")
    output_dir = os.getenv("OUTPUT_DIR", "data/stcray_vlm")
    val_ratio = os.getenv("VAL_RATIO", "0.1")
    seed = os.getenv("SEED", "42")

    print(f"  Using Python: {venv_python}")
    print(f"  Working directory: {project_root}")
    print()
    print("Conversion Configuration:")
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Validation ratio: {val_ratio}")
    print(f"  Seed:             {seed}")
    print()

    # Verify input directory exists
    input_path = project_root / input_dir
    if not input_path.exists():
        print(f"Error: Input directory not found at {input_path}")
        print("   Ensure download_dataset job has completed successfully")
        sys.exit(1)

    print(f"  Input data verified: {input_path}")

    # Check if output already exists (skip if already converted)
    output_path = project_root / output_dir
    train_jsonl = output_path / "stcray_vlm_train.jsonl"
    if train_jsonl.exists():
        print(f"  Output already exists: {train_jsonl}")
        print("  Skipping conversion (set FORCE_REPROCESS=true to override)")
        if os.getenv("FORCE_REPROCESS", "false").lower() != "true":
            return
        print("  FORCE_REPROCESS=true, re-running conversion")

    print()
    print("Starting conversion...")
    print("-" * 60)

    # Build command
    cmd = [
        str(venv_python),
        "data/convert_stcray_to_vlm.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--val-ratio", val_ratio,
        "--seed", seed,
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()

    # Execute conversion with venv activated
    result = subprocess.run(cmd, cwd=str(project_root), env=get_venv_env(project_root))

    if result.returncode != 0:
        print(f"Error: Conversion failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Verify output
    print()
    print("-" * 60)
    print("Verifying output files...")
    expected_files = [
        output_path / "stcray_vlm_train.jsonl",
        output_path / "stcray_vlm_test.jsonl",
    ]
    for f in expected_files:
        if f.exists():
            # Count lines
            with open(f) as fh:
                n_lines = sum(1 for _ in fh)
            print(f"  {f.name}: {n_lines} samples")
        else:
            print(f"  Warning: Expected file not found: {f}")

    val_file = output_path / "stcray_vlm_val.jsonl"
    if val_file.exists():
        with open(val_file) as fh:
            n_lines = sum(1 for _ in fh)
        print(f"  {val_file.name}: {n_lines} samples")

    print()
    print("Conversion completed successfully")


if __name__ == "__main__":
    main()
