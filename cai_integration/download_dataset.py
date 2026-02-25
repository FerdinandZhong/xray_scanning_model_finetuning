#!/usr/bin/env python3
"""
Download STCray dataset in CAI environment.

This script wraps data/download_stcray.py for CAI job execution.
It downloads the STCray dataset from HuggingFace to /home/cdsw/data/stcray/.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Execute dataset download."""
    print("=" * 60)
    print("STCray Dataset Download Job")
    print("=" * 60)

    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"

    # Verify venv exists
    if not venv_python.exists():
        print(f"❌ Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)

    # Validate HuggingFace token
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable is not set")
        print("   STCray is a gated dataset — a HuggingFace token with access is required.")
        print("   Set HF_TOKEN in the CAI job environment variables.")
        sys.exit(1)
    print("✓ HuggingFace token found")

    print(f"✓ Using Python: {venv_python}")
    print(f"✓ Working directory: {project_root}")
    print()

    # Forward HF_TOKEN so the subprocess can authenticate with HuggingFace
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token

    # Run download script
    print("Downloading STCray dataset from HuggingFace...")
    result = subprocess.run(
        [
            str(venv_python),
            "data/download_stcray.py",
            "--output-dir", "data/stcray",
        ],
        cwd=str(project_root),
        env=env,
    )
    
    if result.returncode != 0:
        print(f"❌ Dataset download failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print()
    print("=" * 60)
    print("✓ Dataset downloaded successfully")
    print("=" * 60)
    print()
    print("Dataset location:")
    print("  - Training: /home/cdsw/data/stcray/train/")
    print("  - Test: /home/cdsw/data/stcray/test/")
    print()
    print("Next job: generate_vqa")
    # Success - exit normally without sys.exit(0) for CAI compatibility


if __name__ == "__main__":
    main()
