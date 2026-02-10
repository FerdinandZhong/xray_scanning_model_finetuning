#!/usr/bin/env python3
"""
Setup Python environment for X-ray VQA fine-tuning in CAI.

This script is the Python entry point for the CAI job that:
1. Wraps the bash script setup_environment.sh
2. Handles exit codes and error reporting

The actual setup logic is in setup_environment.sh which:
1. Checks if venv exists (reuses if yes, creates if no)
2. Installs uv (ultra-fast Python package installer, 10-100x faster than pip)
3. Installs PyTorch, Transformers, PEFT, and other dependencies using uv
4. Verifies installation

Benefits of uv:
- 10-100x faster than pip for package installation
- Better dependency resolution
- Parallel downloads with global cache
- Backward compatible with pip (uses same requirements.txt)
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Execute environment setup via bash script."""
    # Get paths
    project_root = Path("/home/cdsw")
    bash_script = project_root / "cai_integration" / "setup_environment.sh"

    # Ensure script is executable
    bash_script.chmod(0o755)

    # Run bash script
    print(f"Executing setup script: {bash_script}")
    result = subprocess.run(
        ["bash", str(bash_script)],
        cwd=str(project_root)
    )

    if result.returncode != 0:
        raise RuntimeError(f"Setup script failed with exit code {result.returncode}")

    print("Setup completed successfully")


if __name__ == "__main__":
    main()
