#!/usr/bin/env python3
"""
Shared utilities for CAI wrapper scripts.

Provides common helpers for venv activation, subprocess execution,
and idempotency guards. All CAI wrappers should import from here
instead of duplicating these functions.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path("/home/cdsw")
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin/python"


def get_venv_env(project_root: Optional[Path] = None) -> dict:
    """Build subprocess environment with venv properly activated."""
    root = project_root or PROJECT_ROOT
    venv_dir = root / ".venv"
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"
    env.pop("PYTHONHOME", None)
    return env


def get_venv_python(project_root: Optional[Path] = None) -> Path:
    """Return venv python path, exit if venv doesn't exist."""
    root = project_root or PROJECT_ROOT
    venv_python = root / ".venv/bin/python"
    if not venv_python.exists():
        print(f"Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)
    return venv_python


def run_in_venv(cmd: list, project_root: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command using the venv python with unbuffered output."""
    root = project_root or PROJECT_ROOT
    env = get_venv_env(root)
    print(f"  Command: {' '.join(str(c) for c in cmd)}")
    print()
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(root),
        env=env,
    )
    if check and result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def check_output_exists(path: str, min_size_mb: float = 0) -> bool:
    """Check if output file/dir exists, optionally with minimum size."""
    p = Path(path)
    if not p.exists():
        return False
    if min_size_mb > 0 and p.is_file():
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb < min_size_mb:
            print(f"  Warning: {p} exists but is only {size_mb:.1f}MB (min: {min_size_mb}MB)")
            return False
    return True


def write_done_marker(job_name: str, output_dir: str):
    """Write a .done marker file after successful job completion."""
    marker = Path(output_dir) / f".{job_name}.done"
    marker.write_text(json.dumps({
        "job": job_name,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    print(f"  Done marker written: {marker}")


def check_done_marker(job_name: str, output_dir: str) -> bool:
    """Check if a .done marker exists for this job."""
    marker = Path(output_dir) / f".{job_name}.done"
    return marker.exists()


def should_skip(job_name: str, output_dir: str, force_env: str = "FORCE_REPROCESS") -> bool:
    """Check if job should be skipped (output exists + done marker + not forced)."""
    if os.getenv(force_env, "false").lower() == "true":
        print(f"  {force_env}=true, running job regardless")
        return False
    if check_done_marker(job_name, output_dir):
        print(f"  Job '{job_name}' already completed (done marker found). Skipping.")
        print(f"  Set {force_env}=true to force re-run")
        return True
    return False


def check_gpu():
    """Print GPU availability info."""
    venv_python = get_venv_python()
    try:
        result = subprocess.run(
            [str(venv_python), "-c",
             "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); "
             "print(f'GPU count: {torch.cuda.device_count()}'); "
             "[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} "
             "({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)') "
             "for i in range(torch.cuda.device_count())]"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=get_venv_env(),
        )
        print(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not check GPU availability: {e}")


def validate_script_exists(script_path: str):
    """Validate that the target script exists, exit with helpful message if not."""
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        print(f"Error: Script not found at {full_path}")
        print(f"   This script needs to be implemented before this CAI job can run.")
        print(f"   Expected path: {script_path}")
        sys.exit(1)
    return full_path
