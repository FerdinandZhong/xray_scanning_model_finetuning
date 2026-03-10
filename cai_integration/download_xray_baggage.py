#!/usr/bin/env python3
"""
CAI Job: Process X-Ray Baggage COCO dataset into YOLO format.

Extracts data/X-Ray Baggage.coco.zip and converts the COCO annotations
into YOLO format with a stratified train/val split.

The zip is checked into Git LFS and is already present in the workspace.
No network download is needed.

Output: data/xray_baggage_yolo/  (5 classes: Gun, Knife, Pliers, Scissors, Wrench)

Environment Variables:
- VAL_RATIO:        Fraction held out for validation (default: 0.1)
- FORCE_REPROCESS:  Set "true" to rebuild if output already exists (default: false)
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("X-Ray Baggage Dataset Processing Job")
    print("=" * 60)

    project_root = Path("/home/cdsw")
    venv_python  = project_root / ".venv/bin/python"

    if not venv_python.exists():
        print(f"❌ Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)

    zip_path = project_root / "data" / "X-Ray Baggage.coco.zip"
    if not zip_path.exists():
        print(f"❌ Dataset zip not found: {zip_path}")
        print("   Ensure 'git lfs pull' completed in setup_environment job")
        sys.exit(1)

    out_yaml = project_root / "data" / "xray_baggage_yolo" / "data.yaml"
    force    = os.getenv("FORCE_REPROCESS", "false").lower() == "true"

    if out_yaml.exists() and not force:
        from pathlib import Path as P
        tr = len(list((P(str(out_yaml.parent)) / "images" / "train").glob("*.jpg")))
        va = len(list((P(str(out_yaml.parent)) / "images" / "valid").glob("*.jpg")))
        print(f"⚠  xray_baggage_yolo already exists: {tr:,} train / {va:,} val")
        print("   Set FORCE_REPROCESS=true to rebuild.")
        print()
        print("✅ Using existing xray_baggage_yolo dataset.")
        return

    print(f"✓ Python:   {venv_python}")
    print(f"✓ Zip:      {zip_path}")
    print()

    cmd = [
        str(venv_python),
        "data/process_xray_baggage.py",
    ]

    env = os.environ.copy()
    env["VAL_RATIO"]       = os.getenv("VAL_RATIO", "0.1")
    env["FORCE_REPROCESS"] = os.getenv("FORCE_REPROCESS", "false")

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(project_root), env=env)

    if result.returncode != 0:
        print()
        print("❌ X-Ray Baggage processing failed!")
        sys.exit(1)

    print()
    print("=" * 60)
    print("✅ X-Ray Baggage dataset ready at data/xray_baggage_yolo/")
    print("=" * 60)


if __name__ == "__main__":
    main()
