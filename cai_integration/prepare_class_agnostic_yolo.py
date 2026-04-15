#!/usr/bin/env python3
"""
Prepare class-agnostic YOLO dataset from STCray + HiXray (CAI wrapper).

Merges all bounding box annotations from both datasets into a single "object" class
for training a localization-only YOLO detector.

Environment Variables:
- STCRAY_DIR:      Path to processed STCray data (default: data/stcray_processed)
- HIXRAY_DIR:      Path to processed HiXray data (default: data/hixray_processed)
- OUTPUT_DIR:      Path for YOLO-format output (default: data/class_agnostic_yolo)
- FORCE_REPROCESS: Force re-processing (default: false)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("/home/cdsw/cai_integration")))
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, should_skip, write_done_marker, validate_script_exists


def main():
    print("=" * 60)
    print("Prepare Class-Agnostic YOLO Dataset")
    print("=" * 60)

    stcray_dir = os.getenv("STCRAY_DIR", "data/stcray_processed")
    hixray_dir = os.getenv("HIXRAY_DIR", "data/hixray_processed")
    output_dir = os.getenv("OUTPUT_DIR", "data/class_agnostic_yolo")

    output_path = PROJECT_ROOT / output_dir

    print(f"  STCray dir:  {stcray_dir}")
    print(f"  HiXray dir:  {hixray_dir}")
    print(f"  Output dir:  {output_dir}")
    print()

    if should_skip("prepare_class_agnostic_yolo", str(output_path)):
        return

    # Validate inputs
    stcray_path = PROJECT_ROOT / stcray_dir
    hixray_path = PROJECT_ROOT / hixray_dir
    if not stcray_path.exists():
        print(f"Error: STCray data not found at {stcray_path}")
        sys.exit(1)
    if not hixray_path.exists():
        print(f"Error: HiXray data not found at {hixray_path}")
        print("   Ensure prepare_hixray job has completed successfully")
        sys.exit(1)

    script = validate_script_exists("scripts/prepare_class_agnostic_yolo.py")
    venv_python = get_venv_python()

    cmd = [
        venv_python, "-u", str(script),
        "--stcray-dir", stcray_dir,
        "--hixray-dir", hixray_dir,
        "--output-dir", output_dir,
    ]
    run_in_venv(cmd)

    # Verify
    data_yaml = output_path / "data.yaml"
    if data_yaml.exists():
        write_done_marker("prepare_class_agnostic_yolo", str(output_path))
        print("Class-agnostic YOLO dataset prepared successfully")
    else:
        print(f"Error: Expected {data_yaml} not found after processing")
        sys.exit(1)


if __name__ == "__main__":
    main()
