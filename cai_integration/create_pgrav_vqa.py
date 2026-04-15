#!/usr/bin/env python3
"""
Generate dual-format PG-RAV VQA training data (CAI wrapper).

Runs class-agnostic YOLO on training images to collect proposals, then generates
dual-format VQA data: (1) full-scene with proposals, (2) focused ROI with category hints.

Environment Variables:
- YOLO_MODEL:      Path to class-agnostic YOLO model (default: runs/detect/class_agnostic_xray/weights/best.pt)
- STCRAY_DIR:      Path to processed STCray data (default: data/stcray_processed)
- HIXRAY_DIR:      Path to processed HiXray data (default: data/hixray_processed)
- OUTPUT_DIR:      Path for VQA JSONL output (default: data/pgrav_vqa)
- FORCE_REPROCESS: Force re-processing (default: false)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("/home/cdsw/cai_integration")))
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, should_skip, write_done_marker, validate_script_exists, check_output_exists


def main():
    print("=" * 60)
    print("Generate PG-RAV Dual-Format VQA Data")
    print("=" * 60)

    yolo_model = os.getenv("YOLO_MODEL", "runs/detect/class_agnostic_xray/weights/best.pt")
    stcray_dir = os.getenv("STCRAY_DIR", "data/stcray_processed")
    hixray_dir = os.getenv("HIXRAY_DIR", "data/hixray_processed")
    output_dir = os.getenv("OUTPUT_DIR", "data/pgrav_vqa")

    output_path = PROJECT_ROOT / output_dir

    print(f"  YOLO model:  {yolo_model}")
    print(f"  STCray dir:  {stcray_dir}")
    print(f"  HiXray dir:  {hixray_dir}")
    print(f"  Output dir:  {output_dir}")
    print()

    if should_skip("create_pgrav_vqa", str(output_path)):
        return

    # Validate YOLO model
    yolo_path = PROJECT_ROOT / yolo_model
    if not check_output_exists(str(yolo_path), min_size_mb=1.0):
        print(f"Error: YOLO model not found or too small: {yolo_path}")
        print("   Ensure train_yolo_class_agnostic job has completed")
        sys.exit(1)

    script = validate_script_exists("data/create_pgrav_vqa.py")
    venv_python = get_venv_python()

    cmd = [
        venv_python, "-u", str(script),
        "--yolo-model", yolo_model,
        "--stcray-dir", stcray_dir,
        "--hixray-dir", hixray_dir,
        "--output-dir", output_dir,
    ]
    run_in_venv(cmd)

    # Verify output
    train_jsonl = output_path / "pgrav_train.jsonl"
    if train_jsonl.exists():
        with open(train_jsonl) as f:
            n_lines = sum(1 for _ in f)
        print(f"  Training samples: {n_lines}")
        write_done_marker("create_pgrav_vqa", str(output_path))
        print("PG-RAV VQA data generated successfully")
    else:
        print(f"Error: Expected output not found: {train_jsonl}")
        sys.exit(1)


if __name__ == "__main__":
    main()
