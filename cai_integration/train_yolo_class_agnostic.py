#!/usr/bin/env python3
"""
Train class-agnostic YOLO detector on merged STCray+HiXray (CAI wrapper).

Trains YOLOv11-nano to detect generic "objectness" in X-ray imagery.
This is a localization-only model — classification is the VLM's job.

Environment Variables:
- DATA_YAML:   Path to class-agnostic data.yaml (default: data/class_agnostic_yolo/data.yaml)
- MODEL_NAME:  YOLO model variant (default: yolo11n.pt)
- EPOCHS:      Max training epochs (default: 100)
- BATCH_SIZE:  Batch size (default: 32)
- IMG_SIZE:    Input image size (default: 640)
- PATIENCE:    Early stopping patience (default: 15)
- PROJECT:     Output project dir (default: runs/detect)
- RUN_NAME:    Run name (default: class_agnostic_xray)
- FORCE_REPROCESS: Force re-training (default: false)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/cdsw/cai_integration")
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, check_output_exists, should_skip, write_done_marker


def main():
    print("=" * 60)
    print("Train Class-Agnostic YOLO Detector")
    print("=" * 60)

    data_yaml = os.getenv("DATA_YAML", "data/class_agnostic_yolo/data.yaml")
    model_name = os.getenv("MODEL_NAME", "yolo11n.pt")
    epochs = os.getenv("EPOCHS", "100")
    batch_size = os.getenv("BATCH_SIZE", "32")
    img_size = os.getenv("IMG_SIZE", "640")
    patience = os.getenv("PATIENCE", "15")
    project = os.getenv("PROJECT", "runs/detect")
    run_name = os.getenv("RUN_NAME", "class_agnostic_xray")

    weights_dir = PROJECT_ROOT / project / run_name / "weights"
    best_pt = weights_dir / "best.pt"

    print(f"  Data YAML:   {data_yaml}")
    print(f"  Model:       {model_name}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Output:      {project}/{run_name}")
    print()

    # Idempotency: check if best.pt exists and is > 1MB
    if not os.getenv("FORCE_REPROCESS", "false").lower() == "true":
        if check_output_exists(str(best_pt), min_size_mb=1.0):
            print(f"  Trained model already exists: {best_pt}")
            print("  Skipping training. Set FORCE_REPROCESS=true to retrain.")
            return

    # Validate data.yaml
    data_yaml_path = PROJECT_ROOT / data_yaml
    if not data_yaml_path.exists():
        print(f"Error: Data YAML not found at {data_yaml_path}")
        print("   Ensure prepare_class_agnostic_yolo job has completed")
        sys.exit(1)

    from utils import check_gpu
    check_gpu()
    print()

    venv_python = get_venv_python()
    cmd = [
        venv_python, "-u",
        "training/train_yolo.py",
        "--data", data_yaml,
        "--model", model_name,
        "--epochs", epochs,
        "--batch", batch_size,
        "--imgsz", img_size,
        "--patience", patience,
        "--project", project,
        "--name", run_name,
    ]

    print("Starting class-agnostic YOLO training...")
    print("-" * 60)
    run_in_venv(cmd)

    # Verify
    if check_output_exists(str(best_pt), min_size_mb=1.0):
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"  Best model: {best_pt} ({size_mb:.1f} MB)")
        write_done_marker("train_yolo_class_agnostic", str(weights_dir.parent))
        print("Class-agnostic YOLO training completed successfully")
    else:
        print(f"Error: best.pt not found or too small at {best_pt}")
        sys.exit(1)


if __name__ == "__main__":
    main()
