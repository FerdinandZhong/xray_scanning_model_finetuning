#!/usr/bin/env python3
"""
CAI Job: Combine X-ray Datasets for Object Detection Training.

Merges luggage_xray_yolo (6,164 images, 12 classes) with STCray
(30,044 images, 22 object classes) into a unified 26-class YOLO dataset
covering all detectable objects in X-ray baggage scans.

Environment Variables:
- OUTPUT_NAME:        Output dataset name under data/ (default: combined_xray_yolo)
- MAX_PER_CLASS:      Cap STCray images per mapped class, 0 = no cap (default: 0)
- FORCE_REBUILD:      Delete and rebuild if dataset already exists (default: false)
- INCLUDE_LUGGAGE:    Include luggage_xray_yolo dataset (default: true)
- INCLUDE_STCRAY:     Include STCray dataset (default: true)
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 65)
    print("X-ray Dataset Combination Job")
    print("=" * 65)

    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"

    if not venv_python.exists():
        print(f"❌ Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)

    # ── Read configuration from environment ──────────────────────
    output_name   = os.getenv("OUTPUT_NAME",   "combined_xray_yolo")
    max_per_class = int(os.getenv("MAX_PER_CLASS", "0"))
    force_rebuild = os.getenv("FORCE_REBUILD", "false").lower() == "true"
    inc_luggage   = os.getenv("INCLUDE_LUGGAGE", "true").lower() == "true"
    inc_stcray    = os.getenv("INCLUDE_STCRAY",  "true").lower() == "true"

    output_dir = project_root / "data" / output_name

    print(f"\nConfiguration:")
    print(f"  Output dataset:    data/{output_name}")
    print(f"  Max per class:     {'no cap' if max_per_class == 0 else max_per_class}")
    print(f"  Force rebuild:     {force_rebuild}")
    print(f"  Include luggage:   {inc_luggage}")
    print(f"  Include STCray:    {inc_stcray}")
    print()

    # ── Check if output already exists ───────────────────────────
    data_yaml = output_dir / "data.yaml"
    if data_yaml.exists() and not force_rebuild:
        train_count = len(list((output_dir / "images" / "train").glob("*.jpg")))
        val_count   = len(list((output_dir / "images" / "valid").glob("*.jpg")))
        print(f"⚠  Dataset already exists at: {output_dir}")
        print(f"   Train: {train_count:,} images  Val: {val_count:,} images")
        print(f"   Set FORCE_REBUILD=true to rebuild from scratch.")
        print()
        print("✅ Using existing combined dataset. Skipping combination step.")
        _print_next_steps(output_name)
        return 0

    # ── Verify source datasets are present ───────────────────────
    print("Checking source datasets...")

    if inc_luggage:
        lug_dir = project_root / "data" / "luggage_xray_yolo"
        if not (lug_dir / "data.yaml").exists():
            print(f"❌ luggage_xray_yolo not found at {lug_dir}")
            print("   Run the 'download_luggage_xray' job first.")
            sys.exit(1)
        lug_train = len(list((lug_dir / "images" / "train").glob("*.jpg")))
        print(f"  ✓ luggage_xray_yolo:  {lug_train:,} train images")

    if inc_stcray:
        stcray_ann = project_root / "data" / "stcray_processed" / "train" / "annotations.json"
        stcray_raw = project_root / "data" / "stcray_raw"
        if not stcray_ann.exists():
            print(f"❌ STCray annotations not found at {stcray_ann}")
            print("   Run the 'download_dataset' job first.")
            sys.exit(1)
        stc_imgs = len(list(stcray_raw.rglob("*.jpg")))
        print(f"  ✓ stcray_processed:   {stc_imgs:,} raw images")

    print()

    # ── Build command ─────────────────────────────────────────────
    cmd = [
        str(venv_python),
        "scripts/combine_xray_datasets.py",
        "--output", output_name,
    ]

    if max_per_class > 0:
        cmd += ["--max-per-class", str(max_per_class)]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(project_root), env=os.environ.copy())

    if result.returncode != 0:
        print()
        print("❌ Dataset combination failed!")
        sys.exit(1)

    # ── Verify output ─────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Verifying output...")

    train_imgs  = list((output_dir / "images" / "train").glob("*.jpg"))
    val_imgs    = list((output_dir / "images" / "valid").glob("*.jpg"))
    train_lbls  = list((output_dir / "labels" / "train").glob("*.txt"))
    val_lbls    = list((output_dir / "labels" / "valid").glob("*.txt"))

    print(f"  data.yaml:      {'✓' if data_yaml.exists() else '✗ MISSING'}")
    print(f"  Train images:   {len(train_imgs):,}")
    print(f"  Train labels:   {len(train_lbls):,}")
    print(f"  Val   images:   {len(val_imgs):,}")
    print(f"  Val   labels:   {len(val_lbls):,}")

    img_lbl_match = len(train_imgs) == len(train_lbls) and len(val_imgs) == len(val_lbls)
    if not img_lbl_match:
        print("⚠  Image/label count mismatch — some images may have been skipped")

    total_size_mb = sum(f.stat().st_size for f in train_imgs + val_imgs) / (1024 * 1024)
    print(f"  Dataset size:   {total_size_mb:.0f} MB")

    print()
    print("=" * 65)
    print("✅ Dataset combination completed successfully!")
    print("=" * 65)

    _print_next_steps(output_name)
    return 0


def _print_next_steps(output_name: str):
    print()
    print("Next steps:")
    print(f"  1. Trigger YOLO training on combined dataset:")
    print(f"     In jobs_config_yolo.yaml set:")
    print(f"       DATASET: \"{output_name}\"")
    print()
    print(f"  2. Or train locally:")
    print(f"     python training/train_yolo.py \\")
    print(f"       --data data/{output_name}/data.yaml \\")
    print(f"       --model yolov8m.pt --epochs 200 --batch 16")
    print()


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        sys.exit(exit_code)
