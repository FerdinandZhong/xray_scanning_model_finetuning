#!/usr/bin/env python3
"""
CAI Job: Prepare STCray dataset from Git LFS tar.gz archives.

The archives (STCray_TrainSet.tar.gz, STCray_TestSet.tar.gz) are stored in
the repo via Git LFS and pulled automatically by setup_environment.sh
(git lfs pull).  Extraction uses Python's built-in tarfile module — no
system packages required.

This job:
  1. Verifies the tar.gz files are present under data/stcray_raw/
  2. Extracts them using Python tarfile (no external tools needed)
  3. Runs data/process_stcray_raw.py to build stcray_processed/annotations.json
"""

import subprocess
import sys
import tarfile
from pathlib import Path


def extract_tar(tar_path: Path, output_dir: Path) -> bool:
    """Extract a tar.gz archive using Python's built-in tarfile module."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting: {tar_path.name} → {output_dir} ...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"  ✓ Extracted: {tar_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def main():
    print("=" * 60)
    print("STCray Dataset Preparation Job")
    print("=" * 60)

    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"
    raw_dir = project_root / "data" / "stcray_raw"
    processed_dir = project_root / "data" / "stcray_processed"

    if not venv_python.exists():
        print(f"❌ Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)

    # ── 1. Verify tar.gz archives are present (pulled from Git LFS) ──
    print("\n[1/3] Checking tar.gz archives from Git LFS...")
    archives = {
        "train": raw_dir / "STCray_TrainSet.tar.gz",
        "test":  raw_dir / "STCray_TestSet.tar.gz",
    }
    for split, archive_path in archives.items():
        if not archive_path.exists():
            print(f"❌ Archive not found: {archive_path}")
            print("   Run 'git lfs pull' to download it from the repository.")
            sys.exit(1)
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {archive_path.name}  ({size_mb:.0f} MB)")

    # ── 2. Extract archives (skip if already extracted) ───────────
    print("\n[2/3] Extracting archives...")
    for split, archive_path in archives.items():
        folder_name = "STCray_TrainSet" if split == "train" else "STCray_TestSet"
        extract_target = raw_dir / folder_name
        images_dir = extract_target / "Images"

        if images_dir.exists() and any(images_dir.rglob("*.jpg")):
            img_count = sum(1 for _ in images_dir.rglob("*.jpg"))
            print(f"  ✓ {folder_name} already extracted ({img_count:,} images) — skipping")
            continue

        if not extract_tar(archive_path, raw_dir):
            sys.exit(1)

    # ── 3. Build stcray_processed/annotations.json ────────────────
    print("\n[3/3] Building annotations from extracted dataset...")

    train_ann = processed_dir / "train" / "annotations.json"
    test_ann  = processed_dir / "test"  / "annotations.json"

    if train_ann.exists() and test_ann.exists():
        import json
        n_train = len(json.load(open(train_ann)))
        n_test  = len(json.load(open(test_ann)))
        print(f"  ✓ Annotations already exist — train: {n_train:,}  test: {n_test:,}")
        print("    Set FORCE_REPROCESS=true to rebuild.")
        import os
        if os.environ.get("FORCE_REPROCESS", "false").lower() != "true":
            _print_summary(processed_dir)
            return

    result = subprocess.run(
        [
            str(venv_python),
            "data/process_stcray_raw.py",
            "--raw-dir",     "data/stcray_raw",
            "--output-dir",  "data/stcray_processed",
            "--project-root", str(project_root),
        ],
        cwd=str(project_root),
    )

    if result.returncode != 0:
        print("❌ Annotation processing failed!")
        sys.exit(1)

    _print_summary(processed_dir)


def _print_summary(processed_dir: Path):
    print()
    print("=" * 60)
    print("✓ STCray dataset ready")
    print("=" * 60)
    print(f"  Train annotations: {processed_dir}/train/annotations.json")
    print(f"  Test  annotations: {processed_dir}/test/annotations.json")
    print()
    print("Next job: combine_datasets")


if __name__ == "__main__":
    main()
