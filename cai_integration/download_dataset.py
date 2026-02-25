#!/usr/bin/env python3
"""
CAI Job: Prepare STCray dataset from Git LFS RAR files.

The RAR files (STCray_TrainSet.rar, STCray_TestSet.rar) are stored in the
repo via Git LFS and pulled automatically by setup_environment.sh (git lfs pull).

This job:
  1. Verifies the RAR files are present under data/stcray_raw/
  2. Extracts them with unrar (installed by setup_environment.sh)
  3. Runs data/process_stcray_raw.py to build stcray_processed/annotations.json
"""

import subprocess
import sys
from pathlib import Path


def extract_rar(rar_path: Path, output_dir: Path, venv_python: Path) -> bool:
    """Extract a RAR file using unrar (with 7z as fallback)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for cmd in [
        ["unrar", "x", "-o+", str(rar_path), str(output_dir) + "/"],
        ["7z", "x", str(rar_path), f"-o{output_dir}", "-y"],
    ]:
        tool = cmd[0]
        check = subprocess.run(["which", tool], capture_output=True)
        if check.returncode != 0:
            print(f"  {tool} not found, trying next...")
            continue

        print(f"  Extracting with {tool}: {rar_path.name} ...")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"  ✓ Extracted: {rar_path.name}")
            return True
        else:
            print(f"  ✗ {tool} extraction failed (exit {result.returncode})")

    print(f"❌ Could not extract {rar_path.name} — install unrar or p7zip-full")
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

    # ── 1. Verify RAR files are present (pulled from Git LFS) ─────
    print("\n[1/3] Checking RAR files from Git LFS...")
    rar_files = {
        "train": raw_dir / "STCray_TrainSet.rar",
        "test":  raw_dir / "STCray_TestSet.rar",
    }
    for split, rar_path in rar_files.items():
        if not rar_path.exists():
            print(f"❌ RAR not found: {rar_path}")
            print("   Run 'git lfs pull' to download it from the repository.")
            sys.exit(1)
        size_mb = rar_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {rar_path.name}  ({size_mb:.0f} MB)")

    # ── 2. Extract RAR files (skip if already extracted) ──────────
    print("\n[2/3] Extracting RAR files...")
    for split, rar_path in rar_files.items():
        folder_name = "STCray_TrainSet" if split == "train" else "STCray_TestSet"
        extract_target = raw_dir / folder_name
        images_dir = extract_target / "Images"

        if images_dir.exists() and any(images_dir.rglob("*.jpg")):
            img_count = sum(1 for _ in images_dir.rglob("*.jpg"))
            print(f"  ✓ {folder_name} already extracted ({img_count:,} images) — skipping")
            continue

        if not extract_rar(rar_path, raw_dir, venv_python):
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
