#!/usr/bin/env python3
"""
Prepare HiXray dataset for Phase B training (CAI wrapper).

Primary path: Extract pre-staged archive from /home/cdsw/data/hixray_raw/
Fallback: Attempt gdown download from Google Drive (unreliable from headless servers)

Environment Variables:
- HIXRAY_ARCHIVE: Path to HiXray archive directory (default: data/hixray_raw)
- OUTPUT_DIR:     Path for processed output (default: data/hixray_processed)
- FORCE_REPROCESS: Force re-extraction (default: false)
"""

import glob
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, "/home/cdsw/cai_integration")
from utils import PROJECT_ROOT, get_venv_python, run_in_venv, should_skip, write_done_marker


def main():
    print("=" * 60)
    print("Prepare HiXray Dataset")
    print("=" * 60)

    archive_dir = os.getenv("HIXRAY_ARCHIVE", "data/hixray_raw")
    output_dir = os.getenv("OUTPUT_DIR", "data/hixray_processed")

    archive_path = PROJECT_ROOT / archive_dir
    output_path = PROJECT_ROOT / output_dir

    print(f"  Archive dir: {archive_path}")
    print(f"  Output dir:  {output_path}")
    print()

    if should_skip("prepare_hixray", str(output_path)):
        return

    # Check for pre-staged archive
    zip_files = list(archive_path.glob("*.zip")) if archive_path.exists() else []
    tar_files = list(archive_path.glob("*.tar*")) if archive_path.exists() else []
    archives = zip_files + tar_files

    if not archives:
        print("  No pre-staged archive found. Attempting gdown fallback...")
        print()

        # Attempt gdown download
        archive_path.mkdir(parents=True, exist_ok=True)
        venv_python = get_venv_python()
        try:
            run_in_venv([
                venv_python, "-c",
                "import gdown; "
                "gdown.download("
                "'https://drive.google.com/uc?id=1jEk-h5Uv0-d3RdLf8cSHKXhuhalqD3l4', "
                f"'{archive_path}/hixray.zip')"
            ], check=False)
            archives = list(archive_path.glob("*.zip"))
        except Exception as e:
            print(f"  gdown fallback failed: {e}")

        if not archives:
            print()
            print("Error: No HiXray archive found.")
            print("  Please manually upload the HiXray archive to:")
            print(f"    {archive_path}/")
            print("  Then re-run this job.")
            sys.exit(1)

    print(f"  Found archive(s): {[a.name for a in archives]}")

    # Extract
    output_path.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f"  Extracting {archive.name}...")
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(output_path)
        else:
            import tarfile
            with tarfile.open(archive) as tf:
                tf.extractall(output_path)

    # Verify
    image_files = glob.glob(str(output_path / "**/*.jpg"), recursive=True)
    image_files += glob.glob(str(output_path / "**/*.png"), recursive=True)
    print(f"  Extracted {len(image_files)} images")

    if len(image_files) < 1000:
        print(f"  Warning: Expected ~45K images, found {len(image_files)}")
        print("  Archive may be incomplete or extraction path may be wrong")
    else:
        write_done_marker("prepare_hixray", str(output_path))
        print()
        print("HiXray dataset prepared successfully")


if __name__ == "__main__":
    main()
