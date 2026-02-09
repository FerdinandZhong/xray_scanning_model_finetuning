#!/usr/bin/env python3
"""
Download and extract OPIXray dataset for X-ray baggage scanning.
Dataset: https://github.com/OPIXray-author/OPIXray
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, Request


def check_git_installed():
    """Check if git is installed."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_file(url, output_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urlopen(request) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 8192
        downloaded = 0
        
        with open(output_path, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end="", flush=True)
    
    print("\nDownload complete.")


def clone_repository(repo_url, output_dir):
    """Clone git repository."""
    print(f"Cloning repository from {repo_url}...")
    subprocess.run(["git", "clone", repo_url, str(output_dir)], check=True)
    print("Repository cloned successfully.")


def verify_checksum(file_path, expected_hash=None):
    """Verify file checksum (optional)."""
    if expected_hash is None:
        return True
    
    print(f"Verifying checksum for {file_path}...")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    actual_hash = sha256_hash.hexdigest()
    if actual_hash != expected_hash:
        print(f"Checksum mismatch! Expected: {expected_hash}, Got: {actual_hash}")
        return False
    
    print("Checksum verified.")
    return True


def download_opixray(output_dir, method="manual"):
    """
    Download OPIXray dataset.
    
    Methods:
    - manual: Provide instructions for manual download
    - git: Clone from GitHub (if available)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("OPIXray Dataset Download")
    print("=" * 60)
    
    if method == "git" and check_git_installed():
        repo_url = "https://github.com/OPIXray-author/OPIXray.git"
        try:
            clone_repository(repo_url, output_path / "OPIXray_repo")
            print(f"\nDataset repository downloaded to: {output_path / 'OPIXray_repo'}")
            print("\nNote: The actual dataset images may need to be downloaded separately.")
            print("Please check the repository README for dataset download links.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed: {e}")
            method = "manual"
    
    if method == "manual":
        print("\nOPIXray Dataset Manual Download Instructions:")
        print("-" * 60)
        print("1. Visit the OPIXray repository:")
        print("   https://github.com/OPIXray-author/OPIXray")
        print("\n2. Download the dataset files:")
        print("   - OPIXray images (8,885 X-ray images)")
        print("   - Annotations (COCO format JSON files)")
        print("\n3. Dataset structure should be:")
        print(f"   {output_path}/")
        print("   ├── images/")
        print("   │   ├── P00001.jpg")
        print("   │   ├── P00002.jpg")
        print("   │   └── ...")
        print("   └── annotations/")
        print("       ├── train.json")
        print("       ├── val.json")
        print("       └── test.json")
        print("\n4. Alternative download sources:")
        print("   - Google Drive link (check repository README)")
        print("   - Baidu Pan link (check repository README)")
        print("-" * 60)
        
        # Create directory structure
        (output_path / "images").mkdir(exist_ok=True)
        (output_path / "annotations").mkdir(exist_ok=True)
        
        print(f"\nDirectory structure created at: {output_path}")
        print("Please download the dataset files into the appropriate directories.")
        
        return False


def verify_dataset_structure(dataset_dir):
    """Verify that the dataset has been downloaded correctly."""
    dataset_path = Path(dataset_dir)
    
    print("\nVerifying dataset structure...")
    
    required_dirs = ["images", "annotations"]
    required_files = [
        "annotations/train.json",
        "annotations/val.json",
        "annotations/test.json",
    ]
    
    missing = []
    
    for dir_name in required_dirs:
        if not (dataset_path / dir_name).exists():
            missing.append(f"Directory: {dir_name}")
    
    for file_path in required_files:
        if not (dataset_path / file_path).exists():
            missing.append(f"File: {file_path}")
    
    # Check for images
    images_dir = dataset_path / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        print(f"Found {len(image_files)} image files")
        if len(image_files) == 0:
            missing.append("No image files in images/ directory")
    
    if missing:
        print("\n⚠ Missing components:")
        for item in missing:
            print(f"  - {item}")
        return False
    
    print("✓ Dataset structure verified successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download OPIXray X-ray baggage scanning dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/opixray",
        help="Output directory for dataset (default: data/opixray)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["manual", "git"],
        default="manual",
        help="Download method: manual instructions or git clone (default: manual)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset structure (use after manual download)",
    )
    
    args = parser.parse_args()
    
    if args.verify:
        if verify_dataset_structure(args.output_dir):
            print("\n✓ Dataset is ready for use!")
            sys.exit(0)
        else:
            print("\n✗ Dataset verification failed. Please complete the download.")
            sys.exit(1)
    
    success = download_opixray(args.output_dir, args.method)
    
    if success:
        verify_dataset_structure(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. If manual download required, complete the download")
    print(f"2. Run: python {__file__} --verify --output-dir {args.output_dir}")
    print("3. Proceed to: python data/create_vqa_pairs.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
