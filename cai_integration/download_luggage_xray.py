#!/usr/bin/env python3
"""
Download and prepare Luggage X-ray dataset for CAI.

This script:
1. Downloads the Luggage X-ray dataset from Roboflow
2. Converts from OpenAI JSONL format to YOLO format
3. Downloads all images from URLs
4. Creates train/val splits with labels
5. Generates data.yaml for YOLO training

Can be run manually in CAI or as a scheduled job.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Configure for CAI environment
PROJECT_ROOT = Path("/home/cdsw")
DOWNLOAD_URL = "https://app.roboflow.com/ds/nMb0ckPbFf?key=EZzAfTucdZ"
RAW_DATA_DIR = PROJECT_ROOT / "data/luggage_xray"
YOLO_DATA_DIR = PROJECT_ROOT / "data/luggage_xray_yolo"

# Standard category list (12 classes)
CATEGORIES = [
    'blade',
    'Cans',
    'CartonDrinks',
    'dagger',
    'GlassBottle',
    'knife',
    'PlasticBottle',
    'scissors',
    'SprayCans',
    'SwissArmyKnife',
    'Tin',
    'VacuumCup',
]

# Category to class ID mapping
CATEGORY_TO_ID = {cat: idx for idx, cat in enumerate(CATEGORIES)}


def download_dataset():
    """Download Luggage X-ray dataset from Roboflow."""
    print("=" * 60)
    print("Step 1: Downloading Luggage X-ray Dataset")
    print("=" * 60)
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = PROJECT_ROOT / "luggage_xray.zip"
    
    print(f"Downloading from Roboflow...")
    print(f"URL: {DOWNLOAD_URL}")
    
    try:
        # Download using curl (more reliable in CAI)
        subprocess.run(
            ["curl", "-L", DOWNLOAD_URL, "-o", str(zip_path)],
            check=True,
            capture_output=True
        )
        
        file_size = zip_path.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded: {file_size:.1f} MB")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)
    
    # Extract
    print(f"\nExtracting to {RAW_DATA_DIR}...")
    try:
        subprocess.run(
            ["unzip", "-q", str(zip_path), "-d", str(RAW_DATA_DIR)],
            check=True
        )
        print("✓ Extraction complete")
        
        # Clean up zip file
        zip_path.unlink()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)
    
    # Verify files
    train_annotations = RAW_DATA_DIR / "_annotations.train.jsonl"
    valid_annotations = RAW_DATA_DIR / "_annotations.valid.jsonl"
    
    if not train_annotations.exists() or not valid_annotations.exists():
        print("❌ Error: Annotation files not found after extraction")
        sys.exit(1)
    
    print(f"✓ Train annotations: {train_annotations}")
    print(f"✓ Valid annotations: {valid_annotations}")
    print()


def parse_loc_bbox(loc_string: str) -> List[Tuple[str, List[float]]]:
    """
    Parse bounding boxes from <loc####> format.
    
    Format: <loc0232><loc0378><loc0452><loc0748> PlasticBottle
    Returns: [(category, [x1, y1, x2, y2]), ...]
    """
    pattern = r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s+(\w+)'
    matches = re.findall(pattern, loc_string)
    
    results = []
    for x1, y1, x2, y2, category in matches:
        bbox = [
            int(x1) / 1000.0,
            int(y1) / 1000.0,
            int(x2) / 1000.0,
            int(y2) / 1000.0
        ]
        results.append((category, bbox))
    
    return results


def bbox_to_yolo(bbox: List[float]) -> List[float]:
    """
    Convert [x1, y1, x2, y2] (normalized) to YOLO format [x_center, y_center, width, height].
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return [x_center, y_center, width, height]


def download_image(url: str, output_path: Path) -> bool:
    """Download image from URL with retry."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_path)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download {url}: {e}")
                return False
    return False


def process_annotation(
    annotation: Dict,
    output_dir: Path,
    split: str,
    image_index: int
) -> Tuple[bool, str]:
    """
    Process a single annotation entry.
    
    Returns: (success, image_name)
    """
    messages = annotation['messages']
    
    # Extract image URL
    image_url = None
    for msg in messages:
        if msg['role'] == 'user' and isinstance(msg['content'], list):
            for item in msg['content']:
                if item.get('type') == 'image_url':
                    image_url = item['image_url']['url']
                    break
    
    if not image_url:
        return False, ""
    
    # Extract bounding boxes
    assistant_content = None
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            break
    
    if not assistant_content:
        return False, ""
    
    # Parse bounding boxes
    bboxes = parse_loc_bbox(assistant_content)
    
    if not bboxes:
        return False, ""
    
    # Create image filename
    image_name = f"{split}_{image_index:06d}.jpg"
    image_path = output_dir / 'images' / split / image_name
    label_path = output_dir / 'labels' / split / f"{split}_{image_index:06d}.txt"
    
    # Download image
    if not download_image(image_url, image_path):
        return False, ""
    
    # Create YOLO label file
    label_lines = []
    for category, bbox in bboxes:
        if category not in CATEGORY_TO_ID:
            continue
        
        class_id = CATEGORY_TO_ID[category]
        yolo_bbox = bbox_to_yolo(bbox)
        
        # Format: class_id x_center y_center width height
        label_line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
        label_lines.append(label_line)
    
    # Write label file
    with open(label_path, 'w') as f:
        f.write('\n'.join(label_lines))
    
    return True, image_name


def convert_to_yolo(max_workers: int = 8):
    """Convert Luggage X-ray dataset to YOLO format."""
    print("=" * 60)
    print("Step 2: Converting to YOLO Format")
    print("=" * 60)
    
    # Create output directories
    YOLO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'valid']:
        (YOLO_DATA_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process train split
    print(f"\nProcessing training set...")
    train_file = RAW_DATA_DIR / "_annotations.train.jsonl"
    train_success = process_split(train_file, 'train', max_workers)
    
    # Process validation split
    print(f"\nProcessing validation set...")
    valid_file = RAW_DATA_DIR / "_annotations.valid.jsonl"
    valid_success = process_split(valid_file, 'valid', max_workers)
    
    print(f"\n✓ Total processed:")
    print(f"  Train: {train_success} images")
    print(f"  Valid: {valid_success} images")
    print()


def process_split(jsonl_file: Path, split: str, max_workers: int) -> int:
    """Process a single split (train/valid)."""
    # Load annotations
    annotations = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))
    
    total = len(annotations)
    print(f"  Loaded {total} annotations")
    print(f"  Downloading images using {max_workers} workers...")
    
    successful = 0
    failed = 0
    
    # Process with progress
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, ann in enumerate(annotations):
            future = executor.submit(
                process_annotation,
                ann,
                YOLO_DATA_DIR,
                split,
                idx
            )
            futures.append(future)
        
        # Track progress
        for i, future in enumerate(as_completed(futures)):
            success, _ = future.result()
            if success:
                successful += 1
            else:
                failed += 1
            
            # Print progress every 100 images
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{total} ({successful} success, {failed} failed)")
    
    print(f"  ✓ Completed: {successful} success, {failed} failed")
    return successful


def create_data_yaml():
    """Create data.yaml configuration file for YOLO training."""
    print("=" * 60)
    print("Step 3: Creating data.yaml")
    print("=" * 60)
    
    yaml_content = f"""# Luggage X-ray Dataset - YOLO Format
# Source: Roboflow (yolov5xray - v1)
# Total: 7,120 images (6,164 train, 956 validation)
# Categories: 12 (5 threats, 7 normal items)

path: {YOLO_DATA_DIR.absolute()}
train: images/train
val: images/valid

# Classes
nc: {len(CATEGORIES)}
names: {CATEGORIES}

# Threat categories (for reference)
threats:
  - blade
  - dagger
  - knife
  - scissors
  - SwissArmyKnife

# Normal items
normal:
  - Cans
  - CartonDrinks
  - GlassBottle
  - PlasticBottle
  - SprayCans
  - Tin
  - VacuumCup
"""
    
    yaml_path = YOLO_DATA_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created: {yaml_path}")
    print()


def verify_dataset():
    """Verify the dataset was created correctly."""
    print("=" * 60)
    print("Step 4: Verification")
    print("=" * 60)
    
    data_yaml = YOLO_DATA_DIR / 'data.yaml'
    train_images = list((YOLO_DATA_DIR / 'images' / 'train').glob('*.jpg'))
    valid_images = list((YOLO_DATA_DIR / 'images' / 'valid').glob('*.jpg'))
    train_labels = list((YOLO_DATA_DIR / 'labels' / 'train').glob('*.txt'))
    valid_labels = list((YOLO_DATA_DIR / 'labels' / 'valid').glob('*.txt'))
    
    print(f"✓ data.yaml: {'EXISTS' if data_yaml.exists() else 'MISSING'}")
    print(f"✓ Train images: {len(train_images)}")
    print(f"✓ Train labels: {len(train_labels)}")
    print(f"✓ Valid images: {len(valid_images)}")
    print(f"✓ Valid labels: {len(valid_labels)}")
    print()
    
    # Verify image/label pairs match
    train_match = len(train_images) == len(train_labels)
    valid_match = len(valid_images) == len(valid_labels)
    
    if train_match and valid_match:
        print("✅ Dataset verification passed!")
    else:
        print("⚠️  Warning: Image/label count mismatch")
        if not train_match:
            print(f"   Train: {len(train_images)} images vs {len(train_labels)} labels")
        if not valid_match:
            print(f"   Valid: {len(valid_images)} images vs {len(valid_labels)} labels")
    
    print()
    
    # Calculate dataset size
    total_size = sum(f.stat().st_size for f in train_images + valid_images)
    size_mb = total_size / (1024 * 1024)
    print(f"Total dataset size: {size_mb:.1f} MB")
    print()


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("Luggage X-ray Dataset Download & Preparation")
    print("For CAI (Cloudera AI Workspace)")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {YOLO_DATA_DIR}")
    print("=" * 60)
    print()
    
    # Check if already exists
    if YOLO_DATA_DIR.exists() and (YOLO_DATA_DIR / 'data.yaml').exists():
        print("⚠️  Dataset already exists at:", YOLO_DATA_DIR)
        print()
        response = input("Delete and re-download? (yes/no): ").strip().lower()
        if response == 'yes':
            print("Removing existing dataset...")
            shutil.rmtree(YOLO_DATA_DIR)
            if RAW_DATA_DIR.exists():
                shutil.rmtree(RAW_DATA_DIR)
            print("✓ Cleaned up")
            print()
        else:
            print("Keeping existing dataset. Exiting.")
            return 0
    
    try:
        # Step 1: Download
        download_dataset()
        
        # Step 2: Convert to YOLO
        max_workers = int(os.getenv('MAX_WORKERS', '8'))
        convert_to_yolo(max_workers=max_workers)
        
        # Step 3: Create data.yaml
        create_data_yaml()
        
        # Step 4: Verify
        verify_dataset()
        
        print("=" * 60)
        print("✅ SUCCESS! Dataset ready for YOLO training")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Train YOLO model:")
        print(f"     python training/train_yolo.py \\")
        print(f"       --data {YOLO_DATA_DIR}/data.yaml \\")
        print(f"       --model yolov8n.pt \\")
        print(f"       --epochs 100")
        print()
        print("  2. Or use in CAI job:")
        print("     Configure 'yolo_training' job to use this dataset")
        print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Download cancelled by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
