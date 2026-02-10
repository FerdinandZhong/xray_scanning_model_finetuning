#!/usr/bin/env python3
"""
Convert Luggage X-ray dataset from OpenAI JSONL format to YOLO format.

This dataset has 12 categories including threat items (knives, blades) and
normal items (bottles, cans). Format is vision-language with bounding boxes
in <loc####> format (normalized x1000).

Dataset: 7,120 images (6,163 train, 955 validation)
Categories: Cans, CartonDrinks, GlassBottle, PlasticBottle, SprayCans, 
           SwissArmyKnife, Tin, VacuumCup, blade, dagger, knife, scissors
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
from tqdm import tqdm


# Standard category list (alphabetical order)
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


def parse_loc_bbox(loc_string: str) -> List[Tuple[List[float], str]]:
    """
    Parse bounding boxes from <loc####> format.
    
    Format: <loc0232><loc0378><loc0452><loc0748> PlasticBottle
    Returns: [(x1, y1, x2, y2, 'PlasticBottle'), ...]
    
    Coordinates are normalized * 1000, need to divide by 1000.
    """
    # Pattern: <loc####><loc####><loc####><loc####> CategoryName
    pattern = r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s+(\w+)'
    matches = re.findall(pattern, loc_string)
    
    bboxes = []
    for x1, y1, x2, y2, category in matches:
        # Convert to normalized coordinates [0-1]
        bbox = [
            int(x1) / 1000.0,
            int(y1) / 1000.0,
            int(x2) / 1000.0,
            int(y2) / 1000.0
        ]
        bboxes.append((bbox, category))
    
    return bboxes


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
    """Download image from URL."""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def process_annotation(
    annotation: Dict,
    output_dir: Path,
    split: str,
    image_index: int,
    download_images: bool = True
) -> Tuple[bool, str]:
    """
    Process a single annotation entry.
    
    Returns: (success, image_name)
    """
    messages = annotation['messages']
    
    # Extract image URL from user message with image_url
    image_url = None
    for msg in messages:
        if msg['role'] == 'user' and isinstance(msg['content'], list):
            for item in msg['content']:
                if item.get('type') == 'image_url':
                    image_url = item['image_url']['url']
                    break
    
    if not image_url:
        return False, ""
    
    # Extract bounding boxes from assistant message
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
    
    # Download image if needed
    if download_images:
        if not download_image(image_url, image_path):
            return False, ""
    
    # Create YOLO label file
    label_lines = []
    for bbox, category in bboxes:
        if category not in CATEGORY_TO_ID:
            print(f"Warning: Unknown category '{category}' in {image_name}")
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


def convert_split(
    jsonl_file: Path,
    output_dir: Path,
    split: str,
    max_workers: int = 8,
    download_images: bool = True
):
    """Convert a single split (train/valid) to YOLO format."""
    print(f"\n{'='*60}")
    print(f"Converting {split} split: {jsonl_file}")
    print(f"{'='*60}")
    
    # Create directories
    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotations = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))
    
    print(f"Loaded {len(annotations)} annotations")
    
    # Process annotations with progress bar
    successful = 0
    failed = 0
    
    if download_images:
        print(f"Downloading images and creating labels (using {max_workers} workers)...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, ann in enumerate(annotations):
                future = executor.submit(
                    process_annotation,
                    ann,
                    output_dir,
                    split,
                    idx,
                    download_images
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                success, image_name = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
    else:
        print("Creating labels only (images already downloaded)...")
        for idx, ann in tqdm(enumerate(annotations), total=len(annotations)):
            success, image_name = process_annotation(
                ann, output_dir, split, idx, download_images
            )
            if success:
                successful += 1
            else:
                failed += 1
    
    print(f"\n✓ Processed: {successful} successful, {failed} failed")
    
    return successful, failed


def create_data_yaml(output_dir: Path):
    """Create data.yaml configuration file for YOLO training."""
    yaml_content = f"""# Luggage X-ray Dataset - YOLO Format
# Source: Roboflow (yolov5xray - v1)
# Total: 7,120 images (6,163 train, 955 validation)
# Categories: 12 (5 threats, 7 normal items)

path: {output_dir.absolute()}
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
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created data.yaml: {yaml_path}")


def print_summary(train_count: int, valid_count: int, output_dir: Path):
    """Print conversion summary."""
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"Dataset: Luggage X-ray (yolov5xray)")
    print(f"Format: OpenAI JSONL → YOLO")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"  Threats: blade, dagger, knife, scissors, SwissArmyKnife")
    print(f"  Normal: Cans, CartonDrinks, GlassBottle, PlasticBottle, etc.")
    print(f"\nSplits:")
    print(f"  Train: {train_count} images")
    print(f"  Valid: {valid_count} images")
    print(f"  Total: {train_count + valid_count} images")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nReady for YOLO training!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Luggage X-ray dataset to YOLO format"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/luggage_xray'),
        help='Input directory with JSONL annotations'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/luggage_xray_yolo'),
        help='Output directory for YOLO format'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Number of parallel workers for downloading'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip image download (labels only)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Luggage X-ray to YOLO Converter")
    print("="*60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.max_workers}")
    print(f"Download images: {not args.no_download}")
    print("="*60)
    
    # Verify input files
    train_file = args.input_dir / '_annotations.train.jsonl'
    valid_file = args.input_dir / '_annotations.valid.jsonl'
    
    if not train_file.exists():
        print(f"❌ Error: Training annotations not found: {train_file}")
        return 1
    
    if not valid_file.exists():
        print(f"❌ Error: Validation annotations not found: {valid_file}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert splits
    train_success, train_failed = convert_split(
        train_file,
        args.output_dir,
        'train',
        args.max_workers,
        not args.no_download
    )
    
    valid_success, valid_failed = convert_split(
        valid_file,
        args.output_dir,
        'valid',
        args.max_workers,
        not args.no_download
    )
    
    # Create data.yaml
    create_data_yaml(args.output_dir)
    
    # Print summary
    print_summary(train_success, valid_success, args.output_dir)
    
    if train_failed > 0 or valid_failed > 0:
        print(f"\n⚠️  Warning: {train_failed + valid_failed} images failed to process")
        return 1
    
    print("\n✅ Conversion complete!")
    return 0


if __name__ == '__main__':
    exit(main())
