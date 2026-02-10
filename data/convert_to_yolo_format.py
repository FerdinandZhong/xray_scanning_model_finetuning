#!/usr/bin/env python3
"""
Convert STCray dataset annotations to YOLO format.

YOLO format requirements:
- One .txt file per image with same basename
- Each line: class_id x_center y_center width height (all normalized to [0, 1])
- Directory structure: images/train/, images/val/, labels/train/, labels/val/
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from PIL import Image
from tqdm import tqdm
import yaml


# STCray category mapping to class IDs (sorted alphabetically)
CATEGORY_MAPPING = {
    '3D Gun': 0,
    '3D printed gun': 1,
    'Battery': 2,
    'Blade': 3,
    'Bullet': 4,
    'Cutter': 5,
    'Explosive': 6,
    'Gun': 7,
    'Hammer': 8,
    'Handcuffs': 9,
    'Injection': 10,
    'Knife': 11,
    'Lighter': 12,
    'Multilabel Threat': 13,
    'Nail Cutter': 14,
    'Non Threat': 15,
    'Other Sharp Item': 16,
    'Pliers': 17,
    'Powerbank': 18,
    'Scissors': 19,
    'Screwdriver': 20,
    'Shaving Razor': 21,
    'Syringe': 22,
    'Wrench': 23,
}


def convert_bbox_to_yolo(bbox: List[int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bbox from [x, y, width, height] (absolute pixels) to YOLO format.
    
    Args:
        bbox: [x_top_left, y_top_left, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # Normalize width and height
    norm_width = w / img_width
    norm_height = h / img_height
    
    # Clamp values to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))
    
    return x_center, y_center, norm_width, norm_height


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None


def convert_annotation_to_yolo(
    annotation: Dict,
    output_labels_dir: Path,
    output_images_dir: Path,
    split: str
) -> bool:
    """
    Convert a single annotation to YOLO format.
    
    Args:
        annotation: STCray annotation dict
        output_labels_dir: Output directory for label .txt files
        output_images_dir: Output directory for images
        split: 'train' or 'val'
    
    Returns:
        True if successful, False otherwise
    """
    image_path_abs = Path(annotation['image_path_absolute'])
    
    # Check if image exists
    if not image_path_abs.exists():
        print(f"Warning: Image not found: {image_path_abs}")
        return False
    
    # Get image dimensions
    img_width, img_height = get_image_dimensions(str(image_path_abs))
    if img_width is None or img_height is None:
        return False
    
    # Prepare output paths
    image_filename = annotation['image_filename']
    label_filename = Path(image_filename).stem + '.txt'
    
    output_image_path = output_images_dir / split / image_filename
    output_label_path = output_labels_dir / split / label_filename
    
    # Create YOLO label file
    yolo_lines = []
    categories = annotation.get('categories', [])
    bboxes = annotation.get('bboxes', [])
    
    # Handle case where there's one category for all bboxes
    if len(categories) == 1 and len(bboxes) > 1:
        categories = categories * len(bboxes)
    
    if len(categories) != len(bboxes):
        print(f"Warning: Mismatch between categories ({len(categories)}) and bboxes ({len(bboxes)}) for {image_filename}")
        # Use first category for all bboxes as fallback
        if len(categories) > 0:
            categories = [categories[0]] * len(bboxes)
        else:
            return False
    
    for category, bbox in zip(categories, bboxes):
        if category not in CATEGORY_MAPPING:
            print(f"Warning: Unknown category '{category}' in {image_filename}")
            continue
        
        class_id = CATEGORY_MAPPING[category]
        x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
        
        # YOLO format: class_id x_center y_center width height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    if not yolo_lines:
        print(f"Warning: No valid annotations for {image_filename}")
        return False
    
    # Write label file
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    # Copy image file
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path_abs, output_image_path)
    
    return True


def create_data_yaml(output_dir: Path, category_mapping: Dict[str, int]):
    """Create data.yaml configuration file for Ultralytics YOLO."""
    
    # Create class names list (sorted by class_id)
    class_names = [None] * len(category_mapping)
    for category, class_id in category_mapping.items():
        class_names[class_id] = category
    
    data_yaml = {
        'path': str(output_dir.absolute()),  # Root directory
        'train': 'images/train',  # Relative to path
        'val': 'images/val',      # Relative to path
        'nc': len(category_mapping),  # Number of classes
        'names': class_names,  # Class names
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCreated data.yaml at {yaml_path}")
    print(f"  Classes: {len(category_mapping)}")
    print(f"  Train path: {data_yaml['train']}")
    print(f"  Val path: {data_yaml['val']}")


def convert_dataset(
    annotations_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    limit: int = None
):
    """
    Convert STCray dataset to YOLO format.
    
    Args:
        annotations_dir: Directory containing train/test annotations.json
        output_dir: Output directory for YOLO dataset
        val_split: Fraction of train data to use for validation (0.0-1.0)
        limit: Optional limit on number of samples to convert (for testing)
    """
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'
    
    for split in ['train', 'val']:
        (output_images_dir / split).mkdir(parents=True, exist_ok=True)
        (output_labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Load train annotations
    train_annotations_path = annotations_dir / 'train' / 'annotations.json'
    test_annotations_path = annotations_dir / 'test' / 'annotations.json'
    
    print(f"Loading train annotations from {train_annotations_path}...")
    with open(train_annotations_path) as f:
        train_data = json.load(f)
    
    print(f"Loading test annotations from {test_annotations_path}...")
    with open(test_annotations_path) as f:
        test_data = json.load(f)
    
    print(f"\nLoaded {len(train_data)} train samples and {len(test_data)} test samples")
    
    # Apply limit if specified
    if limit:
        print(f"Limiting to {limit} samples for testing")
        train_data = train_data[:int(limit * (1 - val_split))]
        test_data = []  # Skip test data when limiting
    
    # Split train data into train/val
    import random
    random.seed(42)
    shuffled_train = train_data.copy()
    random.shuffle(shuffled_train)
    
    val_size = int(len(shuffled_train) * val_split)
    val_data = shuffled_train[:val_size]
    train_data = shuffled_train[val_size:]
    
    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val")
    print(f"Using test set: {len(test_data)} samples (optional)")
    
    # Convert train data
    print("\nConverting train data...")
    train_success = 0
    for annotation in tqdm(train_data, desc="Train"):
        if convert_annotation_to_yolo(annotation, output_labels_dir, output_images_dir, 'train'):
            train_success += 1
    
    # Convert val data
    print("\nConverting validation data...")
    val_success = 0
    for annotation in tqdm(val_data, desc="Val"):
        if convert_annotation_to_yolo(annotation, output_labels_dir, output_images_dir, 'val'):
            val_success += 1
    
    # Create data.yaml
    create_data_yaml(output_dir, CATEGORY_MAPPING)
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nTrain: {train_success}/{len(train_data)} successful")
    print(f"Val:   {val_success}/{len(val_data)} successful")
    print(f"\nTotal images: {train_success + val_success}")
    print(f"Total classes: {len(CATEGORY_MAPPING)}")
    print(f"\nYOLO dataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── data.yaml")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({train_success} images)")
    print(f"  │   └── val/ ({val_success} images)")
    print(f"  └── labels/")
    print(f"      ├── train/ ({train_success} .txt files)")
    print(f"      └── val/ ({val_success} .txt files)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert STCray dataset to YOLO format"
    )
    parser.add_argument(
        '--annotations-dir',
        type=str,
        default='data/stcray_processed',
        help='Directory containing train/test annotations.json (default: data/stcray_processed)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/yolo_dataset',
        help='Output directory for YOLO dataset (default: data/yolo_dataset)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Fraction of train data to use for validation (default: 0.2)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples for testing (default: None)'
    )
    
    args = parser.parse_args()
    
    convert_dataset(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
