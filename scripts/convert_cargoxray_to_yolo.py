#!/usr/bin/env python3
"""
Convert CargoXray COCO format annotations to YOLO format.

CargoXray dataset from Roboflow contains cargo/container X-ray images
with 24 categories in COCO segmentation format.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def load_coco_annotations(annotation_file: Path) -> Dict:
    """Load COCO format annotations."""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def normalize_categories(categories: List[Dict]) -> Dict[int, Tuple[int, str]]:
    """
    Normalize category names and create ID mapping.
    
    Handles duplicates and typos (e.g., 'clothes'/'clohes', 'textile'/'textiles').
    Returns: {coco_id: (yolo_id, normalized_name)}
    """
    # Normalization rules
    normalize_map = {
        'clohes': 'clothes',
        'texstiles': 'textiles',
        'tetiles': 'textiles',
        'textiles': 'textiles',
        'textile': 'textiles',
        'tableware': 'tableware',
        'tablware': 'tableware',
        'table warre': 'tableware',
        'table ware': 'tableware',
        'car weels': 'car_wheels',
        '---- -----': 'unknown',
        'object': 'unknown',
        'xrayobjects': 'xray_objects',
    }
    
    # Collect unique normalized names
    normalized_names = set()
    for cat in categories:
        name = cat['name'].lower().strip()
        normalized = normalize_map.get(name, name.replace(' ', '_'))
        normalized_names.add(normalized)
    
    # Sort for consistent ordering
    sorted_names = sorted(normalized_names)
    
    # Create mapping: coco_id -> (yolo_id, normalized_name)
    id_mapping = {}
    for cat in categories:
        coco_id = cat['id']
        name = cat['name'].lower().strip()
        normalized = normalize_map.get(name, name.replace(' ', '_'))
        yolo_id = sorted_names.index(normalized)
        id_mapping[coco_id] = (yolo_id, normalized)
    
    return id_mapping, sorted_names


def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height].
    
    COCO: [x_min, y_min, width, height] in pixels
    YOLO: [x_center, y_center, width, height] normalized to [0, 1]
    """
    x, y, w, h = bbox
    
    # Convert to center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height
    
    # Clip to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))
    
    return x_center, y_center, norm_width, norm_height


def convert_split_to_yolo(
    split_dir: Path,
    output_dir: Path,
    split_name: str,
    category_mapping: Dict,
    class_names: List[str]
) -> int:
    """Convert a single split (train/valid/test) to YOLO format."""
    annotation_file = split_dir / "_annotations.coco.json"
    
    if not annotation_file.exists():
        print(f"Warning: Annotation file not found: {annotation_file}")
        return 0
    
    print(f"\nConverting {split_name} split...")
    coco_data = load_coco_annotations(annotation_file)
    
    # Create output directories
    images_dir = output_dir / split_name / "images"
    labels_dir = output_dir / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image ID to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    converted_count = 0
    
    for image_id, img_info in image_info.items():
        filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Source image path
        src_image = split_dir / filename
        if not src_image.exists():
            print(f"Warning: Image not found: {src_image}")
            continue
        
        # Destination paths
        dest_image = images_dir / filename
        label_filename = Path(filename).stem + ".txt"
        dest_label = labels_dir / label_filename
        
        # Copy image
        shutil.copy2(src_image, dest_image)
        
        # Convert annotations to YOLO format
        yolo_annotations = []
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                category_id = ann['category_id']
                bbox = ann['bbox']
                
                # Get YOLO class ID
                if category_id not in category_mapping:
                    print(f"Warning: Unknown category ID {category_id}")
                    continue
                
                yolo_class_id, _ = category_mapping[category_id]
                
                # Convert bbox to YOLO format
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    bbox, img_width, img_height
                )
                
                yolo_annotations.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write label file (even if empty - indicates no objects)
        with open(dest_label, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        converted_count += 1
    
    print(f"Converted {converted_count} images for {split_name}")
    return converted_count


def create_data_yaml(output_dir: Path, class_names: List[str]):
    """Create data.yaml file for YOLO training."""
    data_yaml = output_dir / "data.yaml"
    
    yaml_content = f"""# CargoXray Dataset - YOLO Format
# Converted from Roboflow COCO format
# Source: https://app.roboflow.com/ds/BbQux1Jbmr

# Paths (relative to this file)
path: {output_dir.absolute()}
train: train/images
val: valid/images
test: test/images

# Classes
nc: {len(class_names)}
names: {class_names}
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated data.yaml at {data_yaml}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CargoXray COCO annotations to YOLO format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/cargoxray"),
        help="Directory containing train/valid/test folders with COCO annotations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cargoxray_yolo"),
        help="Output directory for YOLO format dataset"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    # Load annotations from train split to get categories
    train_annotations = args.input_dir / "train" / "_annotations.coco.json"
    if not train_annotations.exists():
        print(f"Error: Training annotations not found: {train_annotations}")
        return 1
    
    coco_data = load_coco_annotations(train_annotations)
    category_mapping, class_names = normalize_categories(coco_data['categories'])
    
    print(f"Found {len(class_names)} unique categories:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Convert each split
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    total_converted = 0
    for split_name in ['train', 'valid', 'test']:
        split_dir = args.input_dir / split_name
        if split_dir.exists():
            count = convert_split_to_yolo(
                split_dir,
                args.output_dir,
                split_name,
                category_mapping,
                class_names
            )
            total_converted += count
        else:
            print(f"Warning: {split_name} directory not found: {split_dir}")
    
    # Create data.yaml
    create_data_yaml(args.output_dir, class_names)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Total images converted: {total_converted}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")
    
    print(f"\nNext steps:")
    print(f"1. Train YOLO model:")
    print(f"   python training/train_yolo.py \\")
    print(f"     --data {args.output_dir}/data.yaml \\")
    print(f"     --model yolov8n.pt \\")
    print(f"     --epochs 100 \\")
    print(f"     --batch 16 \\")
    print(f"     --name cargoxray_v1")
    
    return 0


if __name__ == "__main__":
    exit(main())
