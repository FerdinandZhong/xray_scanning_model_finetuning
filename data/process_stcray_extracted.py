#!/usr/bin/env python3
"""
Process extracted STCray dataset into unified annotations format.
Converts the class-based directory structure into a single annotations.json file.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def extract_bbox_from_polygon(points: List[List[float]]) -> List[float]:
    """
    Extract bounding box [x, y, width, height] from polygon points.
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        [x, y, width, height] bounding box
    """
    if not points:
        return [0, 0, 0, 0]
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def process_stcray_dataset(
    input_dir: str,
    output_dir: str,
    max_samples: int = None,
):
    """
    Process STCray dataset from extracted RAR files.
    
    Args:
        input_dir: Path to extracted STCray directory (e.g., data/stcray_raw/STCray_TrainSet)
        output_dir: Output directory for processed annotations
        max_samples: Maximum samples to process (for testing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STCray Dataset Processor")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Directories
    images_base = input_path / "Images"
    json_base = input_path / "Json"
    captions_base = input_path / "Captions"
    
    if not images_base.exists():
        print(f"Error: Images directory not found: {images_base}")
        return False
    
    # Find all class directories
    class_dirs = sorted([d for d in images_base.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} class directories:")
    for d in class_dirs:
        print(f"  - {d.name}")
    print()
    
    # Process all images
    annotations = []
    image_id = 0
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        
        # Extract class label (remove "Class N_" prefix)
        # E.g., "Class 1_Explosive" -> "Explosive"
        if "_" in class_name:
            class_label = class_name.split("_", 1)[1]
        else:
            class_label = class_name
        
        # Get all images in this class
        image_files = sorted(class_dir.glob("*.jpg"))
        
        if max_samples and image_id >= max_samples:
            break
        
        for img_file in image_files:
            if max_samples and image_id >= max_samples:
                break
            
            # Find corresponding JSON file
            json_file = json_base / class_name / f"{img_file.stem}.json"
            
            # Find corresponding caption file
            caption_file = captions_base / class_name / f"{img_file.stem}.txt"
            
            # Read caption if available
            caption = ""
            if caption_file.exists():
                try:
                    with open(caption_file) as f:
                        caption = f.read().strip()
                except:
                    pass
            
            # Read JSON annotation
            bboxes = []
            categories = [class_label]
            
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        json_data = json.load(f)
                    
                    # Extract bounding boxes from shapes
                    for shape in json_data.get("shapes", []):
                        points = shape.get("points", [])
                        if points:
                            bbox = extract_bbox_from_polygon(points)
                            bboxes.append(bbox)
                        
                        # Also track label from shape (in case multiple items)
                        shape_label = shape.get("label", class_label)
                        if shape_label not in categories:
                            categories.append(shape_label)
                except Exception as e:
                    print(f"  Warning: Failed to parse {json_file}: {e}")
            
            # Relative path from output directory
            relative_img_path = f"{class_name}/{img_file.name}"
            
            # Create annotation entry
            annotation = {
                "image_id": image_id,
                "image_filename": img_file.name,
                "image_path": relative_img_path,
                "image_path_absolute": str(img_file),
                "class_name": class_name,
                "categories": categories,
                "bboxes": bboxes,
                "caption": caption,
                "num_annotations": len(bboxes),
            }
            
            annotations.append(annotation)
            image_id += 1
    
    # Save annotations
    output_file = output_path / "annotations.json"
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Total images processed: {len(annotations)}")
    print(f"Output file: {output_file}")
    print()
    
    # Statistics
    print("Statistics:")
    print(f"  Total samples: {len(annotations)}")
    print(f"  With bounding boxes: {sum(1 for a in annotations if a['num_annotations'] > 0)}")
    print(f"  With captions: {sum(1 for a in annotations if a['caption'])}")
    
    # Category distribution
    category_counts = {}
    for ann in annotations:
        for cat in ann['categories']:
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print()
    print("Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    print()
    print("Next step: Generate VQA dataset")
    print(f"  Update generate_vqa_gemini.sh to use:")
    print(f"    ANNOTATIONS_FILE=\"{output_file}\"")
    print(f"    IMAGES_DIR=\"{images_base}\"")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process extracted STCray dataset into annotations format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to extracted STCray directory (e.g., data/stcray_raw/STCray_TrainSet)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed annotations"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    success = process_stcray_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
