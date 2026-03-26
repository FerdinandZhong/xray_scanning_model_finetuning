#!/usr/bin/env python3
"""
Convert STCray annotations to VQA-style JSONL for VLM (Vision-Language Model) training.

Input:
  data/stcray_processed/train/annotations.json
  data/stcray_processed/test/annotations.json
  
Output:
  data/stcray_vlm/stcray_vlm_train.jsonl
  data/stcray_vlm/stcray_vlm_test.jsonl
  data/stcray_vlm/statistics.json

The VQA format is designed for multi-object detection from single X-ray images,
with structured JSON outputs containing all detected objects and their bounding boxes.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
from PIL import Image


# STCray 21 threat categories + Non Threat
STCRAY_CATEGORIES = [
    "Explosive", "Gun", "3D Gun", "Knife", "Dagger", "Blade",
    "Lighter", "Injection", "Battery", "Nail Cutter", "Other Sharp Item",
    "Powerbank", "Scissors", "Hammer", "Pliers", "Wrench", "Screwdriver",
    "Handcuffs", "Bullet", "Multilabel Threat", "Non Threat"
]

# Threat level mapping for each category
THREAT_LEVELS = {
    "Explosive": "critical",
    "Gun": "critical",
    "3D Gun": "critical",
    "Knife": "high",
    "Dagger": "high",
    "Blade": "high",
    "Bullet": "high",
    "Handcuffs": "high",
    "Injection": "high",
    "Other Sharp Item": "high",
    "Multilabel Threat": "high",
    "Wrench": "medium",
    "Hammer": "medium",
    "Pliers": "medium",
    "Screwdriver": "medium",
    "Battery": "medium",
    "Powerbank": "medium",
    "Scissors": "low",
    "Nail Cutter": "low",
    "Lighter": "low",
    "Non Threat": "none"
}


def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert pixel bounding box [x, y, w, h] to normalized format [x1, y1, x2, y2].
    
    Args:
        bbox: [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [x1, y1, x2, y2] normalized to [0, 1] range
    """
    x, y, w, h = bbox
    x1 = x / img_width
    y1 = y / img_height
    x2 = (x + w) / img_width
    y2 = (y + h) / img_height
    
    # Clamp to [0, 1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def create_vqa_prompt() -> str:
    """Generate the standard VQA prompt for multi-object detection."""
    return "Detect and list all prohibited items in this X-ray baggage scan with their bounding boxes."


def create_vqa_answer(objects: List[Dict[str, Any]]) -> str:
    """
    Create structured JSON answer string for VQA.
    
    Args:
        objects: List of detected objects with category, bbox, threat_level
    
    Returns:
        JSON string with all objects
    """
    answer_dict = {"objects": objects}
    return json.dumps(answer_dict)


def process_annotation(
    annotation: Dict[str, Any],
    project_root: Path
) -> Dict[str, Any]:
    """
    Convert a single STCray annotation to VQA format.
    
    Args:
        annotation: STCray annotation dict
        project_root: Project root directory
    
    Returns:
        VQA-formatted dict ready for JSONL
    """
    # Get image path
    image_path = annotation.get("image_path_absolute") or annotation.get("image_path")
    if not Path(image_path).is_absolute():
        image_path = str(project_root / image_path)
    
    # Load image to get dimensions
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except Exception as e:
        print(f"  ⚠️  Failed to load image {image_path}: {e}")
        return None
    
    # Extract categories and bboxes
    categories = annotation.get("categories", [])
    bboxes = annotation.get("bboxes", [])
    
    # Handle case where no bboxes but has categories
    if categories and not bboxes:
        # Create dummy bboxes (will be filtered later)
        bboxes = [[0, 0, 1, 1]] * len(categories)
    
    # Build object list
    objects = []
    for category, bbox in zip(categories, bboxes):
        # Normalize bbox
        normalized_bbox = normalize_bbox(bbox, img_width, img_height)
        
        # Skip invalid bboxes (too small or malformed)
        x1, y1, x2, y2 = normalized_bbox
        if x2 - x1 < 0.001 or y2 - y1 < 0.001:
            continue
        
        # Get threat level
        threat_level = THREAT_LEVELS.get(category, "unknown")
        
        objects.append({
            "category": category,
            "bbox": normalized_bbox,
            "threat_level": threat_level
        })
    
    # Skip images with no valid objects
    if not objects:
        return None
    
    # Create metadata
    threat_categories = [obj["category"] for obj in objects if obj["threat_level"] != "none"]
    has_critical_threat = any(obj["threat_level"] == "critical" for obj in objects)
    has_high_threat = any(obj["threat_level"] == "high" for obj in objects)
    
    # Create VQA entry
    vqa_entry = {
        "image_path": image_path,
        "question": create_vqa_prompt(),
        "answer": create_vqa_answer(objects),
        "metadata": {
            "question_type": "structured_list",
            "num_objects": len(objects),
            "num_threats": len(threat_categories),
            "threat_categories": threat_categories,
            "has_critical_threat": has_critical_threat,
            "has_high_threat": has_high_threat,
            "image_id": annotation.get("image_id"),
            "image_filename": annotation.get("image_filename")
        }
    }
    
    return vqa_entry


def compute_statistics(vqa_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute dataset statistics."""
    total_images = len(vqa_entries)
    total_objects = sum(entry["metadata"]["num_objects"] for entry in vqa_entries)
    
    # Object count distribution
    object_counts = [entry["metadata"]["num_objects"] for entry in vqa_entries]
    multi_object_images = sum(1 for count in object_counts if count >= 2)
    rich_multi_object_images = sum(1 for count in object_counts if count >= 3)
    
    # Category distribution
    category_counter = Counter()
    threat_level_counter = Counter()
    
    for entry in vqa_entries:
        answer = json.loads(entry["answer"])
        for obj in answer["objects"]:
            category_counter[obj["category"]] += 1
            threat_level_counter[obj["threat_level"]] += 1
    
    # Threat statistics
    images_with_critical = sum(1 for entry in vqa_entries if entry["metadata"]["has_critical_threat"])
    images_with_high = sum(1 for entry in vqa_entries if entry["metadata"]["has_high_threat"])
    
    stats = {
        "total_images": total_images,
        "total_objects": total_objects,
        "avg_objects_per_image": round(total_objects / total_images, 2) if total_images > 0 else 0,
        "multi_object_images": multi_object_images,
        "multi_object_ratio": round(multi_object_images / total_images, 3) if total_images > 0 else 0,
        "rich_multi_object_images": rich_multi_object_images,
        "rich_multi_object_ratio": round(rich_multi_object_images / total_images, 3) if total_images > 0 else 0,
        "images_with_critical_threat": images_with_critical,
        "images_with_high_threat": images_with_high,
        "category_distribution": dict(category_counter.most_common()),
        "threat_level_distribution": dict(threat_level_counter),
        "object_count_distribution": {
            "1": sum(1 for c in object_counts if c == 1),
            "2": sum(1 for c in object_counts if c == 2),
            "3": sum(1 for c in object_counts if c == 3),
            "4": sum(1 for c in object_counts if c == 4),
            "5+": sum(1 for c in object_counts if c >= 5)
        }
    }
    
    return stats


def convert_split(
    annotations_file: Path,
    output_file: Path,
    split_name: str,
    project_root: Path
) -> List[Dict[str, Any]]:
    """
    Convert one split (train/test) to VQA format.
    
    Returns:
        List of VQA entries
    """
    print(f"\nProcessing {split_name} split...")
    print(f"  Input:  {annotations_file}")
    print(f"  Output: {output_file}")
    
    # Load annotations
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
    
    print(f"  Loaded {len(annotations):,} annotations")
    
    # Convert to VQA format
    vqa_entries = []
    skipped = 0
    
    for i, annotation in enumerate(annotations):
        vqa_entry = process_annotation(annotation, project_root)
        
        if vqa_entry is None:
            skipped += 1
            continue
        
        vqa_entries.append(vqa_entry)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1:,}/{len(annotations):,} annotations...")
    
    print(f"  ✓ Converted {len(vqa_entries):,} entries ({skipped:,} skipped)")
    
    # Write JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for entry in vqa_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"  ✓ Saved to {output_file}")
    
    return vqa_entries


def main():
    parser = argparse.ArgumentParser(
        description="Convert STCray annotations to VQA-style JSONL for VLM training"
    )
    parser.add_argument(
        "--input-dir",
        default="data/stcray_processed",
        help="Input directory with STCray annotations (default: data/stcray_processed)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/stcray_vlm",
        help="Output directory for VQA JSONL files (default: data/stcray_vlm)"
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root directory (default: cwd)"
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    
    print("=" * 60)
    print("STCray → VQA Conversion for VLM Training")
    print("=" * 60)
    print(f"  Input dir:    {input_dir.resolve()}")
    print(f"  Output dir:   {output_dir.resolve()}")
    print(f"  Project root: {project_root.resolve()}")
    print()
    print("Multi-object detection focus:")
    print("  - All objects in each image will be included")
    print("  - Bounding boxes normalized to [0, 1] range")
    print("  - Structured JSON output with threat levels")
    print("=" * 60)
    
    # Process splits
    all_stats = {}
    
    splits = [
        ("train", "stcray_vlm_train.jsonl"),
        ("test", "stcray_vlm_test.jsonl")
    ]
    
    for split_name, output_filename in splits:
        annotations_file = input_dir / split_name / "annotations.json"
        
        if not annotations_file.exists():
            print(f"\n⚠️  {split_name} annotations not found: {annotations_file}")
            print("     Skipping...")
            continue
        
        output_file = output_dir / output_filename
        
        vqa_entries = convert_split(
            annotations_file,
            output_file,
            split_name,
            project_root
        )
        
        # Compute statistics
        stats = compute_statistics(vqa_entries)
        all_stats[split_name] = stats
        
        # Print statistics
        print(f"\n  {split_name.upper()} Statistics:")
        print(f"    Total images:             {stats['total_images']:,}")
        print(f"    Total objects:            {stats['total_objects']:,}")
        print(f"    Avg objects/image:        {stats['avg_objects_per_image']:.2f}")
        print(f"    Multi-object images (2+): {stats['multi_object_images']:,} ({stats['multi_object_ratio']:.1%})")
        print(f"    Rich multi-obj (3+):      {stats['rich_multi_object_images']:,} ({stats['rich_multi_object_ratio']:.1%})")
        print(f"    Images with critical:     {stats['images_with_critical_threat']:,}")
        print(f"    Images with high threat:  {stats['images_with_high_threat']:,}")
        print()
        print(f"    Top 5 categories:")
        for cat, count in list(stats['category_distribution'].items())[:5]:
            print(f"      - {cat}: {count:,}")
    
    # Save statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print()
    print("=" * 60)
    print(f"✅ Conversion complete!")
    print(f"   Train: {output_dir / 'stcray_vlm_train.jsonl'}")
    print(f"   Test:  {output_dir / 'stcray_vlm_test.jsonl'}")
    print(f"   Stats: {stats_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
