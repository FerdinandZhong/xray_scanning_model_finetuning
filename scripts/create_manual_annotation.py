#!/usr/bin/env python3
"""
Create manual annotations for X-ray images.

Interactive CLI tool to annotate X-ray images with detected items.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def create_annotation_interactive(image_path: Path) -> Dict:
    """
    Interactively create annotation for an X-ray image.
    
    Args:
        image_path: Path to the X-ray image
    
    Returns:
        Annotation dictionary
    """
    print(f"\n{'='*70}")
    print(f"Manual Annotation for: {image_path.name}")
    print(f"{'='*70}\n")
    
    # Common X-ray threat categories
    common_categories = [
        "Knife", "Gun", "Explosive", "Scissors", "Blade",
        "Battery", "Lighter", "Hammer", "Wrench", "Screwdriver",
        "Pliers", "Cutter", "Razor", "Needle", "Syringe",
        "Handcuffs", "Tool", "Weapon", "Sharp Object"
    ]
    
    print("Common categories:")
    for i, cat in enumerate(common_categories, 1):
        print(f"  {i:2d}. {cat}")
    print()
    
    # Collect categories
    categories = []
    items = []
    
    print("Enter detected items (one per line, empty line to finish):")
    print("Format: category_number or custom_name")
    print()
    
    item_count = 0
    while True:
        item_count += 1
        print(f"Item {item_count}:")
        
        # Get category
        category_input = input(f"  Category (1-{len(common_categories)} or custom): ").strip()
        
        if not category_input:
            print("  (Skipping empty input)")
            break
        
        # Parse category
        try:
            cat_index = int(category_input) - 1
            if 0 <= cat_index < len(common_categories):
                category = common_categories[cat_index]
            else:
                print(f"  Invalid number, using as custom category")
                category = category_input
        except ValueError:
            category = category_input
        
        # Get item name (optional)
        item_name = input(f"  Item name (default: {category}): ").strip()
        if not item_name:
            item_name = category
        
        # Get location (optional)
        location = input(f"  Location (e.g., center, upper-left) [optional]: ").strip()
        
        # Get confidence (optional)
        confidence_input = input(f"  Confidence (0.0-1.0) [optional]: ").strip()
        try:
            confidence = float(confidence_input) if confidence_input else None
        except ValueError:
            confidence = None
        
        # Add to lists
        if category not in categories:
            categories.append(category)
        
        item_dict = {
            "category": category,
            "item_name": item_name
        }
        if location:
            item_dict["location"] = location
        if confidence is not None:
            item_dict["confidence"] = confidence
        
        items.append(item_dict)
        print(f"  ✓ Added: {item_name}")
        print()
    
    # Get additional notes
    notes = input("\nAdditional notes [optional]: ").strip()
    
    # Create annotation
    annotation = {
        "image_filename": image_path.name,
        "image_path": str(image_path.relative_to(image_path.parent.parent)) if image_path.parent.parent.exists() else str(image_path),
        "image_path_absolute": str(image_path.absolute()),
        "categories": categories,
        "items": items,
        "num_annotations": len(items),
        "notes": notes if notes else "",
        "source": "manual_annotation",
        "annotator": "human"
    }
    
    return annotation


def main():
    parser = argparse.ArgumentParser(
        description="Create manual annotations for X-ray images"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to X-ray image to annotate'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output annotation file (default: <image_stem>_annotation.json in same directory)'
    )
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    # Create annotation
    annotation = create_annotation_interactive(image_path)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_annotation.json"
    
    # Save annotation
    with open(output_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Annotation Summary")
    print(f"{'='*70}")
    print(f"Image: {image_path.name}")
    print(f"Categories: {', '.join(annotation['categories'])}")
    print(f"Items: {len(annotation['items'])}")
    print(f"\nSaved to: {output_path}")
    print(f"\nTo test with RolmOCR:")
    print(f"  python3 test_rolmocr.py --image {image_path} --annotations {output_path}")
    print()


if __name__ == "__main__":
    main()
