#!/usr/bin/env python3
"""
Transform OPIXray COCO annotations into VQA (Visual Question Answering) format.
Generates question-answer pairs for training vision-language models.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


# OPIXray categories (5 prohibited items)
CATEGORIES = {
    1: "Folding_Knife",
    2: "Straight_Knife",
    3: "Scissor",
    4: "Utility_Knife",
    5: "Multi-tool_Knife",
}

# Friendly category names
CATEGORY_FRIENDLY = {
    "Folding_Knife": "folding knife",
    "Straight_Knife": "straight knife",
    "Scissor": "scissors",
    "Utility_Knife": "utility knife",
    "Multi-tool_Knife": "multi-tool knife",
}

# Question templates (focused on item recognition only)
QUESTION_TEMPLATES = {
    "general": [
        "What items are visible in this X-ray scan?",
        "Analyze this X-ray image and list all detected items.",
        "Describe what you can see in this baggage scan.",
        "What objects are present in this X-ray image?",
    ],
    "specific": [
        "Is there a {item_type} in this scan?",
        "Does this X-ray image contain a {item_type}?",
        "Can you detect a {item_type} in this baggage?",
    ],
    "location": [
        "List all items detected in this scan with their locations.",
        "Where are the items located in this X-ray image?",
        "Describe the position of each item visible in this scan.",
    ],
    "occlusion": [
        "Are there any concealed or partially visible items?",
        "Identify any occluded objects in this X-ray image.",
        "Are any items hidden or overlapping in this scan?",
    ],
    "detailed": [
        "Provide a detailed description of all items in this X-ray scan.",
        "List all detected items with their approximate locations and whether they are concealed.",
    ],
}


def parse_coco_annotations(annotation_file: Path) -> Dict[int, List[Dict]]:
    """Parse COCO format annotations."""
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)
    
    # Group annotations by image_id
    image_annotations = {}
    for ann in coco_data.get("annotations", []):
        image_id = ann["image_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Build image info lookup
    images = {img["id"]: img for img in coco_data.get("images", [])}
    
    return image_annotations, images


def get_bbox_location_description(bbox: List[float], img_width: int, img_height: int) -> str:
    """Convert bounding box to human-readable location description."""
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Determine horizontal position
    if center_x < img_width / 3:
        h_pos = "left"
    elif center_x < 2 * img_width / 3:
        h_pos = "center"
    else:
        h_pos = "right"
    
    # Determine vertical position
    if center_y < img_height / 3:
        v_pos = "upper"
    elif center_y < 2 * img_height / 3:
        v_pos = "middle"
    else:
        v_pos = "lower"
    
    # Combine
    if v_pos == "middle" and h_pos == "center":
        return "center of the image"
    elif v_pos == "middle":
        return f"{h_pos} side"
    elif h_pos == "center":
        return f"{v_pos} section"
    else:
        return f"{v_pos}-{h_pos} quadrant"


def generate_answer_general(annotations: List[Dict], img_info: Dict) -> str:
    """Generate answer for general question (item recognition only)."""
    if not annotations:
        return "No prohibited items detected in this scan."
    
    # Count items by category
    category_counts = {}
    occluded_items = []
    
    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = CATEGORIES.get(cat_id, "Unknown")
        friendly_name = CATEGORY_FRIENDLY.get(cat_name, cat_name.lower())
        
        if friendly_name not in category_counts:
            category_counts[friendly_name] = 0
        category_counts[friendly_name] += 1
        
        # Check if occluded
        if ann.get("occluded", 0) == 1:
            bbox = ann["bbox"]
            location = get_bbox_location_description(
                bbox, img_info.get("width", 1000), img_info.get("height", 1000)
            )
            occluded_items.append((friendly_name, location))
    
    # Build answer - focus on item recognition
    items_list = []
    for item, count in category_counts.items():
        if count == 1:
            items_list.append(f"a {item}")
        else:
            items_list.append(f"{count} {item}s")
    
    answer = f"Detected items: {', '.join(items_list)}."
    
    # Add occlusion information
    if occluded_items:
        answer += " "
        occlusion_desc = []
        for item, loc in occluded_items:
            occlusion_desc.append(f"{item} at {loc} is partially concealed")
        answer += "Note: " + ", ".join(occlusion_desc) + "."
    
    return answer


def generate_answer_specific(annotations: List[Dict], target_item: str, img_info: Dict) -> str:
    """Generate answer for specific item question."""
    matching_anns = []
    
    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = CATEGORIES.get(cat_id, "Unknown")
        friendly_name = CATEGORY_FRIENDLY.get(cat_name, cat_name.lower())
        
        if friendly_name == target_item:
            matching_anns.append(ann)
    
    if not matching_anns:
        return f"No, there is no {target_item} in this scan."
    
    if len(matching_anns) == 1:
        ann = matching_anns[0]
        occluded = ann.get("occluded", 0) == 1
        bbox = ann["bbox"]
        location = get_bbox_location_description(
            bbox, img_info.get("width", 1000), img_info.get("height", 1000)
        )
        
        if occluded:
            return f"Yes, a {target_item} is detected at {location}, partially concealed."
        else:
            return f"Yes, a {target_item} is detected at {location}."
    else:
        return f"Yes, {len(matching_anns)} {target_item}s are detected in this scan."


def generate_answer_location(annotations: List[Dict], img_info: Dict) -> str:
    """Generate answer with location information."""
    if not annotations:
        return "No items detected."
    
    descriptions = []
    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = CATEGORIES.get(cat_id, "Unknown")
        friendly_name = CATEGORY_FRIENDLY.get(cat_name, cat_name.lower())
        bbox = ann["bbox"]
        location = get_bbox_location_description(
            bbox, img_info.get("width", 1000), img_info.get("height", 1000)
        )
        occluded = ann.get("occluded", 0) == 1
        
        desc = f"{friendly_name.capitalize()} in {location}"
        if occluded:
            desc += " (partially concealed)"
        descriptions.append(desc)
    
    answer = "Items detected: " + ", ".join(descriptions) + "."
    
    return answer


def generate_vqa_pair(
    image_id: int,
    image_file: str,
    annotations: List[Dict],
    img_info: Dict,
    question_type: str = None,
) -> Dict[str, Any]:
    """Generate a single VQA pair."""
    if question_type is None:
        question_type = random.choice(["general", "specific", "location", "occlusion"])
    
    # Select question template
    if question_type == "general":
        question = random.choice(QUESTION_TEMPLATES["general"])
        answer = generate_answer_general(annotations, img_info)
    
    elif question_type == "specific":
        # Pick a random category (or one that exists in annotations)
        if annotations and random.random() < 0.7:  # 70% use actual category
            cat_id = random.choice(annotations)["category_id"]
            cat_name = CATEGORIES.get(cat_id, "Folding_Knife")
            target_item = CATEGORY_FRIENDLY.get(cat_name, cat_name.lower())
        else:  # 30% use random category (for negative examples)
            target_item = random.choice(list(CATEGORY_FRIENDLY.values()))
        
        question = random.choice(QUESTION_TEMPLATES["specific"]).format(item_type=target_item)
        answer = generate_answer_specific(annotations, target_item, img_info)
    
    elif question_type == "location":
        question = random.choice(QUESTION_TEMPLATES["location"])
        answer = generate_answer_location(annotations, img_info)
    
    elif question_type == "occlusion":
        question = random.choice(QUESTION_TEMPLATES["occlusion"])
        has_occlusion = any(ann.get("occluded", 0) == 1 for ann in annotations)
        if has_occlusion:
            occlusion_details = []
            for ann in annotations:
                if ann.get("occluded", 0) == 1:
                    cat_id = ann["category_id"]
                    cat_name = CATEGORIES.get(cat_id, "Unknown")
                    friendly_name = CATEGORY_FRIENDLY.get(cat_name, cat_name.lower())
                    bbox = ann["bbox"]
                    location = get_bbox_location_description(
                        bbox, img_info.get("width", 1000), img_info.get("height", 1000)
                    )
                    occlusion_details.append(f"{friendly_name} at {location}")
            answer = f"Yes, partially concealed items detected: {', '.join(occlusion_details)}."
        else:
            if annotations:
                answer = "No, all items are clearly visible without concealment."
            else:
                answer = "No concealed items detected."
    
    elif question_type == "detailed":
        question = random.choice(QUESTION_TEMPLATES["detailed"])
        if not annotations:
            answer = "No items detected in this scan."
        else:
            details = []
            for ann in annotations:
                cat_id = ann["category_id"]
                cat_name = CATEGORIES.get(cat_id, "Unknown")
                friendly_name = CATEGORY_FRIENDLY.get(cat_name, cat_name.lower())
                bbox = ann["bbox"]
                location = get_bbox_location_description(
                    bbox, img_info.get("width", 1000), img_info.get("height", 1000)
                )
                occluded = ann.get("occluded", 0) == 1
                
                item_desc = f"{friendly_name} at {location}"
                if occluded:
                    item_desc += " (partially concealed)"
                details.append(item_desc)
            
            answer = f"Detected {len(annotations)} item(s): {', '.join(details)}."
    
    # Prepare metadata (no declaration info in training data)
    metadata = {
        "image_id": image_id,
        "num_annotations": len(annotations),
        "categories": [CATEGORIES.get(ann["category_id"]) for ann in annotations],
        "has_occlusion": any(ann.get("occluded", 0) == 1 for ann in annotations),
        "question_type": question_type,
        "bboxes": [ann["bbox"] for ann in annotations],
    }
    
    return {
        "image_path": image_file,
        "question": question,
        "answer": answer,
        "metadata": metadata,
    }


def create_vqa_dataset(
    opixray_root: Path,
    annotation_file: Path,
    output_file: Path,
    samples_per_image: int = 2,
):
    """Create VQA dataset from OPIXray annotations."""
    print(f"Processing {annotation_file}...")
    
    image_annotations, images = parse_coco_annotations(annotation_file)
    
    vqa_pairs = []
    
    for image_id, img_info in images.items():
        image_filename = img_info["file_name"]
        image_path = f"data/opixray/images/{image_filename}"
        
        annotations = image_annotations.get(image_id, [])
        
        # Generate multiple VQA pairs per image
        for _ in range(samples_per_image):
            vqa_pair = generate_vqa_pair(image_id, image_path, annotations, img_info)
            vqa_pairs.append(vqa_pair)
    
    # Write to JSONL
    print(f"Writing {len(vqa_pairs)} VQA pairs to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for pair in vqa_pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"✓ Created {len(vqa_pairs)} VQA pairs from {len(images)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Create VQA pairs from OPIXray annotations"
    )
    parser.add_argument(
        "--opixray-root",
        type=str,
        default="data/opixray",
        help="Root directory of OPIXray dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/opixray_vqa.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to process",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=2,
        help="Number of VQA pairs to generate per image",
    )
    
    args = parser.parse_args()
    
    opixray_root = Path(args.opixray_root)
    
    # Determine which annotation files to process
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]
    
    for split in splits:
        annotation_file = opixray_root / "annotations" / f"{split}.json"
        
        if not annotation_file.exists():
            print(f"⚠ Annotation file not found: {annotation_file}")
            print("Please download the dataset first using: python data/download_opixray.py")
            continue
        
        if args.split == "all":
            output_file = Path(args.output).parent / f"opixray_vqa_{split}.jsonl"
        else:
            output_file = Path(args.output)
        
        create_vqa_dataset(opixray_root, annotation_file, output_file, args.samples_per_image)
    
    print("\n✓ VQA dataset creation complete!")
    print("\nNote: This dataset focuses on item recognition only.")
    print("Declaration comparison will be handled in post-processing.")
    print("\nNext step: python data/split_dataset.py")


if __name__ == "__main__":
    main()
