#!/usr/bin/env python3
"""
Visualize VLM training samples with bounding box overlays.

Renders N random samples from a VQA JSONL file with ground-truth bounding
boxes drawn on the images. Use this to inspect data quality before training.

Usage:
  python scripts/visualize_vlm_samples.py \
      --data data/stcray_vlm/stcray_vlm_train.jsonl \
      --n 20 --output data_check/
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Colors for different threat levels
THREAT_COLORS = {
    "critical": (255, 0, 0),      # Red
    "high": (255, 128, 0),        # Orange
    "medium": (255, 255, 0),      # Yellow
    "low": (0, 200, 0),           # Green
    "none": (128, 128, 128),      # Gray
    "unknown": (200, 200, 200),   # Light gray
}


def draw_bbox(draw, bbox, label, color, img_width, img_height):
    """Draw a bounding box with label on the image."""
    x1 = bbox[0] * img_width
    y1 = bbox[1] * img_height
    x2 = bbox[2] * img_width
    y2 = bbox[3] * img_height

    # Draw rectangle (2px border)
    for offset in range(2):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color,
        )

    # Draw label background
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((x1, y1), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
    draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)


def visualize_sample(entry, output_dir, idx):
    """Visualize a single VQA sample with bounding boxes."""
    image_path = entry["image_path"]
    answer = json.loads(entry["answer"])
    objects = answer.get("objects", [])

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  WARNING: Could not load {image_path}: {e}")
        return False

    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    for obj in objects:
        category = obj["category"]
        bbox = obj["bbox"]
        threat_level = obj.get("threat_level", "unknown")
        color = THREAT_COLORS.get(threat_level, THREAT_COLORS["unknown"])
        label = f"{category} [{threat_level}]"
        draw_bbox(draw, bbox, label, color, img_width, img_height)

    # Save
    filename = f"sample_{idx:03d}_{Path(image_path).stem}.jpg"
    output_path = output_dir / filename
    img.save(output_path, quality=90)
    num_objects = len(objects)
    categories = [o["category"] for o in objects]
    print(f"  [{idx:3d}] {num_objects} objects: {', '.join(categories)} -> {filename}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize VLM training samples with bounding boxes"
    )
    parser.add_argument("--data", required=True, help="Path to VQA JSONL file")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument("--output", default="data_check/", help="Output directory for images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}...")
    entries = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    print(f"  Total entries: {len(entries)}")

    # Sample
    random.seed(args.seed)
    n = min(args.n, len(entries))
    sampled = random.sample(entries, n)

    print(f"\nVisualizing {n} random samples...")
    print(f"Output: {output_dir.resolve()}\n")

    success = 0
    for idx, entry in enumerate(sampled):
        if visualize_sample(entry, output_dir, idx):
            success += 1

    print(f"\nDone: {success}/{n} samples rendered to {output_dir}/")
    print("Manually inspect the images to verify:")
    print("  - Bounding boxes align with visible objects")
    print("  - Categories match the items in the image")
    print("  - No systematic annotation errors")


if __name__ == "__main__":
    main()
