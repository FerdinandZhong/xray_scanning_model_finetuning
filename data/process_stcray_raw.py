#!/usr/bin/env python3
"""
Convert STCray raw extracted folders → stcray_processed annotations.

Input structure (after RAR extraction):
  data/stcray_raw/
    STCray_TrainSet/
      Images/Class XX_ClassName/*.jpg
      Json_BB/Class XX_ClassName/*.json   ← Labelme bounding-box format
    STCray_TestSet/
      Images/Class XX_ClassName/*.jpg
      Json_BB/Class XX_ClassName/*.json

Output:
  data/stcray_processed/
    train/annotations.json
    test/annotations.json

Annotation format (compatible with combine_xray_datasets.py):
  [{
    "image_id":          int,
    "image_filename":    str,
    "image_path":        str   (relative to project root),
    "image_path_absolute": str (absolute path on current machine),
    "caption":           "",
    "categories":        [str, ...],   # class labels from Json_BB
    "bboxes":            [[x, y, w, h], ...]  # pixel coords
  }, ...]
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple


def class_from_folder(folder_name: str) -> str:
    """Extract class name from 'Class XX_ClassName' folder name."""
    parts = folder_name.split("_", 1)
    return parts[1] if len(parts) == 2 else folder_name


def labelme_points_to_bbox(points: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert Labelme rectangle points [[x1,y1],[x2,y2]] to [x, y, w, h].
    Also handles > 2 points (polygon) by using bounding box of all points.
    """
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return x1, y1, w, h


def parse_labelme_json(json_path: Path):
    """
    Parse a Labelme annotation JSON file.
    Returns (categories, bboxes, img_w, img_h).
    """
    with open(json_path) as f:
        data = json.load(f)

    categories = []
    bboxes = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") not in ("rectangle", "polygon", None):
            continue
        bbox = labelme_points_to_bbox(shape.get("points", []))
        if bbox is None:
            continue
        label = shape.get("label", "").strip()
        if label:
            categories.append(label)
            bboxes.append(list(bbox))

    img_w = data.get("imageWidth", 0)
    img_h = data.get("imageHeight", 0)
    return categories, bboxes, img_w, img_h


def process_split(
    raw_split_dir: Path,
    output_dir: Path,
    split_name: str,
    project_root: Path,
) -> int:
    """
    Process one split (TrainSet or TestSet) and write annotations.json.
    Returns number of annotations written.
    """
    images_root = raw_split_dir / "Images"
    json_bb_root = raw_split_dir / "Json_BB"

    if not images_root.exists():
        print(f"  ⚠  Images directory not found: {images_root}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    image_id = 0
    skipped = 0

    class_dirs = sorted(images_root.iterdir())
    for class_dir in class_dirs:
        if not class_dir.is_dir():
            continue

        class_name = class_from_folder(class_dir.name)
        bb_dir = json_bb_root / class_dir.name

        for img_path in sorted(class_dir.glob("*.jpg")):
            json_path = bb_dir / (img_path.stem + ".json")

            if json_path.exists():
                try:
                    categories, bboxes, _, _ = parse_labelme_json(json_path)
                except Exception as e:
                    print(f"  ⚠  Failed to parse {json_path.name}: {e}")
                    categories, bboxes = [class_name], []
            else:
                # No bbox annotation — use folder class name, empty bboxes
                categories, bboxes = [class_name], []

            rel_path = img_path.relative_to(project_root)
            annotations.append({
                "image_id":             image_id,
                "image_filename":       img_path.name,
                "image_path":           str(rel_path),
                "image_path_absolute":  str(img_path.resolve()),
                "caption":              "",
                "categories":           categories,
                "bboxes":               bboxes,
            })
            image_id += 1

    ann_file = output_dir / "annotations.json"
    with open(ann_file, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"  ✓ {split_name}: {len(annotations):,} images → {ann_file}")
    return len(annotations)


def main():
    parser = argparse.ArgumentParser(
        description="Convert STCray raw folders to stcray_processed annotations"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/stcray_raw",
        help="Path to stcray_raw directory (default: data/stcray_raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/stcray_processed",
        help="Output directory for processed annotations (default: data/stcray_processed)",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root for relative paths (default: cwd)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    project_root = Path(args.project_root) if args.project_root else Path.cwd()

    print("=" * 60)
    print("Processing STCray raw dataset")
    print("=" * 60)
    print(f"  Raw dir:      {raw_dir.resolve()}")
    print(f"  Output dir:   {output_dir.resolve()}")
    print(f"  Project root: {project_root.resolve()}")
    print()

    splits = [
        ("STCray_TrainSet", "train"),
        ("STCray_TestSet",  "test"),
    ]

    total = 0
    for folder_name, split_name in splits:
        split_raw = raw_dir / folder_name
        split_out = output_dir / split_name
        if not split_raw.exists():
            print(f"  ⚠  {folder_name} not found at {split_raw} — skipping")
            continue
        print(f"Processing {split_name} ({folder_name})...")
        n = process_split(split_raw, split_out, split_name, project_root)
        total += n

    print()
    print("=" * 60)
    print(f"✓ Done — {total:,} total annotations written to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
