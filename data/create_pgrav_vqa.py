#!/usr/bin/env python3
"""
Generate dual-format PG-RAV VQA training data for Phase B.

Runs class-agnostic YOLO on training images to collect proposals, then generates
two VQA formats per image:
  Format 1 (full-scene): proposals as spatial tokens → per-region classification
  Format 2 (focused ROI): cropped region + category hints → single-item classification

Input:
  - Class-agnostic YOLO model (best.pt from Phase B.1)
  - STCray processed annotations (with ground-truth categories)
  - HiXray processed annotations

Output:
  - data/pgrav_vqa/pgrav_train.jsonl
  - data/pgrav_vqa/pgrav_val.jsonl
  - data/pgrav_vqa/pgrav_test.jsonl
  - data/pgrav_vqa/statistics.json

Usage:
  python data/create_pgrav_vqa.py \
      --yolo-model runs/detect/class_agnostic_xray/weights/best.pt \
      --stcray-dir data/stcray_processed \
      --hixray-dir data/hixray_processed \
      --output-dir data/pgrav_vqa
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

from PIL import Image


def bbox_to_location(x_center: float, y_center: float) -> str:
    """Convert normalized center coordinates to location string."""
    h = "left" if x_center < 0.33 else ("right" if x_center > 0.67 else "center")
    v = "upper" if y_center < 0.33 else ("lower" if y_center > 0.67 else "center")
    if v == "center" and h == "center":
        return "center"
    if v == "center":
        return h
    if h == "center":
        return v
    return f"{v}-{h}"


# Known confusable pairs for focused ROI hints
CONFUSABLES = {
    "Knife": ["Dagger", "Blade", "Scissors"],
    "Dagger": ["Knife", "Blade", "Other Sharp Item"],
    "Blade": ["Knife", "Dagger", "Scissors"],
    "Gun": ["3D Gun", "Wrench", "Pliers"],
    "3D Gun": ["Gun", "Lighter", "Powerbank"],
    "Scissors": ["Pliers", "Knife", "Nail Cutter"],
    "Pliers": ["Scissors", "Wrench", "Hammer"],
    "Wrench": ["Pliers", "Hammer", "Screwdriver"],
    "Screwdriver": ["Wrench", "Injection", "Nail Cutter"],
    "Hammer": ["Wrench", "Pliers", "Powerbank"],
    "Lighter": ["Battery", "Powerbank", "3D Gun"],
    "Battery": ["Powerbank", "Lighter", "Phone"],
    "Powerbank": ["Battery", "Phone", "Lighter"],
    "Injection": ["Screwdriver", "Nail Cutter", "Other Sharp Item"],
    "Nail Cutter": ["Scissors", "Injection", "Other Sharp Item"],
    "Handcuffs": ["Pliers", "Wrench", "Other Sharp Item"],
    "Bullet": ["Battery", "Lighter", "Nail Cutter"],
    "Explosive": ["Powerbank", "Battery", "Lighter"],
    "Other Sharp Item": ["Knife", "Dagger", "Blade"],
    # HiXray everyday items
    "Laptop": ["Tablet", "Book", "Cutting Board"],
    "Phone": ["Powerbank", "Battery", "Calculator"],
    "Tablet": ["Laptop", "Book", "Phone"],
    "Cosmetic": ["Bottle", "Medicine", "Food Container"],
    "Water": ["Bottle", "Thermos", "Can"],
    "Umbrella": ["Wrench", "Baton", "Stick"],
    "Portable_Charger_1": ["Powerbank", "Battery", "Phone"],
    "Portable_Charger_2": ["Powerbank", "Battery", "Laptop Charger"],
}


def get_confusable_hints(category: str, k: int = 3) -> list[str]:
    """Get confusable category names for a given category."""
    return CONFUSABLES.get(category, ["unknown_item", "other_object", "miscellaneous"])[:k]


def run_yolo_on_image(yolo_model, image_path: str) -> list[dict]:
    """Run YOLO on an image and return proposals."""
    results = yolo_model.predict(image_path, conf=0.1, verbose=False)
    proposals = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x_c, y_c, w, h = box.xywhn[0].tolist()
            conf = float(box.conf[0])
            proposals.append({
                "bbox": [x_c, y_c, w, h],
                "confidence": round(conf, 2),
                "location": bbox_to_location(x_c, y_c),
            })
    return proposals


def create_full_scene_entry(image_path: str, proposals: list[dict],
                            gt_categories: list[str], gt_bboxes: list) -> dict:
    """Create Format 1: full-scene analysis with proposals."""
    # Build proposal prompt
    region_strs = []
    for i, prop in enumerate(proposals, 1):
        loc = prop["location"]
        conf = prop["confidence"]
        region_strs.append(f"[R{i}: {loc}, {conf:.2f} conf]")

    if region_strs:
        proposal_text = "Objects detected at: " + ", ".join(region_strs) + "."
    else:
        proposal_text = "No objects detected by the detector."

    prompt = (
        f"{proposal_text}\n"
        "Classify each detected region and identify any additional items not detected. "
        "Respond with JSON: {\"objects\": [{\"category\": \"...\", \"location\": \"...\"}], "
        "\"additional_items\": [{\"category\": \"...\"}]}"
    )

    # Build answer from ground truth
    objects = []
    for cat, bbox in zip(gt_categories, gt_bboxes):
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            loc = bbox_to_location(bbox[0], bbox[1])
        else:
            loc = "unknown"
        objects.append({"category": cat, "location": loc})

    answer = json.dumps({"objects": objects, "additional_items": []})

    return {
        "image_path": image_path,
        "question": prompt,
        "answer": answer,
        "metadata": {
            "format": "full_scene_proposals",
            "num_proposals": len(proposals),
            "num_gt_objects": len(gt_categories),
        },
    }


def create_focused_roi_entry(image_path: str, category: str, bbox: list,
                             img_w: int, img_h: int) -> dict:
    """Create Format 2: focused ROI re-analysis with category hints."""
    hints = get_confusable_hints(category)

    prompt = (
        f"Identify this item in the X-ray crop.\n"
        f"Similar categories: {', '.join(hints)}.\n"
        "Classify this item. Respond with JSON: {\"category\": \"...\", \"confidence\": 0.0}"
    )

    answer = json.dumps({"category": category, "confidence": 0.95})

    # Compute ROI crop coordinates (normalized)
    if len(bbox) >= 4:
        x, y, w, h = bbox[:4]
        # Add padding
        pad = 0.1
        x1 = max(0, (x - pad * w)) / img_w if img_w > 0 else 0
        y1 = max(0, (y - pad * h)) / img_h if img_h > 0 else 0
        x2 = min(img_w, (x + w + pad * w)) / img_w if img_w > 0 else 1
        y2 = min(img_h, (y + h + pad * h)) / img_h if img_h > 0 else 1
    else:
        x1, y1, x2, y2 = 0, 0, 1, 1

    return {
        "image_path": image_path,
        "question": prompt,
        "answer": answer,
        "metadata": {
            "format": "roi_focused",
            "ground_truth_category": category,
            "roi_bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            "confusables": hints,
        },
    }


def load_stcray_entries(stcray_dir: Path, split: str) -> list[dict]:
    """Load STCray annotations as raw entries."""
    ann_file = stcray_dir / split / "annotations.json"
    if not ann_file.exists():
        return []

    with open(ann_file) as f:
        annotations = json.load(f)

    entries = []
    for ann in annotations:
        image_path = ann.get("image_path_absolute") or str(
            stcray_dir / split / "Images" / ann["image_filename"]
        )
        categories = ann.get("categories", [])
        bboxes = ann.get("bboxes", [])
        if not categories or not bboxes:
            continue

        entries.append({
            "image_path": image_path,
            "categories": categories,
            "bboxes": bboxes,
            "source": "stcray",
        })

    return entries


def load_hixray_entries(hixray_dir: Path) -> list[dict]:
    """Load HiXray annotations as raw entries."""
    entries = []

    # Try JSON annotations
    for ann_file in hixray_dir.rglob("*.json"):
        if ann_file.name in ("statistics.json", "category_hints.json"):
            continue
        try:
            with open(ann_file) as f:
                data = json.load(f)
            anns = data if isinstance(data, list) else [data]
            for ann in anns:
                image_path = ann.get("image_path", "")
                if not image_path:
                    continue
                if not Path(image_path).is_absolute():
                    image_path = str(hixray_dir / image_path)
                categories = ann.get("categories", [])
                bboxes = ann.get("bboxes", ann.get("annotations", []))
                if categories and bboxes:
                    entries.append({
                        "image_path": image_path,
                        "categories": categories,
                        "bboxes": bboxes,
                        "source": "hixray",
                    })
        except (json.JSONDecodeError, KeyError):
            continue

    # Try XML (Pascal VOC)
    if not entries:
        import xml.etree.ElementTree as ET
        for xml_file in hixray_dir.rglob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename = root.findtext("filename", "")

                image_path = None
                for ext in (".jpg", ".png", ".jpeg"):
                    candidate = xml_file.with_suffix(ext)
                    if candidate.exists():
                        image_path = str(candidate)
                        break

                if not image_path:
                    continue

                categories = []
                bboxes = []
                for obj in root.findall("object"):
                    cat = obj.findtext("name", "unknown")
                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        xmin = float(bndbox.findtext("xmin", "0"))
                        ymin = float(bndbox.findtext("ymin", "0"))
                        xmax = float(bndbox.findtext("xmax", "0"))
                        ymax = float(bndbox.findtext("ymax", "0"))
                        bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                        categories.append(cat)

                if categories:
                    entries.append({
                        "image_path": image_path,
                        "categories": categories,
                        "bboxes": bboxes,
                        "source": "hixray",
                    })
            except Exception:
                continue

    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate PG-RAV dual-format VQA data")
    parser.add_argument("--yolo-model", type=str, required=True, help="Path to class-agnostic YOLO model")
    parser.add_argument("--stcray-dir", type=str, default="data/stcray_processed")
    parser.add_argument("--hixray-dir", type=str, default="data/hixray_processed")
    parser.add_argument("--output-dir", type=str, default="data/pgrav_vqa")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-proposals", type=int, default=20, help="Max YOLO proposals per image")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generate PG-RAV Dual-Format VQA Data")
    print("=" * 60)

    # Load YOLO model
    print(f"Loading YOLO model: {args.yolo_model}")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)

    # Load annotations
    print("Loading STCray annotations...")
    stcray_train = load_stcray_entries(Path(args.stcray_dir), "train")
    stcray_test = load_stcray_entries(Path(args.stcray_dir), "test")
    print(f"  STCray: {len(stcray_train)} train, {len(stcray_test)} test")

    print("Loading HiXray annotations...")
    hixray_all = load_hixray_entries(Path(args.hixray_dir))
    print(f"  HiXray: {len(hixray_all)} total")

    # Split HiXray
    random.shuffle(hixray_all)
    val_idx = int(len(hixray_all) * args.val_ratio)
    test_idx = int(len(hixray_all) * args.val_ratio * 2)
    hixray_val = hixray_all[:val_idx]
    hixray_test = hixray_all[val_idx:test_idx]
    hixray_train = hixray_all[test_idx:]

    splits = {
        "train": stcray_train + hixray_train,
        "val": hixray_val,
        "test": stcray_test + hixray_test,
    }

    stats = {"total": 0, "format_1": 0, "format_2": 0, "categories": Counter()}

    for split_name, entries in splits.items():
        random.shuffle(entries)
        output_file = output_dir / f"pgrav_{split_name}.jsonl"
        n_written = 0

        print(f"\nProcessing {split_name} split ({len(entries)} images)...")

        with open(output_file, "w") as f:
            for i, entry in enumerate(entries):
                image_path = entry["image_path"]
                if not Path(image_path).exists():
                    continue

                try:
                    img = Image.open(image_path)
                    img_w, img_h = img.size
                    img.close()
                except Exception:
                    continue

                categories = entry["categories"]
                bboxes = entry["bboxes"]

                # Run YOLO to get proposals
                proposals = run_yolo_on_image(yolo, image_path)[:args.max_proposals]

                # Format 1: Full-scene with proposals
                # Convert bboxes to normalized centers for location
                gt_centers = []
                for bbox in bboxes:
                    if len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        gt_centers.append([(x + w / 2) / img_w, (y + h / 2) / img_h])
                    else:
                        gt_centers.append([0.5, 0.5])

                full_scene = create_full_scene_entry(
                    image_path, proposals, categories, gt_centers
                )
                f.write(json.dumps(full_scene) + "\n")
                n_written += 1
                stats["format_1"] += 1

                # Format 2: Focused ROI for each annotated object
                for cat, bbox in zip(categories, bboxes):
                    if cat in ("Non Threat", "Multilabel Threat"):
                        continue  # Skip non-informative categories

                    roi_entry = create_focused_roi_entry(
                        image_path, cat, bbox, img_w, img_h
                    )
                    f.write(json.dumps(roi_entry) + "\n")
                    n_written += 1
                    stats["format_2"] += 1
                    stats["categories"][cat] += 1

                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(entries)} images...")

        stats["total"] += n_written
        print(f"  {split_name}: {n_written} samples written to {output_file}")

    # Write statistics
    stats_file = output_dir / "statistics.json"
    stats["categories"] = dict(stats["categories"].most_common())
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PG-RAV VQA data generation complete")
    print(f"  Total samples: {stats['total']}")
    print(f"  Format 1 (full-scene): {stats['format_1']}")
    print(f"  Format 2 (focused ROI): {stats['format_2']}")
    print(f"  Categories: {len(stats['categories'])}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
