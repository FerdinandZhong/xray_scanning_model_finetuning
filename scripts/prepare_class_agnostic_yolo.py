#!/usr/bin/env python3
"""
Prepare class-agnostic YOLO dataset from STCray + HiXray.

Merges all bounding box annotations from both datasets into a single "object"
class for training a localization-only YOLO detector.

Input:
  - data/stcray_processed/{train,test}/annotations.json
  - data/hixray_processed/ (images + annotations)

Output:
  - data/class_agnostic_yolo/images/{train,val}/
  - data/class_agnostic_yolo/labels/{train,val}/
  - data/class_agnostic_yolo/data.yaml

Usage:
  python scripts/prepare_class_agnostic_yolo.py \
      --stcray-dir data/stcray_processed \
      --hixray-dir data/hixray_processed \
      --output-dir data/class_agnostic_yolo
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image


def load_stcray_annotations(stcray_dir: Path, split: str) -> list[dict]:
    """Load STCray annotations and convert bboxes to normalized YOLO format."""
    ann_file = stcray_dir / split / "annotations.json"
    if not ann_file.exists():
        print(f"  Warning: {ann_file} not found, skipping")
        return []

    with open(ann_file) as f:
        annotations = json.load(f)

    entries = []
    for ann in annotations:
        image_path = ann.get("image_path_absolute") or str(stcray_dir / split / "Images" / ann["image_filename"])
        if not Path(image_path).exists():
            # Try relative path
            image_path = str(stcray_dir / split / "Images" / ann["image_filename"])

        bboxes = ann.get("bboxes", [])
        if not bboxes:
            continue

        # Get image dimensions for normalization
        try:
            img = Image.open(image_path)
            img_w, img_h = img.size
            img.close()
        except Exception:
            continue

        yolo_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox  # absolute pixels: x_top_left, y_top_left, width, height
            # Convert to YOLO format: x_center, y_center, width, height (normalized)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0.001, min(1, w_norm))
            h_norm = max(0.001, min(1, h_norm))
            yolo_bboxes.append((0, x_center, y_center, w_norm, h_norm))

        entries.append({
            "image_path": image_path,
            "image_filename": ann["image_filename"],
            "bboxes": yolo_bboxes,
            "source": "stcray",
        })

    return entries


def load_hixray_annotations(hixray_dir: Path) -> list[dict]:
    """Load HiXray annotations. Supports common HiXray directory layouts."""
    entries = []

    # HiXray typical structure: category subdirs with images + XML/JSON annotations
    # Try JSON annotations first
    ann_files = list(hixray_dir.rglob("*.json"))
    if ann_files:
        for ann_file in ann_files:
            if ann_file.name in ("statistics.json", "category_hints.json"):
                continue
            try:
                with open(ann_file) as f:
                    data = json.load(f)
                # Handle list-of-annotations or single annotation
                anns = data if isinstance(data, list) else [data]
                for ann in anns:
                    image_path = ann.get("image_path", "")
                    if not image_path:
                        continue
                    if not Path(image_path).is_absolute():
                        image_path = str(hixray_dir / image_path)
                    bboxes = ann.get("bboxes", ann.get("annotations", []))
                    if not bboxes:
                        continue

                    try:
                        img = Image.open(image_path)
                        img_w, img_h = img.size
                        img.close()
                    except Exception:
                        continue

                    yolo_bboxes = []
                    for bbox in bboxes:
                        if isinstance(bbox, dict):
                            x = bbox.get("x", bbox.get("x1", 0))
                            y = bbox.get("y", bbox.get("y1", 0))
                            w = bbox.get("w", bbox.get("width", 0))
                            h = bbox.get("h", bbox.get("height", 0))
                        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            x, y, w, h = bbox[:4]
                        else:
                            continue

                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        w_norm = max(0.001, min(1, w_norm))
                        h_norm = max(0.001, min(1, h_norm))
                        yolo_bboxes.append((0, x_center, y_center, w_norm, h_norm))

                    if yolo_bboxes:
                        entries.append({
                            "image_path": image_path,
                            "image_filename": Path(image_path).name,
                            "bboxes": yolo_bboxes,
                            "source": "hixray",
                        })
            except (json.JSONDecodeError, KeyError):
                continue

    # Try XML annotations (Pascal VOC format, common for HiXray)
    if not entries:
        xml_files = list(hixray_dir.rglob("*.xml"))
        if xml_files:
            try:
                import xml.etree.ElementTree as ET
                for xml_file in xml_files:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    filename = root.findtext("filename", "")
                    size = root.find("size")
                    if size is None:
                        continue
                    img_w = int(size.findtext("width", "0"))
                    img_h = int(size.findtext("height", "0"))
                    if img_w == 0 or img_h == 0:
                        continue

                    # Find image
                    image_path = None
                    for ext in (".jpg", ".png", ".jpeg"):
                        candidate = xml_file.with_suffix(ext)
                        if candidate.exists():
                            image_path = str(candidate)
                            break
                    if not image_path:
                        img_dir = hixray_dir / "images"
                        if img_dir.exists():
                            for ext in (".jpg", ".png"):
                                candidate = img_dir / (xml_file.stem + ext)
                                if candidate.exists():
                                    image_path = str(candidate)
                                    break
                    if not image_path:
                        continue

                    yolo_bboxes = []
                    for obj in root.findall("object"):
                        bndbox = obj.find("bndbox")
                        if bndbox is None:
                            continue
                        xmin = float(bndbox.findtext("xmin", "0"))
                        ymin = float(bndbox.findtext("ymin", "0"))
                        xmax = float(bndbox.findtext("xmax", "0"))
                        ymax = float(bndbox.findtext("ymax", "0"))

                        w = xmax - xmin
                        h = ymax - ymin
                        x_center = (xmin + w / 2) / img_w
                        y_center = (ymin + h / 2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        w_norm = max(0.001, min(1, w_norm))
                        h_norm = max(0.001, min(1, h_norm))
                        yolo_bboxes.append((0, x_center, y_center, w_norm, h_norm))

                    if yolo_bboxes:
                        entries.append({
                            "image_path": image_path,
                            "image_filename": filename or Path(image_path).name,
                            "bboxes": yolo_bboxes,
                            "source": "hixray",
                        })
            except Exception as e:
                print(f"  Warning: XML parsing error: {e}")

    return entries


def write_yolo_dataset(entries: list[dict], output_dir: Path, split: str):
    """Write entries to YOLO format (images + labels)."""
    img_dir = output_dir / "images" / split
    lbl_dir = output_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for entry in entries:
        src_path = Path(entry["image_path"])
        if not src_path.exists():
            continue

        # Unique filename to avoid collisions between datasets
        prefix = entry["source"][:3]
        dst_name = f"{prefix}_{entry['image_filename']}"
        dst_img = img_dir / dst_name
        dst_lbl = lbl_dir / (Path(dst_name).stem + ".txt")

        # Copy image
        if not dst_img.exists():
            shutil.copy2(src_path, dst_img)

        # Write label file
        with open(dst_lbl, "w") as f:
            for bbox in entry["bboxes"]:
                cls_id, xc, yc, w, h = bbox
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Prepare class-agnostic YOLO dataset")
    parser.add_argument("--stcray-dir", type=str, default="data/stcray_processed")
    parser.add_argument("--hixray-dir", type=str, default="data/hixray_processed")
    parser.add_argument("--output-dir", type=str, default="data/class_agnostic_yolo")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stcray_dir = Path(args.stcray_dir)
    hixray_dir = Path(args.hixray_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Prepare Class-Agnostic YOLO Dataset")
    print("=" * 60)

    # Load STCray
    print("Loading STCray annotations...")
    stcray_train = load_stcray_annotations(stcray_dir, "train")
    stcray_test = load_stcray_annotations(stcray_dir, "test")
    print(f"  STCray: {len(stcray_train)} train, {len(stcray_test)} test")

    # Load HiXray
    print("Loading HiXray annotations...")
    hixray_all = load_hixray_annotations(hixray_dir)
    print(f"  HiXray: {len(hixray_all)} total")

    # Split HiXray into train/val
    random.seed(args.seed)
    random.shuffle(hixray_all)
    split_idx = int(len(hixray_all) * (1 - args.val_ratio))
    hixray_train = hixray_all[:split_idx]
    hixray_val = hixray_all[split_idx:]
    print(f"  HiXray split: {len(hixray_train)} train, {len(hixray_val)} val")

    # Combine
    train_entries = stcray_train + hixray_train
    val_entries = stcray_test + hixray_val
    random.shuffle(train_entries)
    print(f"  Combined: {len(train_entries)} train, {len(val_entries)} val")

    # Write YOLO dataset
    print("\nWriting YOLO dataset...")
    n_train = write_yolo_dataset(train_entries, output_dir, "train")
    n_val = write_yolo_dataset(val_entries, output_dir, "val")
    print(f"  Written: {n_train} train, {n_val} val")

    # Count total bboxes
    total_bboxes = sum(len(e["bboxes"]) for e in train_entries + val_entries)
    print(f"  Total bounding boxes: {total_bboxes}")

    # Write data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["object"],
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        for key, val in data_yaml.items():
            if isinstance(val, list):
                f.write(f"{key}: {val}\n")
            else:
                f.write(f"{key}: {val}\n")

    print(f"\n  data.yaml: {yaml_path}")
    print("=" * 60)
    print("Class-agnostic YOLO dataset prepared successfully")
    print(f"  {n_train + n_val} images, {total_bboxes} bboxes, 1 class ('object')")


if __name__ == "__main__":
    main()
