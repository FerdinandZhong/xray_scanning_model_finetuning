#!/usr/bin/env python3
"""
Process X-Ray Baggage COCO dataset into YOLO format.

Source: data/X-Ray Baggage.coco.zip  (Roboflow COCO export)
  - 1,569 images, 3,304 annotations
  - 5 classes: Gun, Knife, Pliers, Scissors, Wrench
  - Single train split in the zip (no val provided)

Output: data/xray_baggage_yolo/
  images/train/*.jpg
  images/valid/*.jpg
  labels/train/*.txt
  labels/valid/*.txt
  data.yaml

Conversion:
  COCO bbox [x, y, w, h]  (pixel, top-left origin)
  → YOLO  [cx, cy, w, h]  (normalised 0-1, centre origin)

The supercategory id=0 ("Gun-Knife-Pliers-Scissors-Wrench") is skipped —
it is a Roboflow grouping label, not a real annotation class.

Usage (local):
  python data/process_xray_baggage.py

Usage (CAI job via download_xray_baggage.py):
  Called automatically — do not run directly on CAI.

Environment variables:
  VAL_RATIO        Fraction of images held out for validation (default: 0.1)
  FORCE_REPROCESS  Set "true" to overwrite existing output (default: false)
"""

import json
import os
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────

ZIP_NAME   = "X-Ray Baggage.coco.zip"
RAW_DIR    = "xray_baggage_raw"
OUTPUT_DIR = "xray_baggage_yolo"

# Category id=0 is the Roboflow supercategory grouping — skip it
SKIP_CATEGORY_ID = 0

# Class list in the order they will appear in data.yaml (ids 0-4)
CLASSES = ["Gun", "Knife", "Pliers", "Scissors", "Wrench"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def coco_to_yolo(bbox, img_w: int, img_h: int):
    """Convert COCO [x, y, w, h] pixels → YOLO [cx, cy, w, h] normalised."""
    x, y, bw, bh = [float(v) for v in bbox]
    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def image_size(path: Path):
    with Image.open(path) as img:
        return img.size  # (width, height)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    val_ratio      = float(os.getenv("VAL_RATIO", "0.1"))
    force_reprocess = os.getenv("FORCE_REPROCESS", "false").lower() == "true"
    seed           = 42

    project_root = Path(__file__).parent.parent
    data_dir     = project_root / "data"
    zip_path     = data_dir / ZIP_NAME
    raw_dir      = data_dir / RAW_DIR
    out_dir      = data_dir / OUTPUT_DIR

    print("=" * 65)
    print("X-Ray Baggage COCO → YOLO Conversion")
    print("=" * 65)
    print(f"  Zip:       {zip_path}")
    print(f"  Raw dir:   {raw_dir}")
    print(f"  Output:    {out_dir}")
    print(f"  Val ratio: {val_ratio:.0%}")
    print()

    if not zip_path.exists():
        print(f"❌ Zip not found: {zip_path}")
        sys.exit(1)

    # ── Idempotency check ─────────────────────────────────────────────────────
    data_yaml_path = out_dir / "data.yaml"
    if data_yaml_path.exists() and not force_reprocess:
        tr = len(list((out_dir / "images" / "train").glob("*.jpg")))
        va = len(list((out_dir / "images" / "valid").glob("*.jpg")))
        print(f"⚠  Output already exists: {tr:,} train / {va:,} val images")
        print("   Set FORCE_REPROCESS=true to rebuild.")
        return

    # ── Extract zip ───────────────────────────────────────────────────────────
    print("[1/4] Extracting zip...")
    if raw_dir.exists() and not force_reprocess:
        imgs_found = len(list(raw_dir.rglob("*.jpg")))
        print(f"  ✓ Raw dir exists ({imgs_found:,} images), skipping extraction")
    else:
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
        imgs_found = len(list(raw_dir.rglob("*.jpg")))
        print(f"  ✓ Extracted {imgs_found:,} images to {raw_dir}")

    # ── Load COCO annotation ──────────────────────────────────────────────────
    print("\n[2/4] Loading COCO annotations...")
    ann_file = raw_dir / "train" / "_annotations.coco.json"
    if not ann_file.exists():
        print(f"❌ Annotation file not found: {ann_file}")
        sys.exit(1)

    with open(ann_file) as f:
        coco = json.load(f)

    # Build category id → zero-based class index (skip supercategory id=0)
    cat_id_to_cls: dict = {}
    for cat in coco["categories"]:
        if cat["id"] == SKIP_CATEGORY_ID:
            continue
        name = cat["name"]
        if name in CLASSES:
            cat_id_to_cls[cat["id"]] = CLASSES.index(name)
        else:
            print(f"  ⚠ Unknown category '{name}' (id={cat['id']}), skipping")

    print(f"  Categories mapped: {cat_id_to_cls}")

    # Build image_id → image info
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Build image_id → list of annotations
    id_to_anns: dict = defaultdict(list)
    skipped_cat = 0
    for ann in coco["annotations"]:
        if ann["category_id"] not in cat_id_to_cls:
            skipped_cat += 1
            continue
        id_to_anns[ann["image_id"]].append(ann)

    print(f"  Images with annotations: {len(id_to_anns):,}")
    print(f"  Annotations: {len(coco['annotations']):,}  "
          f"(skipped supercategory: {skipped_cat})")

    # ── Stratified train/val split ────────────────────────────────────────────
    print(f"\n[3/4] Splitting {val_ratio:.0%} val per class (stratified)...")
    random.seed(seed)

    # Group image_ids by their primary class (first annotation)
    class_to_imgs: dict = defaultdict(list)
    for img_id, anns in id_to_anns.items():
        primary_cls = cat_id_to_cls[anns[0]["category_id"]]
        class_to_imgs[primary_cls].append(img_id)

    train_ids, val_ids = [], []
    for cls_idx, img_ids in class_to_imgs.items():
        random.shuffle(img_ids)
        n_val = max(1, int(len(img_ids) * val_ratio))
        val_ids.extend(img_ids[:n_val])
        train_ids.extend(img_ids[n_val:])

    # Images without any valid annotations go to train
    all_annotated = set(id_to_anns.keys())
    unannotated = set(id_to_img.keys()) - all_annotated
    print(f"  Images without valid annotations: {len(unannotated)} (skipped)")
    print(f"  Train: {len(train_ids):,}   Val: {len(val_ids):,}")

    # ── Write YOLO dataset ────────────────────────────────────────────────────
    print("\n[4/4] Writing YOLO format files...")
    for split in ("train", "valid"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_src_dir = raw_dir / "train"
    stats = defaultdict(int)
    skipped_no_img = 0
    skipped_no_bbox = 0

    for split_name, img_ids in [("train", train_ids), ("valid", val_ids)]:
        converted = 0
        for img_id in img_ids:
            img_info  = id_to_img[img_id]
            src_img   = img_src_dir / img_info["file_name"]

            if not src_img.exists():
                skipped_no_img += 1
                continue

            anns = id_to_anns.get(img_id, [])
            if not anns:
                continue

            try:
                img_w = img_info.get("width")
                img_h = img_info.get("height")
                if not img_w or not img_h:
                    img_w, img_h = image_size(src_img)
            except Exception:
                skipped_no_img += 1
                continue

            yolo_lines = []
            for ann in anns:
                cls_idx = cat_id_to_cls.get(ann["category_id"])
                if cls_idx is None:
                    continue
                bbox = ann["bbox"]
                if not bbox or len(bbox) < 4:
                    skipped_no_bbox += 1
                    continue
                cx, cy, nw, nh = coco_to_yolo(bbox, img_w, img_h)
                if nw > 0 and nh > 0:
                    yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    stats[f"{split_name}/{CLASSES[cls_idx]}"] += 1

            if not yolo_lines:
                skipped_no_bbox += 1
                continue

            dst_img = out_dir / "images" / split_name / img_info["file_name"]
            dst_lbl = out_dir / "labels" / split_name / (src_img.stem + ".txt")
            shutil.copy2(src_img, dst_img)
            dst_lbl.write_text("\n".join(yolo_lines) + "\n")
            converted += 1

        print(f"  {split_name:5s}: {converted:4d} images written")

    if skipped_no_img:
        print(f"  ⚠ Skipped {skipped_no_img} images (file not found)")
    if skipped_no_bbox:
        print(f"  ⚠ Skipped {skipped_no_bbox} annotations (bad bbox)")

    # ── Write data.yaml ───────────────────────────────────────────────────────
    data_yaml = {
        "path":    str(out_dir.absolute()),
        "train":   "images/train",
        "val":     "images/valid",
        "nc":      len(CLASSES),
        "names":   CLASSES,
        "source":  "X-Ray Baggage (Roboflow COCO export)",
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    tr_imgs = len(list((out_dir / "images" / "train").glob("*.jpg")))
    va_imgs = len(list((out_dir / "images" / "valid").glob("*.jpg")))

    print("\n" + "=" * 65)
    print("CONVERSION COMPLETE")
    print("=" * 65)
    print(f"  Train images: {tr_imgs:,}")
    print(f"  Val   images: {va_imgs:,}")
    print(f"  Classes: {CLASSES}")
    print()
    print(f"  {'Class':<12}  {'Train':>6}  {'Val':>5}")
    for cls in CLASSES:
        tr = stats[f"train/{cls}"]
        va = stats[f"valid/{cls}"]
        print(f"  {cls:<12}  {tr:6d}  {va:5d}")
    print(f"\n  Output: {out_dir}")


if __name__ == "__main__":
    main()
