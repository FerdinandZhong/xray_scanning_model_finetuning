#!/usr/bin/env python3
"""
Combine X-ray datasets for improved YOLO training.

Merges luggage_xray_yolo (6,164 images, 12 classes) with STCray
(30,044 images, 24 threat categories), mapping compatible threat classes
to create a larger unified dataset.

Source datasets:
  - luggage_xray_yolo:   6,164 train + 956 val  (12 classes, all threats)
  - stcray_processed:   30,044 train + 16,598 test (24 categories, threats only used)

Output:
  - data/combined_xray_yolo/  ~32K+ train images, unified class set

Usage:
  python scripts/combine_xray_datasets.py              # default: luggage + stcray
  python scripts/combine_xray_datasets.py --dry-run    # preview without writing
  python scripts/combine_xray_datasets.py --classes-only  # print class mapping
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image

# ─────────────────────────────────────────────
# Unified class definitions
# ─────────────────────────────────────────────

# Base 12 classes from luggage_xray (kept as-is)
LUGGAGE_CLASSES = [
    "blade", "Cans", "CartonDrinks", "dagger", "GlassBottle",
    "knife", "PlasticBottle", "scissors", "SprayCans",
    "SwissArmyKnife", "Tin", "VacuumCup",
]

# Additional threat classes from STCray not already in luggage_xray
STCRAY_EXTRA_CLASSES = [
    "Gun", "Bullet", "Explosive", "Handcuffs",
]

# Full unified class list: luggage classes + stcray extras
UNIFIED_CLASSES = LUGGAGE_CLASSES + STCRAY_EXTRA_CLASSES

THREAT_CLASSES = {
    "blade", "dagger", "knife", "scissors", "SwissArmyKnife",
    "Gun", "Bullet", "Explosive", "Handcuffs",
}

# ─────────────────────────────────────────────
# STCray → unified class mapping
# ─────────────────────────────────────────────
# Maps STCray category name to unified class name (None = skip this category)
STCRAY_CLASS_MAP: Dict[str, Optional[str]] = {
    # Direct matches
    "Knife":            "knife",
    "Scissors":         "scissors",
    "Blade":            "blade",
    # Close-enough mappings (sharp cutting tools)
    "Cutter":           "blade",
    "Other Sharp Item": "blade",
    # Unique STCray threats added to unified set
    "Gun":              "Gun",
    "3D printed gun":   "Gun",
    "3D Gun":           "Gun",
    "Bullet":           "Bullet",
    "Explosive":        "Explosive",
    "Handcuffs":        "Handcuffs",
    # Skip: no equivalent in luggage task scope
    "Battery":          None,
    "Wrench":           None,
    "Hammer":           None,
    "Screwdriver":      None,
    "Pliers":           None,
    "Powerbank":        None,
    "Nail Cutter":      None,
    "Shaving Razor":    None,
    "Injection":        None,
    "Syringe":          None,
    "Lighter":          None,
    "Non Threat":       None,
    "Multilabel Threat": None,  # ambiguous, skip
}


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Return (width, height) of image."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert [x, y, w, h] pixel bbox to YOLO normalized [cx, cy, w, h]."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def copy_luggage_xray(
    src_root: Path,
    out_root: Path,
    class_id_offset: int,
    stats: Counter,
    dry_run: bool = False,
):
    """
    Copy luggage_xray_yolo into the combined dataset.
    Class IDs remain the same (luggage classes are the base).
    """
    print("\n── luggage_xray_yolo ──────────────────────────")

    for split, out_split in [("train", "train"), ("valid", "valid")]:
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split
        out_img_dir = out_root / "images" / out_split
        out_lbl_dir = out_root / "labels" / out_split

        if not img_dir.exists():
            print(f"  ⚠ {img_dir} not found, skipping {split}")
            continue

        images = sorted(img_dir.glob("*.jpg"))
        copied = 0

        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            out_name = f"lug_{img_path.name}"
            if not dry_run:
                shutil.copy2(img_path, out_img_dir / out_name)
                shutil.copy2(lbl_path, out_lbl_dir / out_name.replace(".jpg", ".txt"))

            # Track per-class stats
            for line in lbl_path.read_text().strip().splitlines():
                if line.strip():
                    cid = int(line.split()[0])
                    stats[f"{out_split}/{LUGGAGE_CLASSES[cid]}"] += 1

            copied += 1

        print(f"  {split:5s}: {copied:5d} images")


def convert_stcray(
    ann_file: Path,
    raw_images_root: Path,
    out_root: Path,
    out_split: str,
    stats: Counter,
    dry_run: bool = False,
    max_per_class: Optional[int] = None,
) -> int:
    """
    Convert STCray annotations to YOLO format and copy images.

    Args:
        ann_file:        Path to stcray_processed/{split}/annotations.json
        raw_images_root: Root dir containing Class XX_Name/image.jpg
        out_root:        Combined dataset root
        out_split:       "train" or "valid"
        stats:           Counter for tracking instances added
        dry_run:         Print only, don't write files
        max_per_class:   Cap images per mapped class (balance dataset)
    """
    if not ann_file.exists():
        print(f"  ⚠ {ann_file} not found, skipping")
        return 0

    with open(ann_file) as f:
        data = json.load(f)

    class_to_id = {name: i for i, name in enumerate(UNIFIED_CLASSES)}
    out_img_dir = out_root / "images" / out_split
    out_lbl_dir = out_root / "labels" / out_split

    # Group entries by mapped class for optional capping
    class_buckets: Dict[str, List[dict]] = defaultdict(list)
    for entry in data:
        for cat in entry["categories"]:
            mapped = STCRAY_CLASS_MAP.get(cat)
            if mapped is not None:
                class_buckets[mapped].append(entry)
                break  # one image → one class bucket

    # Apply per-class cap
    selected = []
    for mapped_cls, entries in class_buckets.items():
        if max_per_class and len(entries) > max_per_class:
            entries = random.sample(entries, max_per_class)
        selected.extend(entries)

    # Deduplicate (some entries may appear in multiple buckets above)
    seen_ids = set()
    unique = []
    for entry in selected:
        if entry["image_id"] not in seen_ids:
            seen_ids.add(entry["image_id"])
            unique.append(entry)

    random.shuffle(unique)
    converted = 0
    skipped_no_img = 0
    skipped_no_bbox = 0

    for entry in unique:
        # Resolve absolute image path
        img_abs = Path(entry["image_path_absolute"])
        if not img_abs.exists():
            # Try relative to project root
            img_abs = Path(entry["image_path_absolute"].lstrip("/"))
        if not img_abs.exists():
            skipped_no_img += 1
            continue

        # Map all category labels for this image
        yolo_lines = []
        try:
            img_w, img_h = get_image_size(img_abs)
        except Exception:
            skipped_no_img += 1
            continue

        # Each entry has one primary category and bboxes
        primary_cats = [
            STCRAY_CLASS_MAP.get(c) for c in entry["categories"]
            if STCRAY_CLASS_MAP.get(c) is not None
        ]
        if not primary_cats:
            skipped_no_bbox += 1
            continue

        bboxes = entry.get("bboxes", [])
        if not bboxes:
            skipped_no_bbox += 1
            continue

        for bbox, mapped_cls in zip(bboxes, primary_cats * len(bboxes)):
            cid = class_to_id[mapped_cls]
            cx, cy, nw, nh = bbox_to_yolo(bbox, img_w, img_h)
            if nw > 0 and nh > 0:
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats[f"{out_split}/{mapped_cls}"] += 1

        if not yolo_lines:
            skipped_no_bbox += 1
            continue

        out_name = f"stc_{entry['image_id']:06d}.jpg"
        if not dry_run:
            shutil.copy2(img_abs, out_img_dir / out_name)
            (out_lbl_dir / out_name.replace(".jpg", ".txt")).write_text(
                "\n".join(yolo_lines) + "\n"
            )

        converted += 1

    print(f"  {out_split:5s}: {converted:5d} images  "
          f"(skipped: {skipped_no_img} no-image, {skipped_no_bbox} no-bbox)")
    return converted


def build_combined_dataset(
    project_root: Path,
    out_name: str = "combined_xray_yolo",
    val_ratio: float = 0.1,
    max_stcray_per_class: Optional[int] = None,
    dry_run: bool = False,
    seed: int = 42,
):
    """
    Build the combined YOLO dataset from luggage_xray + STCray.

    Output structure:
        data/{out_name}/
            images/train/   *.jpg
            images/valid/   *.jpg
            labels/train/   *.txt
            labels/valid/   *.txt
            data.yaml
    """
    random.seed(seed)

    out_root = project_root / "data" / out_name
    lug_root = project_root / "data" / "luggage_xray_yolo"
    stcray_root = project_root / "data" / "stcray_processed"
    stcray_raw = project_root / "data" / "stcray_raw"

    print("=" * 65)
    print("COMBINING X-RAY DATASETS")
    print("=" * 65)
    print(f"\nOutput:   {out_root}")
    print(f"Classes:  {len(UNIFIED_CLASSES)}")
    print(f"  Base (luggage_xray): {LUGGAGE_CLASSES}")
    print(f"  Extra (stcray):      {STCRAY_EXTRA_CLASSES}")
    print(f"Dry run:  {dry_run}")

    # ── Print class mapping ──────────────────────────────────────
    print("\n── STCray class mapping ────────────────────────────────")
    used = {v for v in STCRAY_CLASS_MAP.values() if v is not None}
    skipped = {k for k, v in STCRAY_CLASS_MAP.items() if v is None}
    for src, dst in sorted(STCRAY_CLASS_MAP.items(), key=lambda x: (x[1] is None, x[0])):
        if dst:
            print(f"  ✅  {src:<25} → {dst}")
        else:
            print(f"  ⏭   {src:<25} → (skipped)")

    if dry_run:
        print("\n[DRY RUN] Exiting before writing files.")
        return

    # ── Create output directories ────────────────────────────────
    for split in ("train", "valid"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()

    # ── 1. Copy luggage_xray ─────────────────────────────────────
    print("\n[1/2] Copying luggage_xray_yolo...")
    copy_luggage_xray(lug_root, out_root, class_id_offset=0, stats=stats, dry_run=dry_run)

    # ── 2. Convert STCray ────────────────────────────────────────
    print("\n[2/2] Converting STCray (threat classes only)...")

    stcray_train_ann = stcray_root / "train" / "annotations.json"
    stcray_test_ann  = stcray_root / "test"  / "annotations.json"

    # STCray train → combined train
    print(f"\n── STCray train ────────────────────────────────────────")
    convert_stcray(
        stcray_train_ann, stcray_raw, out_root, "train",
        stats, dry_run, max_stcray_per_class,
    )

    # STCray test  → combined valid (provides extra validation diversity)
    print(f"\n── STCray test (used as additional validation) ─────────")
    convert_stcray(
        stcray_test_ann, stcray_raw, out_root, "valid",
        stats, dry_run, max_stcray_per_class,
    )

    # ── Write data.yaml ──────────────────────────────────────────
    data_yaml = {
        "path":   str(out_root.absolute()),
        "train":  "images/train",
        "val":    "images/valid",
        "nc":     len(UNIFIED_CLASSES),
        "names":  UNIFIED_CLASSES,
        "threats": sorted(THREAT_CLASSES),
        "sources": ["luggage_xray_yolo", "stcray_processed"],
    }
    if not dry_run:
        with open(out_root / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # ── Print summary ─────────────────────────────────────────────
    train_imgs = len(list((out_root / "images" / "train").glob("*.jpg")))
    val_imgs   = len(list((out_root / "images" / "valid").glob("*.jpg")))

    print("\n" + "=" * 65)
    print("DATASET COMBINED SUCCESSFULLY")
    print("=" * 65)
    print(f"\n  Train images: {train_imgs:,}")
    print(f"  Val images:   {val_imgs:,}")
    print(f"  Total:        {train_imgs + val_imgs:,}")
    print(f"  Classes:      {len(UNIFIED_CLASSES)}")
    print(f"\n  Output: {out_root}")

    print("\n── Per-class instance counts ────────────────────────────")
    print(f"  {'Class':<22}  {'Train':>6}  {'Val':>5}  {'Threat':>6}")
    for cls in UNIFIED_CLASSES:
        tr = stats[f"train/{cls}"]
        va = stats[f"valid/{cls}"]
        t  = "✅" if cls in THREAT_CLASSES else ""
        print(f"  {cls:<22}  {tr:6d}  {va:5d}  {t}")

    print(f"\n  data.yaml: {out_root / 'data.yaml'}")
    print("\n── Training command ─────────────────────────────────────")
    print(f"  # Update jobs_config_yolo.yaml:")
    print(f"  DATASET: \"{out_name}\"")
    print(f"  # Or train locally:")
    print(f"  python training/train_yolo.py \\")
    print(f"    --data data/{out_name}/data.yaml \\")
    print(f"    --model yolov8m.pt --epochs 200 --batch 16")


def main():
    parser = argparse.ArgumentParser(
        description="Combine luggage_xray + STCray datasets for YOLO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: combine everything
  python scripts/combine_xray_datasets.py

  # Preview class mapping without writing
  python scripts/combine_xray_datasets.py --dry-run

  # Balance STCray (max 2000 images per class)
  python scripts/combine_xray_datasets.py --max-per-class 2000

  # Custom output name
  python scripts/combine_xray_datasets.py --output combined_v2_yolo
        """,
    )
    parser.add_argument(
        "--output", type=str, default="combined_xray_yolo",
        help="Output dataset name under data/ (default: combined_xray_yolo)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Cap STCray images per mapped class for class balance (e.g. 3000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print class mapping and stats without writing any files",
    )
    parser.add_argument(
        "--classes-only", action="store_true",
        help="Print unified class list and exit",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.classes_only:
        print("Unified classes:")
        for i, cls in enumerate(UNIFIED_CLASSES):
            threat = " ← THREAT" if cls in THREAT_CLASSES else ""
            print(f"  {i:2d}  {cls}{threat}")
        print(f"\nSTCray mapping:")
        for src, dst in sorted(STCRAY_CLASS_MAP.items()):
            print(f"  {src:<25} → {dst or '(skip)'}")
        return

    project_root = Path(__file__).parent.parent
    build_combined_dataset(
        project_root=project_root,
        out_name=args.output,
        max_stcray_per_class=args.max_per_class,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
