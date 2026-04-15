#!/usr/bin/env python3
"""
Build static category hints lookup from STCray + HiXray annotations.

Generates a JSON file mapping categories to descriptions, visual cues, and
confusable pairs for Phase C inference-time retrieval-as-text.

Output schema:
  {
    "knife": {
      "description": "Sharp blade with handle, elongated shape",
      "visual_cues": ["pointed tip", "elongated", "high contrast edge"],
      "confusables": ["dagger", "blade", "scissors"],
      "source": "stcray"
    },
    ...
  }

Usage:
  python scripts/build_category_hints.py \
      --stcray-dir data/stcray_processed \
      --hixray-dir data/hixray_processed \
      --output data/category_hints.json
"""

import argparse
import json
from pathlib import Path


# Curated category hints — visual characteristics in X-ray imagery
# These are domain-specific descriptions of how items appear under X-ray scanning
CATEGORY_HINTS = {
    # === STCray threat categories (21) ===
    "Explosive": {
        "description": "Dense irregular mass, often with wiring or detonator components",
        "visual_cues": ["irregular dense mass", "wiring visible", "cylindrical or block shape"],
        "confusables": ["Powerbank", "Battery", "Lighter"],
        "source": "stcray",
    },
    "Gun": {
        "description": "Metallic L-shaped object with barrel and trigger mechanism",
        "visual_cues": ["L-shape", "barrel visible", "trigger guard", "high density metal"],
        "confusables": ["3D Gun", "Wrench", "Pliers"],
        "source": "stcray",
    },
    "3D Gun": {
        "description": "Plastic firearm with lower density than metal gun, may have irregular shape",
        "visual_cues": ["gun-like shape", "lower density than metal", "possible layer lines"],
        "confusables": ["Gun", "Lighter", "Powerbank"],
        "source": "stcray",
    },
    "Knife": {
        "description": "Elongated blade with handle, sharp edge visible as high contrast line",
        "visual_cues": ["elongated shape", "pointed tip", "high contrast edge", "handle visible"],
        "confusables": ["Dagger", "Blade", "Scissors"],
        "source": "stcray",
    },
    "Dagger": {
        "description": "Double-edged blade, typically shorter and wider than knife",
        "visual_cues": ["symmetrical blade", "double edge", "pointed tip", "cross guard"],
        "confusables": ["Knife", "Blade", "Other Sharp Item"],
        "source": "stcray",
    },
    "Blade": {
        "description": "Flat cutting edge, may be standalone or part of a tool",
        "visual_cues": ["thin flat shape", "sharp edge", "high density line"],
        "confusables": ["Knife", "Dagger", "Scissors"],
        "source": "stcray",
    },
    "Lighter": {
        "description": "Small rectangular object with fluid reservoir and ignition mechanism",
        "visual_cues": ["small rectangular", "fluid visible", "metal components inside"],
        "confusables": ["Battery", "Powerbank", "3D Gun"],
        "source": "stcray",
    },
    "Injection": {
        "description": "Thin cylindrical tube with needle, syringe shape",
        "visual_cues": ["thin cylinder", "needle tip", "plunger visible"],
        "confusables": ["Screwdriver", "Nail Cutter", "Other Sharp Item"],
        "source": "stcray",
    },
    "Battery": {
        "description": "Cylindrical or rectangular dense object with uniform internal structure",
        "visual_cues": ["cylindrical or rectangular", "uniform density", "small size"],
        "confusables": ["Powerbank", "Lighter", "Bullet"],
        "source": "stcray",
    },
    "Nail Cutter": {
        "description": "Small folding metal tool with lever mechanism",
        "visual_cues": ["small metal", "folding mechanism", "lever shape"],
        "confusables": ["Scissors", "Injection", "Other Sharp Item"],
        "source": "stcray",
    },
    "Other Sharp Item": {
        "description": "Miscellaneous sharp or pointed metal object",
        "visual_cues": ["pointed end", "metal density", "irregular shape"],
        "confusables": ["Knife", "Dagger", "Blade"],
        "source": "stcray",
    },
    "Powerbank": {
        "description": "Rectangular dense object with battery cells visible inside",
        "visual_cues": ["rectangular", "multiple cells visible", "moderate to high density"],
        "confusables": ["Battery", "Phone", "Lighter"],
        "source": "stcray",
    },
    "Scissors": {
        "description": "X-shaped crossed blades with finger rings",
        "visual_cues": ["X-shape", "crossed blades", "ring handles", "pivot point"],
        "confusables": ["Pliers", "Knife", "Nail Cutter"],
        "source": "stcray",
    },
    "Hammer": {
        "description": "T-shaped tool with heavy head and straight handle",
        "visual_cues": ["T-shape", "dense head", "straight handle"],
        "confusables": ["Wrench", "Pliers", "Powerbank"],
        "source": "stcray",
    },
    "Pliers": {
        "description": "Hinged tool with two handles and gripping jaws",
        "visual_cues": ["hinged mechanism", "two handles", "jaw opening"],
        "confusables": ["Scissors", "Wrench", "Hammer"],
        "source": "stcray",
    },
    "Wrench": {
        "description": "Elongated metal tool with open or closed jaw at one end",
        "visual_cues": ["elongated", "jaw end", "dense metal", "straight or angled"],
        "confusables": ["Pliers", "Hammer", "Screwdriver"],
        "source": "stcray",
    },
    "Screwdriver": {
        "description": "Long thin shaft with wider handle and narrow tip",
        "visual_cues": ["long thin shaft", "wider handle end", "narrow tip"],
        "confusables": ["Wrench", "Injection", "Nail Cutter"],
        "source": "stcray",
    },
    "Handcuffs": {
        "description": "Two connected metal rings with locking mechanism",
        "visual_cues": ["double ring shape", "chain connection", "metal density"],
        "confusables": ["Pliers", "Wrench", "Other Sharp Item"],
        "source": "stcray",
    },
    "Bullet": {
        "description": "Small dense cylindrical projectile, very high metal density",
        "visual_cues": ["small cylindrical", "very dense", "pointed or rounded tip"],
        "confusables": ["Battery", "Lighter", "Nail Cutter"],
        "source": "stcray",
    },
    # === HiXray everyday item categories ===
    "Laptop": {
        "description": "Large rectangular electronic device with hinge, high density",
        "visual_cues": ["large rectangular", "hinge visible", "circuit board density", "screen layer"],
        "confusables": ["Tablet", "Book", "Cutting Board"],
        "source": "hixray",
    },
    "Phone": {
        "description": "Small thin rectangular device with uniform internal components",
        "visual_cues": ["small rectangular", "thin profile", "battery visible", "circuit board"],
        "confusables": ["Powerbank", "Battery", "Calculator"],
        "source": "hixray",
    },
    "Tablet": {
        "description": "Thin rectangular electronic device, larger than phone, no hinge",
        "visual_cues": ["thin rectangular", "no hinge", "uniform density", "larger than phone"],
        "confusables": ["Laptop", "Book", "Phone"],
        "source": "hixray",
    },
    "Cosmetic": {
        "description": "Small container with liquid or cream contents, various shapes",
        "visual_cues": ["small container", "liquid fill", "cap visible", "cylindrical or tube"],
        "confusables": ["Bottle", "Medicine", "Food"],
        "source": "hixray",
    },
    "Water": {
        "description": "Bottle or container with liquid fill visible as uniform density",
        "visual_cues": ["bottle shape", "liquid fill level", "cap/lid", "cylindrical"],
        "confusables": ["Bottle", "Thermos", "Can"],
        "source": "hixray",
    },
    "Portable_Charger_1": {
        "description": "Rectangular portable battery pack with USB ports",
        "visual_cues": ["rectangular", "battery cells inside", "similar to powerbank"],
        "confusables": ["Powerbank", "Battery", "Phone"],
        "source": "hixray",
    },
    "Portable_Charger_2": {
        "description": "Larger portable charger or laptop charger brick",
        "visual_cues": ["rectangular brick", "dense internals", "cable attached"],
        "confusables": ["Powerbank", "Battery", "Laptop"],
        "source": "hixray",
    },
    "Umbrella": {
        "description": "Long thin cylindrical object with folding mechanism",
        "visual_cues": ["long cylindrical", "folding ribs", "pointed tip", "handle"],
        "confusables": ["Wrench", "Stick", "Baton"],
        "source": "hixray",
    },
    # === Common everyday items (supplementary) ===
    "Clothing": {
        "description": "Soft folded fabric with very low X-ray density",
        "visual_cues": ["low density", "folded layers", "soft edges", "large area"],
        "confusables": ["Towel", "Bag", "Fabric"],
        "source": "common",
    },
    "Book": {
        "description": "Rectangular stack of pages with uniform density",
        "visual_cues": ["rectangular", "layered pages visible", "spine edge", "uniform density"],
        "confusables": ["Laptop", "Tablet", "Notebook"],
        "source": "common",
    },
    "Bottle": {
        "description": "Cylindrical container, may contain liquid or be empty",
        "visual_cues": ["cylindrical", "neck and cap", "liquid level if filled"],
        "confusables": ["Water", "Thermos", "Can"],
        "source": "common",
    },
    "Food": {
        "description": "Organic material in packaging, irregular density",
        "visual_cues": ["irregular shape", "packaging visible", "organic density"],
        "confusables": ["Cosmetic", "Medicine", "Toiletry"],
        "source": "common",
    },
    "Shoes": {
        "description": "Footwear with sole visible as denser layer",
        "visual_cues": ["sole outline", "moderate density", "paired items"],
        "confusables": ["Clothing", "Bag", "Boot"],
        "source": "common",
    },
    "Wallet": {
        "description": "Small folded item with cards and metal clasp visible",
        "visual_cues": ["small rectangular", "card shapes inside", "metal clasp"],
        "confusables": ["Phone", "Passport", "Card Holder"],
        "source": "common",
    },
    "Keys": {
        "description": "Small metal objects with distinctive tooth pattern",
        "visual_cues": ["small metal", "tooth pattern", "ring shape", "high density"],
        "confusables": ["Nail Cutter", "USB Drive", "Coin"],
        "source": "common",
    },
    "Charger Cable": {
        "description": "Thin wire with connector ends, coiled or bundled",
        "visual_cues": ["thin wire", "connector ends", "coiled bundle"],
        "confusables": ["Headphones", "Wire", "Rope"],
        "source": "common",
    },
    "Medicine": {
        "description": "Small bottles or blister packs with pills visible",
        "visual_cues": ["small containers", "pill shapes", "blister pack grid"],
        "confusables": ["Cosmetic", "Food", "Battery"],
        "source": "common",
    },
    "Toiletry": {
        "description": "Small tubes or bottles with liquid/cream contents",
        "visual_cues": ["tube shape", "liquid fill", "cap visible"],
        "confusables": ["Cosmetic", "Medicine", "Food"],
        "source": "common",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Build category hints lookup")
    parser.add_argument("--stcray-dir", type=str, default="data/stcray_processed")
    parser.add_argument("--hixray-dir", type=str, default="data/hixray_processed")
    parser.add_argument("--output", type=str, default="data/category_hints.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Build Category Hints Lookup")
    print("=" * 60)

    # Start with curated hints
    hints = dict(CATEGORY_HINTS)

    # Discover additional categories from annotations if available
    stcray_dir = Path(args.stcray_dir)
    for split in ("train", "test"):
        ann_file = stcray_dir / split / "annotations.json"
        if ann_file.exists():
            with open(ann_file) as f:
                annotations = json.load(f)
            for ann in annotations:
                for cat in ann.get("categories", []):
                    if cat not in hints and cat not in ("Non Threat", "Multilabel Threat"):
                        hints[cat] = {
                            "description": f"X-ray item: {cat}",
                            "visual_cues": ["see training data"],
                            "confusables": [],
                            "source": "stcray_auto",
                        }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hints, f, indent=2)

    # Summary
    by_source = {}
    for cat, info in hints.items():
        src = info.get("source", "unknown")
        by_source.setdefault(src, []).append(cat)

    print(f"\n  Total categories: {len(hints)}")
    for src, cats in sorted(by_source.items()):
        print(f"  {src}: {len(cats)} ({', '.join(cats[:5])}{'...' if len(cats) > 5 else ''})")
    print(f"\n  Output: {output_path}")
    print("Category hints built successfully")


if __name__ == "__main__":
    main()
