#!/usr/bin/env python3
"""
End-to-end declaration matching evaluation (E7).

Runs the full XScan-Agent pipeline on test images with synthetic declarations
and measures customs verification accuracy.

Usage:
  python evaluation/eval_declaration_matching.py \
      --model-path /home/cdsw/models/qwen3vl-2b-pgrav-merged \
      --yolo-model runs/detect/class_agnostic_xray/weights/best.pt \
      --category-hints data/category_hints.json \
      --declaration-data data/declaration_benchmark.json \
      --output-dir test_results/declaration_matching
"""

import argparse
import base64
import json
import random
import time
from collections import defaultdict
from pathlib import Path

# Customs category mapping (from inference/hybrid_pipeline.py)
CUSTOMS_CATEGORIES = {
    "electronics": [
        "laptop", "computer", "phone", "mobile", "tablet", "ipad", "charger",
        "powerbank", "power bank", "battery", "camera", "headphone", "earphone",
        "cable", "wire", "circuit", "device", "electronic", "gadget", "speaker",
        "portable_charger_1", "portable_charger_2",
    ],
    "clothing": [
        "clothing", "clothes", "shirt", "pants", "dress", "jacket", "coat",
        "shoes", "shoe", "boot", "sock", "hat", "belt", "fabric", "textile",
    ],
    "liquids": [
        "bottle", "water", "liquid", "drink", "spray", "aerosol", "perfume",
        "container", "flask", "thermos",
    ],
    "food": [
        "food", "snack", "fruit", "vegetable", "candy", "chocolate", "bread",
    ],
    "toiletries": [
        "cosmetic", "makeup", "shampoo", "soap", "toothbrush", "toothpaste",
        "razor", "deodorant", "lotion", "cream", "toiletry",
    ],
    "metal_tools": [
        "tool", "wrench", "pliers", "screwdriver", "hammer", "spanner",
    ],
    "weapons": [
        "gun", "firearm", "pistol", "rifle", "handgun", "weapon", "explosive",
        "ammunition", "bullet", "3d gun",
    ],
    "sharp_objects": [
        "knife", "blade", "dagger", "scissors", "cutter", "needle",
        "injection", "nail cutter", "other sharp item",
    ],
    "flammable": ["lighter", "match", "flammable"],
    "documents": ["book", "document", "paper", "passport", "magazine", "notebook"],
    "medication": ["medicine", "medication", "pill", "drug"],
    "other": ["umbrella", "toy", "key", "wallet", "jewelry", "watch", "glasses", "handcuffs"],
}

THREAT_SEVERITY = {
    "weapons": "critical",
    "sharp_objects": "high",
    "flammable": "high",
    "explosives": "critical",
    "metal_tools": "medium",
    "medication": "low",
}


def map_item_to_customs(item_name: str) -> str:
    """Map a detected item to its customs category."""
    item_lower = item_name.lower()
    for customs_cat, keywords in CUSTOMS_CATEGORIES.items():
        if any(kw in item_lower for kw in keywords):
            return customs_cat
    return "other"


def compare_declaration(detected_categories: dict, declared_categories: dict) -> dict:
    """Compare detected vs declared categories."""
    detected_set = set(detected_categories.keys())
    declared_set = {k for k, v in declared_categories.items() if v}

    matched = detected_set & declared_set
    undeclared = detected_set - declared_set - {"other"}
    undetected = declared_set - detected_set

    mismatches = []
    for cat in undeclared:
        severity = THREAT_SEVERITY.get(cat, "medium")
        mismatches.append({
            "category": cat,
            "type": "UNDECLARED",
            "severity": severity,
            "items": detected_categories.get(cat, []),
        })

    return {
        "match": len(undeclared) == 0,
        "matched": sorted(matched),
        "undeclared": sorted(undeclared),
        "undetected": sorted(undetected),
        "mismatches": mismatches,
    }


def generate_declaration_benchmark(test_samples: list, output_path: Path, seed: int = 42):
    """Generate synthetic declaration benchmark from test samples."""
    random.seed(seed)
    benchmark = []

    for i, sample in enumerate(test_samples):
        gt_answer = sample.get("answer", "")
        try:
            parsed = json.loads(gt_answer) if isinstance(gt_answer, str) else gt_answer
        except json.JSONDecodeError:
            continue

        # Extract ground truth categories
        gt_items = []
        if isinstance(parsed, dict):
            objects = parsed.get("objects", [])
            if not objects and "category" in parsed:
                objects = [parsed]
            for obj in objects:
                cat = obj.get("category", "")
                if cat:
                    gt_items.append(cat)

        if not gt_items:
            continue

        # Map to customs categories
        gt_customs = {}
        for item in gt_items:
            customs_cat = map_item_to_customs(item)
            gt_customs.setdefault(customs_cat, []).append(item)

        # Generate declaration scenarios
        scenario = random.choice(["honest", "honest", "partial_hide", "fraud"])

        if scenario == "honest":
            # Declare everything
            declaration = {cat: True for cat in gt_customs}
        elif scenario == "partial_hide":
            # Hide some categories
            declaration = {}
            for cat in gt_customs:
                declaration[cat] = random.random() > 0.4
        else:  # fraud
            # Declare almost nothing
            declaration = {cat: random.random() > 0.8 for cat in gt_customs}

        benchmark.append({
            "sample_idx": i,
            "image_path": sample.get("image_path", ""),
            "question": sample.get("question", ""),
            "ground_truth_items": gt_items,
            "ground_truth_customs": {k: v for k, v in gt_customs.items()},
            "declaration": declaration,
            "scenario": scenario,
        })

    # Save benchmark
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"  Generated {len(benchmark)} declaration scenarios")
    scenarios = defaultdict(int)
    for b in benchmark:
        scenarios[b["scenario"]] += 1
    for s, n in sorted(scenarios.items()):
        print(f"    {s}: {n}")

    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Declaration matching evaluation (E7)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--yolo-model", type=str, default="runs/detect/class_agnostic_xray/weights/best.pt")
    parser.add_argument("--category-hints", type=str, default="data/category_hints.json")
    parser.add_argument("--declaration-data", type=str, default="data/declaration_benchmark.json")
    parser.add_argument("--test-data", type=str, default="data/pgrav_vqa/pgrav_test.jsonl",
                        help="Source test data for generating benchmark if declaration-data doesn't exist")
    parser.add_argument("--output-dir", type=str, default="test_results/declaration_matching")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--vlm-endpoint", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Declaration Matching Evaluation (E7)")
    print("=" * 60)

    # Load or generate benchmark
    benchmark_path = Path(args.declaration_data)
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            benchmark = json.load(f)
        print(f"  Loaded benchmark: {len(benchmark)} scenarios")
    else:
        print("  Benchmark not found, generating from test data...")
        test_samples = []
        with open(args.test_data) as f:
            for line in f:
                if line.strip():
                    test_samples.append(json.loads(line))
        # Use full-scene format for declaration matching
        test_samples = [s for s in test_samples
                        if s.get("metadata", {}).get("format") == "full_scene_proposals"]
        benchmark = generate_declaration_benchmark(test_samples, benchmark_path)

    if args.num_samples > 0:
        benchmark = benchmark[:args.num_samples]

    endpoint = args.vlm_endpoint
    model_name = args.model_path
    if not endpoint:
        print("\n  Note: --vlm-endpoint required for VLM inference.")
        print("  Running in ground-truth-only mode (evaluating declaration logic, not VLM accuracy).")
        # Use ground truth items directly
        use_ground_truth = True
    else:
        use_ground_truth = False

    # Run evaluation
    print(f"\nEvaluating {len(benchmark)} scenarios...")
    results = []
    confusion = {"true_match": 0, "true_mismatch": 0, "false_positive": 0, "false_negative": 0}
    by_severity = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})

    for i, entry in enumerate(benchmark):
        if use_ground_truth:
            detected_customs = entry["ground_truth_customs"]
        else:
            # Run VLM inference
            image_path = entry.get("image_path", "")
            if not Path(image_path).exists():
                continue
            prompt = entry.get("question", "")
            try:
                import requests
                image_uri = f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}"
                resp = requests.post(endpoint, json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": prompt},
                    ]}],
                    "max_tokens": 512, "temperature": 0.0,
                }, timeout=120)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                detected_items = []
                for obj in parsed.get("objects", []):
                    detected_items.append(obj.get("category", ""))
                detected_customs = {}
                for item in detected_items:
                    cc = map_item_to_customs(item)
                    detected_customs.setdefault(cc, []).append(item)
            except Exception as e:
                print(f"  Sample {i}: Error - {e}")
                continue

        declaration = entry["declaration"]
        comparison = compare_declaration(detected_customs, declaration)

        gt_has_undeclared = any(
            cat in entry["ground_truth_customs"]
            and not entry["declaration"].get(cat, False)
            for cat in entry["ground_truth_customs"]
            if cat != "other"
        )

        if comparison["match"] and not gt_has_undeclared:
            confusion["true_match"] += 1
        elif not comparison["match"] and gt_has_undeclared:
            confusion["true_mismatch"] += 1
        elif not comparison["match"] and not gt_has_undeclared:
            confusion["false_positive"] += 1
        else:
            confusion["false_negative"] += 1

        # Track by severity
        for mismatch in comparison["mismatches"]:
            sev = mismatch["severity"]
            by_severity[sev]["total"] += 1
            by_severity[sev]["tp"] += 1

        results.append({
            "sample_idx": entry["sample_idx"],
            "scenario": entry["scenario"],
            "comparison": comparison,
            "ground_truth_has_undeclared": gt_has_undeclared,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(benchmark)}] processed")

    # Compute metrics
    total = sum(confusion.values())
    precision = confusion["true_mismatch"] / max(confusion["true_mismatch"] + confusion["false_positive"], 1)
    recall = confusion["true_mismatch"] / max(confusion["true_mismatch"] + confusion["false_negative"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (confusion["true_match"] + confusion["true_mismatch"]) / max(total, 1)

    metrics = {
        "experiment": "E7_declaration_matching",
        "total_scenarios": total,
        "mode": "ground_truth" if use_ground_truth else "vlm_inference",
        "accuracy": round(accuracy, 4),
        "undeclared_precision": round(precision, 4),
        "undeclared_recall": round(recall, 4),
        "undeclared_f1": round(f1, 4),
        "confusion_matrix": confusion,
        "by_severity": {k: dict(v) for k, v in by_severity.items()},
    }

    with open(output_dir / "declaration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "declaration_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("DECLARATION MATCHING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Mode:       {'Ground truth items' if use_ground_truth else 'VLM inference'}")
    print(f"  Scenarios:  {total}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    True match:      {confusion['true_match']}")
    print(f"    True mismatch:   {confusion['true_mismatch']}")
    print(f"    False positive:  {confusion['false_positive']}")
    print(f"    False negative:  {confusion['false_negative']}")
    if by_severity:
        print(f"\n  By severity:")
        for sev in ("critical", "high", "medium", "low"):
            if sev in by_severity:
                s = by_severity[sev]
                print(f"    {sev}: {s['total']} detections")
    print(f"\n  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
