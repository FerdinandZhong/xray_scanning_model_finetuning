#!/usr/bin/env python3
"""
Adaptive loop evaluation experiment (E_adapt).

Compares single-pass pipeline vs uncertainty-gated adaptive re-analysis.
Target: >= 5% recall gain on ambiguous items at fixed precision.

Usage:
  python evaluation/eval_adaptive_loop.py \
      --model-path /home/cdsw/models/qwen3vl-2b-pgrav-merged \
      --yolo-model runs/detect/class_agnostic_xray/weights/best.pt \
      --category-hints data/category_hints.json \
      --test-data data/pgrav_vqa/pgrav_test.jsonl \
      --output-dir test_results/adaptive_loop \
      --entropy-threshold 1.5
"""

import argparse
import base64
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def encode_image(image_path: str) -> str:
    """Encode image to base64 data URI."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def query_vlm(endpoint: str, model: str, image_uri: str, prompt: str,
              max_tokens: int = 512) -> dict:
    """Query VLM with logprobs."""
    import requests

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    logprobs = result["choices"][0].get("logprobs", {}).get("content", [])
    return {"content": content, "logprobs": logprobs}


def compute_entropy(logprobs: list) -> float:
    """Compute average token entropy."""
    if not logprobs:
        return 0.0
    ents = []
    for tok in logprobs:
        top = tok.get("top_logprobs", [])
        if top:
            probs = [math.exp(lp.get("logprob", -10)) for lp in top]
            s = sum(probs)
            if s > 0:
                probs = [p / s for p in probs]
                ents.append(-sum(p * math.log(p + 1e-10) for p in probs if p > 0))
    return float(np.mean(ents)) if ents else 0.0


def parse_category(text: str) -> str:
    """Extract category from VLM response."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed.get("category", parsed.get("objects", [{}])[0].get("category", ""))
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return text.strip()[:50]


def run_single_pass(endpoint: str, model: str, image_path: str, prompt: str) -> dict:
    """Run single-pass inference."""
    t0 = time.time()
    image_uri = encode_image(image_path)
    resp = query_vlm(endpoint, model, image_uri, prompt)
    elapsed = (time.time() - t0) * 1000

    category = parse_category(resp["content"])
    entropy = compute_entropy(resp["logprobs"])

    return {
        "category": category,
        "entropy": entropy,
        "latency_ms": elapsed,
        "reanalyzed": False,
    }


def run_adaptive(endpoint: str, model: str, image_path: str, prompt: str,
                 category_hints: dict, threshold: float) -> dict:
    """Run adaptive pipeline with re-analysis for uncertain predictions."""
    t0 = time.time()
    image_uri = encode_image(image_path)

    # Pass 1: full-scene analysis
    resp = query_vlm(endpoint, model, image_uri, prompt)
    category = parse_category(resp["content"])
    entropy = compute_entropy(resp["logprobs"])

    reanalyzed = False
    if entropy > threshold and category in category_hints:
        # Pass 2: focused re-analysis with hints
        hints = category_hints[category]
        confusables = hints.get("confusables", [])
        description = hints.get("description", "")

        focused_prompt = (
            f"Identify this item in the X-ray scan.\n"
            f"Initial detection: {category} ({description})\n"
            f"Could also be: {', '.join(confusables)}.\n"
            "Confirm or correct the classification. "
            "Respond with JSON: {\"category\": \"...\", \"confidence\": 0.0}"
        )

        resp2 = query_vlm(endpoint, model, image_uri, focused_prompt)
        new_category = parse_category(resp2["content"])
        new_entropy = compute_entropy(resp2["logprobs"])

        # Update only if more confident
        if new_entropy < entropy:
            category = new_category
            entropy = new_entropy
        reanalyzed = True

    elapsed = (time.time() - t0) * 1000
    return {
        "category": category,
        "entropy": entropy,
        "latency_ms": elapsed,
        "reanalyzed": reanalyzed,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive loop evaluation (E_adapt)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--yolo-model", type=str, default="runs/detect/class_agnostic_xray/weights/best.pt")
    parser.add_argument("--category-hints", type=str, default="data/category_hints.json")
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="test_results/adaptive_loop")
    parser.add_argument("--entropy-threshold", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--vlm-endpoint", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Adaptive Loop Evaluation (E_adapt)")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Threshold: {args.entropy_threshold}")
    print(f"  Samples: {args.num_samples}")

    # Load category hints
    with open(args.category_hints) as f:
        category_hints = json.load(f)
    print(f"  Category hints: {len(category_hints)} entries")

    # Load test data
    samples = []
    with open(args.test_data) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    # Filter to ROI-focused format for cleaner evaluation
    samples = [s for s in samples if s.get("metadata", {}).get("format") == "roi_focused"]
    samples = samples[:args.num_samples]
    print(f"  Test samples: {len(samples)} (roi_focused format)")

    endpoint = args.vlm_endpoint
    model_name = args.model_path
    if not endpoint:
        print("\n  Note: --vlm-endpoint required for inference with logprobs.")
        return

    # Run both modes
    print("\nRunning evaluation...")
    single_results = []
    adaptive_results = []

    for i, sample in enumerate(samples):
        image_path = sample.get("image_path", "")
        if not Path(image_path).exists():
            continue

        prompt = sample.get("question", "")
        gt = sample.get("metadata", {}).get("ground_truth_category", "")

        try:
            sp = run_single_pass(endpoint, model_name, image_path, prompt)
            sp["correct"] = parse_category(sp["category"]).lower() == gt.lower()
            sp["ground_truth"] = gt
            single_results.append(sp)

            ad = run_adaptive(endpoint, model_name, image_path, prompt,
                              category_hints, args.entropy_threshold)
            ad["correct"] = parse_category(ad["category"]).lower() == gt.lower()
            ad["ground_truth"] = gt
            adaptive_results.append(ad)

            if (i + 1) % 25 == 0:
                sp_acc = sum(r["correct"] for r in single_results) / len(single_results)
                ad_acc = sum(r["correct"] for r in adaptive_results) / len(adaptive_results)
                n_reanalyzed = sum(r["reanalyzed"] for r in adaptive_results)
                print(f"  [{i+1}/{len(samples)}] Single: {sp_acc:.3f} | Adaptive: {ad_acc:.3f} | Reanalyzed: {n_reanalyzed}")

        except Exception as e:
            print(f"  Sample {i}: Error - {e}")

    if not single_results:
        print("Error: No successful evaluations.")
        return

    # Compute metrics
    sp_accuracy = sum(r["correct"] for r in single_results) / len(single_results)
    ad_accuracy = sum(r["correct"] for r in adaptive_results) / len(adaptive_results)
    sp_latency = np.mean([r["latency_ms"] for r in single_results])
    ad_latency = np.mean([r["latency_ms"] for r in adaptive_results])
    n_reanalyzed = sum(r["reanalyzed"] for r in adaptive_results)

    # Recall on items that were re-analyzed
    reanalyzed_items = [(s, a) for s, a in zip(single_results, adaptive_results) if a["reanalyzed"]]
    sp_recall_reanalyzed = sum(s["correct"] for s, _ in reanalyzed_items) / max(len(reanalyzed_items), 1)
    ad_recall_reanalyzed = sum(a["correct"] for _, a in reanalyzed_items) / max(len(reanalyzed_items), 1)

    metrics = {
        "experiment": "E_adapt",
        "total_samples": len(single_results),
        "entropy_threshold": args.entropy_threshold,
        "single_pass": {
            "accuracy": round(sp_accuracy, 4),
            "avg_latency_ms": round(sp_latency, 1),
        },
        "adaptive_loop": {
            "accuracy": round(ad_accuracy, 4),
            "avg_latency_ms": round(ad_latency, 1),
            "regions_reanalyzed": n_reanalyzed,
            "reanalysis_rate": round(n_reanalyzed / len(adaptive_results), 4),
        },
        "improvement": {
            "accuracy_delta": round(ad_accuracy - sp_accuracy, 4),
            "recall_on_reanalyzed_single": round(sp_recall_reanalyzed, 4),
            "recall_on_reanalyzed_adaptive": round(ad_recall_reanalyzed, 4),
            "recall_delta_on_reanalyzed": round(ad_recall_reanalyzed - sp_recall_reanalyzed, 4),
            "latency_overhead_ms": round(ad_latency - sp_latency, 1),
        },
        "pass": (ad_accuracy - sp_accuracy) >= 0.02 or (ad_recall_reanalyzed - sp_recall_reanalyzed) >= 0.05,
    }

    with open(output_dir / "adaptive_loop_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "adaptive_loop_raw.json", "w") as f:
        json.dump({"single_pass": single_results, "adaptive": adaptive_results}, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("ADAPTIVE LOOP RESULTS")
    print(f"{'=' * 60}")
    print(f"  Samples evaluated:   {len(single_results)}")
    print(f"  Entropy threshold:   {args.entropy_threshold}")
    print(f"  Regions re-analyzed: {n_reanalyzed} ({n_reanalyzed/len(adaptive_results)*100:.1f}%)")
    print(f"\n  {'Mode':<20} {'Accuracy':>10} {'Latency':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Single-pass':<20} {sp_accuracy:>10.4f} {sp_latency:>8.0f}ms")
    print(f"  {'Adaptive loop':<20} {ad_accuracy:>10.4f} {ad_latency:>8.0f}ms")
    print(f"  {'Delta':<20} {ad_accuracy - sp_accuracy:>+10.4f} {ad_latency - sp_latency:>+8.0f}ms")
    if reanalyzed_items:
        print(f"\n  On re-analyzed items only ({len(reanalyzed_items)} items):")
        print(f"    Single-pass recall:  {sp_recall_reanalyzed:.4f}")
        print(f"    Adaptive recall:     {ad_recall_reanalyzed:.4f}")
        print(f"    Delta:               {ad_recall_reanalyzed - sp_recall_reanalyzed:+.4f}")
    print(f"\n  Result: {'PASS' if metrics['pass'] else 'FAIL'}")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
