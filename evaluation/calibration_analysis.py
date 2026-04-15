#!/usr/bin/env python3
"""
Logprob calibration experiment (E_cal).

Validates that VLM token logprobs are a useful uncertainty signal by measuring
entropy-accuracy correlation. Target: ECE < 0.15.

Queries a vLLM endpoint (or loads model locally) with logprobs enabled,
computes per-prediction token entropy, and buckets predictions by entropy
to measure calibration.

Usage:
  python evaluation/calibration_analysis.py \
      --model-path /home/cdsw/models/qwen3vl-2b-pgrav-merged \
      --test-data data/pgrav_vqa/pgrav_val.jsonl \
      --output-dir test_results/calibration \
      --num-samples 500
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_token_entropy(logprobs: list[dict]) -> float:
    """Compute average Shannon entropy from token logprobs."""
    if not logprobs:
        return 0.0

    entropies = []
    for token_lp in logprobs:
        # Each token has top-k logprobs: [{token, logprob}, ...]
        top_logprobs = token_lp.get("top_logprobs", [])
        if not top_logprobs:
            lp = token_lp.get("logprob", 0)
            if lp < 0:
                entropies.append(-lp)
            continue

        # Convert logprobs to probabilities and compute entropy
        probs = [math.exp(lp.get("logprob", -10)) for lp in top_logprobs]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            entropies.append(entropy)

    return float(np.mean(entropies)) if entropies else 0.0


def check_prediction_accuracy(predicted: str, ground_truth: str) -> bool:
    """Check if predicted category matches ground truth (case-insensitive)."""
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    return pred == gt or pred in gt or gt in pred


def query_vlm_with_logprobs(endpoint: str, model: str, image_path: str,
                            prompt: str, max_tokens: int = 256) -> dict:
    """Query vLLM endpoint with logprobs enabled."""
    import base64
    import requests

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    uri = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": uri}},
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
    logprobs_data = result["choices"][0].get("logprobs", {}).get("content", [])

    return {"content": content, "logprobs": logprobs_data}


def main():
    parser = argparse.ArgumentParser(description="Logprob calibration analysis (E_cal)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to merged VLM model or vLLM endpoint")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test JSONL")
    parser.add_argument("--output-dir", type=str, default="test_results/calibration")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--vlm-endpoint", type=str, default=None,
                        help="vLLM chat completions URL (if not provided, derives from model-path)")
    parser.add_argument("--num-entropy-buckets", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Logprob Calibration Analysis (E_cal)")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Test data: {args.test_data}")
    print(f"  Samples: {args.num_samples}")
    print()

    # Load test data
    samples = []
    with open(args.test_data) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[:args.num_samples]
    print(f"  Loaded {len(samples)} test samples")

    # Determine endpoint
    endpoint = args.vlm_endpoint
    model_name = args.model_path
    if not endpoint:
        print("  Note: --vlm-endpoint not provided.")
        print("  Please provide a vLLM endpoint URL for logprob extraction.")
        print("  Example: --vlm-endpoint http://localhost:8000/v1/chat/completions")
        return

    # Run inference with logprobs
    print("\nRunning inference with logprobs...")
    results = []
    for i, sample in enumerate(samples):
        image_path = sample.get("image_path", "")
        if not Path(image_path).exists():
            continue

        prompt = sample.get("question", "")
        gt_answer = sample.get("answer", "")

        try:
            gt_parsed = json.loads(gt_answer) if isinstance(gt_answer, str) else gt_answer
            gt_category = gt_parsed.get("category", gt_parsed.get("objects", [{}])[0].get("category", ""))
        except (json.JSONDecodeError, IndexError, KeyError):
            gt_category = str(gt_answer)

        try:
            resp = query_vlm_with_logprobs(endpoint, model_name, image_path, prompt)
            content = resp["content"]
            logprobs = resp["logprobs"]

            # Parse predicted category
            try:
                pred_parsed = json.loads(content)
                pred_category = pred_parsed.get("category", pred_parsed.get("objects", [{}])[0].get("category", ""))
            except (json.JSONDecodeError, IndexError, KeyError):
                pred_category = content.strip()

            entropy = compute_token_entropy(logprobs)
            correct = check_prediction_accuracy(pred_category, gt_category)

            results.append({
                "sample_idx": i,
                "entropy": entropy,
                "correct": correct,
                "predicted": pred_category,
                "ground_truth": gt_category,
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(samples)}] Avg entropy: {np.mean([r['entropy'] for r in results]):.3f}, "
                      f"Accuracy: {sum(r['correct'] for r in results) / len(results):.3f}")

        except Exception as e:
            print(f"  Sample {i}: Error - {e}")
            continue

    if not results:
        print("Error: No successful predictions. Check endpoint and data paths.")
        return

    # Compute calibration metrics
    print(f"\nComputing calibration metrics from {len(results)} predictions...")
    entropies = [r["entropy"] for r in results]
    corrects = [r["correct"] for r in results]

    # Bucket by entropy
    min_e, max_e = min(entropies), max(entropies)
    bucket_width = (max_e - min_e) / args.num_entropy_buckets if max_e > min_e else 1.0
    buckets = defaultdict(lambda: {"correct": 0, "total": 0, "sum_entropy": 0})

    for e, c in zip(entropies, corrects):
        bucket_idx = min(int((e - min_e) / bucket_width), args.num_entropy_buckets - 1)
        bucket_key = f"{min_e + bucket_idx * bucket_width:.2f}-{min_e + (bucket_idx + 1) * bucket_width:.2f}"
        buckets[bucket_key]["total"] += 1
        buckets[bucket_key]["correct"] += int(c)
        buckets[bucket_key]["sum_entropy"] += e

    # Compute ECE
    ece = 0.0
    total_samples = len(results)
    bucket_stats = {}
    for key, b in sorted(buckets.items()):
        accuracy = b["correct"] / b["total"] if b["total"] > 0 else 0
        avg_entropy = b["sum_entropy"] / b["total"] if b["total"] > 0 else 0
        confidence = 1.0 - avg_entropy / (max_e + 1e-10)  # entropy → confidence
        ece += abs(accuracy - confidence) * b["total"] / total_samples
        bucket_stats[key] = {
            "accuracy": round(accuracy, 4),
            "avg_entropy": round(avg_entropy, 4),
            "count": b["total"],
        }

    # Correlation
    correlation = float(np.corrcoef(entropies, [float(c) for c in corrects])[0, 1])

    # Find recommended threshold
    # Threshold where accuracy >= 0.9
    sorted_results = sorted(results, key=lambda r: r["entropy"])
    threshold = max_e
    for i in range(len(sorted_results)):
        subset = sorted_results[:i+1]
        acc = sum(r["correct"] for r in subset) / len(subset)
        if acc < 0.9 and i > 10:
            threshold = sorted_results[i]["entropy"]
            break

    overall_accuracy = sum(corrects) / len(corrects)

    # Build output
    metrics = {
        "experiment": "E_cal",
        "total_predictions": len(results),
        "overall_accuracy": round(overall_accuracy, 4),
        "entropy_range": [round(min_e, 4), round(max_e, 4)],
        "entropy_buckets": bucket_stats,
        "ece": round(ece, 4),
        "entropy_accuracy_correlation": round(correlation, 4),
        "threshold_recommendation": round(threshold, 4),
        "pass": ece < 0.15,
    }

    # Save results
    with open(output_dir / "calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "calibration_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("CALIBRATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total predictions:  {len(results)}")
    print(f"  Overall accuracy:   {overall_accuracy:.4f}")
    print(f"  ECE:                {ece:.4f} {'PASS' if ece < 0.15 else 'FAIL'} (target < 0.15)")
    print(f"  Entropy-accuracy r: {correlation:.4f} (should be negative)")
    print(f"  Recommended thresh: {threshold:.4f}")
    print(f"\n  Entropy buckets:")
    for key, b in sorted(bucket_stats.items()):
        bar = "#" * int(b["accuracy"] * 20)
        print(f"    {key}: acc={b['accuracy']:.2f} n={b['count']:4d} {bar}")

    print(f"\n  Results saved to: {output_dir}")

    # Try to generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter: entropy vs accuracy
        ax1.scatter(entropies, [float(c) for c in corrects], alpha=0.1, s=5)
        ax1.set_xlabel("Token Entropy")
        ax1.set_ylabel("Correct (0/1)")
        ax1.set_title("Entropy vs Correctness")
        ax1.axvline(x=threshold, color="r", linestyle="--", label=f"Threshold={threshold:.2f}")
        ax1.legend()

        # Calibration curve
        bucket_accs = [b["accuracy"] for b in sorted(bucket_stats.values(), key=lambda x: x["avg_entropy"])]
        bucket_ents = [b["avg_entropy"] for b in sorted(bucket_stats.values(), key=lambda x: x["avg_entropy"])]
        ax2.plot(bucket_ents, bucket_accs, "o-", label="Observed")
        ax2.plot([0, max_e], [1, 0], "k--", alpha=0.3, label="Perfect calibration")
        ax2.set_xlabel("Average Entropy")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Calibration Curve (ECE={ece:.3f})")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "calibration_plot.png", dpi=150)
        print(f"  Plot saved: {output_dir / 'calibration_plot.png'}")
    except ImportError:
        print("  (matplotlib not available, skipping plot)")


if __name__ == "__main__":
    main()
