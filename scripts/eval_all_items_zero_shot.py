#!/usr/bin/env python3
"""
Zero-shot evaluation: Can Qwen3-VL-2B identify ALL items in X-ray images?

Sends X-ray images to a vLLM OpenAI-compatible endpoint with an "all items"
prompt and evaluates the quality of responses.

Usage:
  python scripts/eval_all_items_zero_shot.py \
      --endpoint https://ray-cluster-head.ml-1841266f-15a.qzhong-1.a465-9q4k.cloudera.site/qwen3-vl/v1/chat/completions \
      --images-dir data/stcray_raw/STCray_TestSet/Images \
      --n 20 --output test_results/all_items_zero_shot/
"""

import argparse
import base64
import json
import os
import random
import time
from pathlib import Path

import requests


ALL_ITEMS_PROMPT = """\
You are an expert X-ray baggage scanner. Identify every distinct item visible in this scan.

Include ALL items: electronics, clothing, bottles, tools, weapons, food, toiletries, bags, etc.

Use short descriptions (1-3 words). Respond as JSON with "items" array and "total_count".
"""


def encode_image(image_path: str) -> str:
    """Encode image to base64 data URI."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext.lstrip("."), "image/jpeg"
    )
    return f"data:{mime};base64,{data}"


ALL_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["category", "description"],
            },
        },
        "total_count": {"type": "integer"},
    },
    "required": ["items", "total_count"],
}


def call_endpoint(endpoint: str, model: str, image_path: str, prompt: str, temperature: float = 0.0) -> dict:
    """Call the vLLM OpenAI-compatible chat completions endpoint with guided JSON."""
    image_uri = encode_image(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "all_items_detection",
                "schema": ALL_ITEMS_SCHEMA,
            },
        },
    }

    start = time.time()
    resp = requests.post(endpoint, json=payload, timeout=120)
    elapsed_ms = (time.time() - start) * 1000
    resp.raise_for_status()
    result = resp.json()

    content = result["choices"][0]["message"]["content"]
    return {"content": content, "elapsed_ms": elapsed_ms}


def parse_json_response(text: str) -> dict | None:
    """Try to extract JSON from the model response, including truncated JSON."""
    import re

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ```)
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try to find the outermost JSON object
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Handle truncated JSON: try to repair by closing open structures
    # Find all complete item objects and wrap them
    item_pattern = r'\{\s*"category"\s*:\s*"[^"]*"\s*,\s*"description"\s*:\s*"[^"]*"\s*\}'
    items = re.findall(item_pattern, stripped)
    if items:
        repaired = {"items": [json.loads(it) for it in items], "total_count": len(items)}
        return repaired

    return None


def collect_images(images_dir: str, n: int, seed: int) -> list[str]:
    """Collect n random images from the directory (across class subdirs)."""
    all_images = []
    root = Path(images_dir)

    # Handle both flat and nested (class subdirectory) layouts
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_images.extend(root.glob(ext))
        all_images.extend(root.glob(f"*/{ext}"))

    all_images = [str(p) for p in all_images]
    random.seed(seed)
    random.shuffle(all_images)

    # Pick diverse samples: 1 from each class subdir if possible
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    if class_dirs and len(class_dirs) >= n:
        selected = []
        random.shuffle(class_dirs)
        for d in class_dirs[:n]:
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if imgs:
                selected.append(str(random.choice(imgs)))
        return selected[:n]

    return all_images[:n]


def main():
    parser = argparse.ArgumentParser(description="Zero-shot all-items evaluation")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="vLLM chat completions endpoint URL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from /v1/models if not set)",
    )
    parser.add_argument("--images-dir", required=True, help="Directory with X-ray images")
    parser.add_argument("--n", type=int, default=20, help="Number of images to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="test_results/all_items_zero_shot/", help="Output dir")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect model name
    model = args.model
    if not model:
        base_url = args.endpoint.rsplit("/chat/completions", 1)[0]
        models_resp = requests.get(f"{base_url}/models", timeout=10).json()
        model = models_resp["data"][0]["id"]
        print(f"Auto-detected model: {model}")

    # Collect images
    images = collect_images(args.images_dir, args.n, args.seed)
    print(f"Selected {len(images)} images from {args.images_dir}")
    print(f"Output: {output_dir.resolve()}\n")

    results = []
    stats = {
        "total": len(images),
        "valid_json": 0,
        "has_items": 0,
        "categories_seen": set(),
        "avg_items_per_image": 0,
        "avg_latency_ms": 0,
        "errors": 0,
    }

    for i, img_path in enumerate(images):
        class_dir = Path(img_path).parent.name
        print(f"[{i+1:2d}/{len(images)}] {class_dir}/{Path(img_path).name}...", end=" ", flush=True)

        try:
            resp = call_endpoint(args.endpoint, model, img_path, ALL_ITEMS_PROMPT, args.temperature)
            raw = resp["content"]
            elapsed = resp["elapsed_ms"]

            parsed = parse_json_response(raw)
            is_valid_json = parsed is not None
            items = parsed.get("items", []) if parsed else []
            has_items = len(items) > 0
            categories = [it.get("category", "unknown") for it in items]
            summary = parsed.get("summary", "") if parsed else ""

            if is_valid_json:
                stats["valid_json"] += 1
            if has_items:
                stats["has_items"] += 1
            stats["categories_seen"].update(categories)
            stats["avg_items_per_image"] += len(items)
            stats["avg_latency_ms"] += elapsed

            status = "OK" if is_valid_json and has_items else "WEAK" if is_valid_json else "FAIL"
            print(f"{status} | {len(items)} items | {elapsed:.0f}ms | {', '.join(categories[:5])}")

            result = {
                "image_path": img_path,
                "class_dir": class_dir,
                "raw_output": raw,
                "parsed": parsed,
                "valid_json": is_valid_json,
                "num_items": len(items),
                "categories": categories,
                "summary": summary,
                "elapsed_ms": elapsed,
            }
            results.append(result)

        except Exception as e:
            print(f"ERROR: {e}")
            stats["errors"] += 1
            results.append({
                "image_path": img_path,
                "class_dir": class_dir,
                "error": str(e),
                "valid_json": False,
                "num_items": 0,
                "categories": [],
            })

    # Compute final stats
    n_success = stats["total"] - stats["errors"]
    stats["avg_items_per_image"] = stats["avg_items_per_image"] / max(n_success, 1)
    stats["avg_latency_ms"] = stats["avg_latency_ms"] / max(n_success, 1)
    stats["categories_seen"] = sorted(stats["categories_seen"])
    stats["num_unique_categories"] = len(stats["categories_seen"])

    # Print report
    print("\n" + "=" * 70)
    print("ZERO-SHOT ALL-ITEMS EVALUATION REPORT")
    print("=" * 70)
    print(f"  Images tested:          {stats['total']}")
    print(f"  Valid JSON responses:    {stats['valid_json']}/{stats['total']} ({stats['valid_json']/stats['total']:.0%})")
    print(f"  Responses with items:    {stats['has_items']}/{stats['total']} ({stats['has_items']/stats['total']:.0%})")
    print(f"  Avg items per image:     {stats['avg_items_per_image']:.1f}")
    print(f"  Avg latency:             {stats['avg_latency_ms']:.0f}ms")
    print(f"  Unique categories seen:  {stats['num_unique_categories']}")
    print(f"  Categories:              {', '.join(stats['categories_seen'])}")
    print(f"  Errors:                  {stats['errors']}")

    # Pass/fail criteria
    print("\n" + "-" * 70)
    print("PASS/FAIL CRITERIA")
    print("-" * 70)
    json_rate = stats["valid_json"] / stats["total"]
    items_rate = stats["has_items"] / stats["total"]
    diverse = stats["num_unique_categories"] >= 3

    checks = [
        (json_rate >= 0.80, f"Valid JSON >= 80%: {json_rate:.0%}"),
        (items_rate >= 0.80, f"Responses with items >= 80%: {items_rate:.0%}"),
        (stats["avg_items_per_image"] >= 1.0, f"Avg items/image >= 1.0: {stats['avg_items_per_image']:.1f}"),
        (diverse, f"Diverse categories (>= 3 unique): {stats['num_unique_categories']}"),
    ]

    all_pass = True
    for passed, desc in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {desc}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: ALL CHECKS PASSED -- zero-shot all-items detection is viable")
    else:
        print("RESULT: SOME CHECKS FAILED -- may need fine-tuning or prompt iteration")
    print("=" * 70)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Save markdown report
    with open(output_dir / "report.md", "w") as f:
        f.write("# Zero-Shot All-Items Evaluation Report\n\n")
        f.write(f"- **Model**: {model}\n")
        f.write(f"- **Images tested**: {stats['total']}\n")
        f.write(f"- **Valid JSON**: {stats['valid_json']}/{stats['total']} ({json_rate:.0%})\n")
        f.write(f"- **With items**: {stats['has_items']}/{stats['total']} ({items_rate:.0%})\n")
        f.write(f"- **Avg items/image**: {stats['avg_items_per_image']:.1f}\n")
        f.write(f"- **Avg latency**: {stats['avg_latency_ms']:.0f}ms\n")
        f.write(f"- **Unique categories**: {stats['num_unique_categories']}\n")
        f.write(f"- **Result**: {'PASSED' if all_pass else 'FAILED'}\n\n")
        f.write("## Categories Detected\n\n")
        for cat in stats["categories_seen"]:
            f.write(f"- {cat}\n")
        f.write("\n## Per-Image Results\n\n")
        f.write("| # | Class | Items | Categories | JSON |\n")
        f.write("|---|-------|-------|------------|------|\n")
        for i, r in enumerate(results):
            cats = ", ".join(r.get("categories", [])[:4])
            valid = "Y" if r.get("valid_json") else "N"
            f.write(f"| {i+1} | {r.get('class_dir','')} | {r.get('num_items',0)} | {cats} | {valid} |\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
