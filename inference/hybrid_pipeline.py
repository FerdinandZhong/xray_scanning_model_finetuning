#!/usr/bin/env python3
"""
XScan-Agent: YOLO-VLM Hybrid Pipeline for X-Ray Baggage Understanding.

Architecture:
  1. YOLO proposes object regions (fast, high recall)
  2. VLM classifies each ROI with open-vocabulary categories
  3. Full-image VLM pass catches items YOLO missed
  4. Results are fused and deduplicated
  5. Category mapping layer maps to customs declaration categories
  6. Declaration comparator flags mismatches

Usage:
  python inference/hybrid_pipeline.py \
      --image data/stcray_raw/STCray_TestSet/Images/gun1.jpg \
      --vlm-endpoint https://... \
      --yolo-model models/best.pt \
      --declared '{"electronics": true, "clothing": false}'
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from inference.roi_extractor import create_proposal_generator


# --- Customs Category Mapping ---

CUSTOMS_CATEGORIES = {
    "electronics": [
        "laptop", "computer", "phone", "mobile", "tablet", "ipad", "charger",
        "powerbank", "power bank", "battery", "camera", "headphone", "earphone",
        "cable", "wire", "circuit", "device", "electronic", "gadget", "speaker",
        "monitor", "keyboard", "mouse", "usb", "adapter", "router",
    ],
    "clothing": [
        "clothing", "clothes", "shirt", "pants", "dress", "jacket", "coat",
        "shoes", "shoe", "boot", "sock", "hat", "belt", "fabric", "textile",
        "garment", "scarf", "glove", "sweater", "jeans",
    ],
    "liquids": [
        "bottle", "water", "liquid", "drink", "spray", "aerosol", "perfume",
        "container", "flask", "thermos", "vacuum cup", "can", "jar",
    ],
    "food": [
        "food", "snack", "fruit", "vegetable", "candy", "chocolate", "bread",
        "meat", "cheese", "cereal", "rice", "pasta", "canned food", "beverage",
        "juice", "milk", "carton", "tin",
    ],
    "toiletries": [
        "cosmetic", "makeup", "shampoo", "soap", "toothbrush", "toothpaste",
        "razor", "deodorant", "lotion", "cream", "toiletry", "hygiene",
    ],
    "metal_tools": [
        "tool", "wrench", "pliers", "screwdriver", "hammer", "spanner",
        "drill", "saw", "clamp", "vice",
    ],
    "weapons": [
        "gun", "firearm", "pistol", "rifle", "handgun", "weapon", "explosive",
        "ammunition", "bullet", "grenade",
    ],
    "sharp_objects": [
        "knife", "blade", "dagger", "scissors", "cutter", "razor", "needle",
        "syringe", "injection", "nail cutter", "sharp", "sword",
    ],
    "flammable": [
        "lighter", "match", "flammable", "fuel", "gas canister",
    ],
    "documents": [
        "book", "document", "paper", "passport", "magazine", "notebook",
        "envelope", "folder", "office supply",
    ],
    "medication": [
        "medicine", "medication", "pill", "drug", "pharmaceutical", "syringe",
        "medical",
    ],
    "other": [
        "umbrella", "toy", "key", "wallet", "jewelry", "watch", "glasses",
        "bag", "purse", "backpack", "suitcase", "tableware", "lamp",
    ],
}


ROI_CLASSIFY_PROMPT = (
    "What is this item in the X-ray scan? "
    "Respond with JSON: {\"category\": \"...\", \"description\": \"...\"}"
)

FULL_IMAGE_PROMPT = (
    "You are an expert X-ray baggage scanner. Identify every distinct item visible in this scan. "
    "Include ALL items: electronics, clothing, bottles, tools, weapons, food, toiletries, bags, etc. "
    "Use short descriptions (1-3 words). Respond as JSON with \"items\" array and \"total_count\"."
)

FULL_IMAGE_SCHEMA = {
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

ROI_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["category", "description"],
}


def encode_pil_image(image: Image.Image) -> str:
    """Encode a PIL image to a base64 data URI."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def call_vlm(
    endpoint: str,
    model: str,
    image: Image.Image | str,
    prompt: str,
    schema: dict | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict:
    """Call the VLM endpoint with an image and prompt."""
    if isinstance(image, str):
        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        uri = f"data:image/jpeg;base64,{b64}"
    else:
        uri = encode_pil_image(image)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }

    resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        # Try extracting complete item objects from potentially truncated JSON
        items = re.findall(
            r'\{\s*"category"\s*:\s*"([^"]*)"\s*,\s*"description"\s*:\s*"([^"]*)"\s*\}',
            content,
        )
        if items:
            return {
                "items": [{"category": c, "description": d} for c, d in items],
                "total_count": len(items),
            }
        return {"raw": content}


def map_to_customs_categories(items: list[dict]) -> dict[str, list[str]]:
    """Map detected items to customs declaration categories."""
    result: dict[str, list[str]] = {}
    for item in items:
        cat = item.get("category", "").lower()
        desc = item.get("description", "").lower()
        text = f"{cat} {desc}"

        matched = False
        for customs_cat, keywords in CUSTOMS_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                result.setdefault(customs_cat, []).append(
                    item.get("description", cat)
                )
                matched = True
                break
        if not matched:
            result.setdefault("other", []).append(item.get("description", cat))

    return result


def compare_with_declaration(
    detected_categories: dict[str, list[str]],
    declared_categories: dict[str, bool],
) -> dict:
    """Compare detected categories against passenger declaration."""
    detected_set = set(detected_categories.keys())
    declared_set = {k for k, v in declared_categories.items() if v}

    matched = detected_set & declared_set
    undeclared = detected_set - declared_set - {"other"}
    undetected = declared_set - detected_set

    mismatches = []
    for cat in undeclared:
        mismatches.append({
            "category": cat,
            "type": "UNDECLARED",
            "severity": "high" if cat in ("weapons", "sharp_objects", "flammable") else "medium",
            "evidence": detected_categories[cat],
        })
    for cat in undetected:
        mismatches.append({
            "category": cat,
            "type": "DECLARED_NOT_FOUND",
            "severity": "low",
            "evidence": [],
        })

    return {
        "declaration_match": len(undeclared) == 0,
        "matched_categories": sorted(matched),
        "undeclared_categories": sorted(undeclared),
        "undetected_categories": sorted(undetected),
        "mismatches": mismatches,
        "detected_summary": {k: len(v) for k, v in detected_categories.items()},
    }


class XScanAgentPipeline:
    """End-to-end YOLO-VLM hybrid pipeline for X-ray baggage understanding."""

    def __init__(
        self,
        vlm_endpoint: str,
        vlm_model: str | None = None,
        yolo_model_path: str | None = None,
        yolo_endpoint: str | None = None,
        yolo_conf: float = 0.15,
        classify_rois: bool = True,
        full_image_pass: bool = True,
    ):
        self.vlm_endpoint = vlm_endpoint
        self.full_image_pass = full_image_pass
        self.classify_rois = classify_rois

        # Auto-detect VLM model name
        if vlm_model:
            self.vlm_model = vlm_model
        else:
            base = vlm_endpoint.rsplit("/chat/completions", 1)[0]
            models = requests.get(f"{base}/models", timeout=10).json()
            self.vlm_model = models["data"][0]["id"]

        # Proposal generator (YOLO API, local YOLO, or grid fallback)
        self.proposal_gen = create_proposal_generator(
            model_path=yolo_model_path,
            endpoint=yolo_endpoint,
            conf_threshold=yolo_conf,
        )

    def run(
        self,
        image_path: str,
        declared_categories: dict[str, bool] | None = None,
    ) -> dict:
        """
        Run the full XScan-Agent pipeline on an image.

        Returns dict with: yolo_detections, vlm_items, fused_items,
        customs_categories, declaration_comparison, timing.
        """
        timing = {}
        t0 = time.time()

        # Step 1: YOLO proposals
        t1 = time.time()
        rois = self.proposal_gen.extract_rois(image_path)
        timing["yolo_proposals_ms"] = (time.time() - t1) * 1000

        # Step 2: VLM classification per ROI
        roi_items = []
        if self.classify_rois and rois:
            t2 = time.time()
            for roi in rois:
                try:
                    result = call_vlm(
                        self.vlm_endpoint,
                        self.vlm_model,
                        roi["crop"],
                        ROI_CLASSIFY_PROMPT,
                        schema=ROI_SCHEMA,
                        max_tokens=128,
                    )
                    roi_items.append({
                        "category": result.get("category", "unknown"),
                        "description": result.get("description", ""),
                        "source": "yolo_roi",
                        "yolo_class": roi["class_name"],
                        "yolo_confidence": roi["confidence"],
                        "location": roi["location"],
                        "bbox": roi["bbox"],
                    })
                except Exception as e:
                    roi_items.append({
                        "category": "unknown",
                        "description": f"error: {e}",
                        "source": "yolo_roi",
                        "yolo_class": roi["class_name"],
                        "location": roi["location"],
                        "bbox": roi["bbox"],
                    })
            timing["vlm_roi_classify_ms"] = (time.time() - t2) * 1000

        # Step 3: Full-image VLM pass
        full_image_items = []
        if self.full_image_pass:
            t3 = time.time()
            result = call_vlm(
                self.vlm_endpoint,
                self.vlm_model,
                image_path,
                FULL_IMAGE_PROMPT,
                schema=FULL_IMAGE_SCHEMA,
                max_tokens=2048,
            )
            for item in result.get("items", []):
                full_image_items.append({
                    "category": item.get("category", "unknown"),
                    "description": item.get("description", ""),
                    "source": "vlm_full_image",
                })
            timing["vlm_full_image_ms"] = (time.time() - t3) * 1000

        # Step 4: Fuse results (ROI items take priority, full-image fills gaps)
        fused = list(roi_items)
        roi_categories = {it["category"].lower() for it in roi_items}
        for item in full_image_items:
            if item["category"].lower() not in roi_categories:
                fused.append(item)

        # Step 5: Map to customs categories
        customs = map_to_customs_categories(fused)

        # Step 6: Compare with declaration
        declaration_result = None
        if declared_categories:
            declaration_result = compare_with_declaration(customs, declared_categories)

        timing["total_ms"] = (time.time() - t0) * 1000

        return {
            "image_path": image_path,
            "yolo_detections": len(rois),
            "roi_items": roi_items,
            "full_image_items": full_image_items,
            "fused_items": fused,
            "customs_categories": customs,
            "declaration_comparison": declaration_result,
            "timing": timing,
        }


def main():
    parser = argparse.ArgumentParser(description="XScan-Agent Hybrid Pipeline")
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    parser.add_argument("--vlm-endpoint", required=True, help="VLM chat completions URL")
    parser.add_argument("--vlm-model", default=None, help="VLM model name (auto-detected)")
    parser.add_argument("--yolo-model", default=None, help="Path to YOLO .pt model")
    parser.add_argument("--yolo-endpoint", default=None, help="YOLO API endpoint URL")
    parser.add_argument("--declared", default=None, help="Declaration JSON string")
    parser.add_argument("--no-roi-classify", action="store_true", help="Skip per-ROI VLM classification")
    parser.add_argument("--no-full-image", action="store_true", help="Skip full-image VLM pass")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    declared = json.loads(args.declared) if args.declared else None

    pipeline = XScanAgentPipeline(
        vlm_endpoint=args.vlm_endpoint,
        vlm_model=args.vlm_model,
        yolo_model_path=args.yolo_model,
        yolo_endpoint=args.yolo_endpoint,
        classify_rois=not args.no_roi_classify,
        full_image_pass=not args.no_full_image,
    )

    print(f"Running XScan-Agent on: {args.image}")
    result = pipeline.run(args.image, declared)

    # Print summary
    print(f"\n{'='*60}")
    print("XScan-Agent Results")
    print(f"{'='*60}")
    print(f"  YOLO proposals:     {result['yolo_detections']}")
    print(f"  ROI-classified:     {len(result['roi_items'])}")
    print(f"  Full-image items:   {len(result['full_image_items'])}")
    print(f"  Fused total:        {len(result['fused_items'])}")
    print(f"\n  Customs categories detected:")
    for cat, items in result["customs_categories"].items():
        print(f"    {cat}: {len(items)} items ({', '.join(items[:3])})")

    if result["declaration_comparison"]:
        dc = result["declaration_comparison"]
        status = "MATCH" if dc["declaration_match"] else "MISMATCH"
        print(f"\n  Declaration: {status}")
        if dc["undeclared_categories"]:
            print(f"    UNDECLARED: {', '.join(dc['undeclared_categories'])}")
        if dc["undetected_categories"]:
            print(f"    Declared but not found: {', '.join(dc['undetected_categories'])}")

    print(f"\n  Timing:")
    for k, v in result["timing"].items():
        print(f"    {k}: {v:.0f}ms")

    # Save output
    if args.output:
        # Remove PIL crops before serializing
        for item in result.get("roi_items", []):
            item.pop("crop", None)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
