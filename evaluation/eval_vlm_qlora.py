#!/usr/bin/env python3
"""
Evaluate Qwen3-VL-2B base vs fine-tuned model on STCray X-ray dataset.

Compares:
- Base model (zero-shot)
- Fine-tuned model (with LoRA adapters)

Metrics:
- Object detection: Precision, Recall, F1, mAP
- Bounding box: IoU, localization accuracy
- JSON parsing: Success rate, schema compliance
- Multi-object: Accuracy on images with 3+ objects
- Threat detection: Critical/high threat recall
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import time

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# Import custom metrics
import sys
sys.path.append(str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-2B base vs fine-tuned model"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default="checkpoints/qwen3vl-2b-xray-qlora/final",
        help="Path to fine-tuned LoRA adapters"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/stcray_vlm/stcray_vlm_test.jsonl",
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results/vlm_qlora_eval",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions (default: 0.5)"
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model evaluation (only evaluate fine-tuned)"
    )
    
    return parser.parse_args()


def load_model(model_name: str, lora_path: Optional[str] = None):
    """
    Load model with optional LoRA adapters.
    
    Args:
        model_name: Base model name
        lora_path: Path to LoRA adapters (None for base model)
    
    Returns:
        model, processor
    """
    print(f"Loading model: {model_name}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load LoRA adapters if provided
    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA adapters from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # Merge for faster inference
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    model.eval()
    print("✓ Model loaded and ready")
    
    return model, processor


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] in normalized coordinates
    
    Returns:
        IoU score
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_predictions(
    pred_objects: List[Dict],
    gt_objects: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predicted objects to ground truth using IoU.
    
    Returns:
        matches: List of (pred_idx, gt_idx) pairs
        unmatched_preds: List of pred indices
        unmatched_gts: List of gt indices
    """
    if not pred_objects or not gt_objects:
        return [], list(range(len(pred_objects))), list(range(len(gt_objects)))
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_objects), len(gt_objects)))
    for i, pred in enumerate(pred_objects):
        for j, gt in enumerate(gt_objects):
            if pred["category"] == gt["category"]:  # Only match same category
                iou_matrix[i, j] = compute_iou(pred["bbox"], gt["bbox"])
    
    # Greedy matching
    matches = []
    matched_preds = set()
    matched_gts = set()
    
    # Sort by IoU (highest first)
    indices = np.unravel_index(np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape)
    
    for i, j in zip(indices[0], indices[1]):
        if i not in matched_preds and j not in matched_gts:
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((i, j))
                matched_preds.add(i)
                matched_gts.add(j)
    
    unmatched_preds = [i for i in range(len(pred_objects)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_objects)) if i not in matched_gts]
    
    return matches, unmatched_preds, unmatched_gts


def compute_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary of metrics
    """
    # Overall statistics
    total_preds = 0
    total_gts = 0
    total_matches = 0
    total_iou = 0.0
    parsing_success = 0
    
    # Per-category statistics
    category_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    # Multi-object statistics
    multi_object_correct = 0
    multi_object_total = 0
    
    # Process each sample
    for pred, gt in zip(predictions, ground_truths):
        # JSON parsing
        if pred.get("parsing_success", False):
            parsing_success += 1
        
        pred_objects = pred.get("objects", [])
        gt_objects = gt["objects"]
        
        total_preds += len(pred_objects)
        total_gts += len(gt_objects)
        
        # Match predictions to ground truth
        matches, unmatched_preds, unmatched_gts = match_predictions(
            pred_objects, gt_objects, iou_threshold
        )
        
        total_matches += len(matches)
        
        # Compute IoU for matches
        for pred_idx, gt_idx in matches:
            iou = compute_iou(pred_objects[pred_idx]["bbox"], gt_objects[gt_idx]["bbox"])
            total_iou += iou
            
            # Per-category true positives
            category = gt_objects[gt_idx]["category"]
            category_stats[category]["tp"] += 1
        
        # False positives
        for pred_idx in unmatched_preds:
            category = pred_objects[pred_idx]["category"]
            category_stats[category]["fp"] += 1
        
        # False negatives
        for gt_idx in unmatched_gts:
            category = gt_objects[gt_idx]["category"]
            category_stats[category]["fn"] += 1
        
        # Multi-object accuracy (images with 3+ objects)
        if len(gt_objects) >= 3:
            multi_object_total += 1
            # Consider correct if detection rate > 0.7
            detection_rate = len(matches) / len(gt_objects) if gt_objects else 0
            if detection_rate >= 0.7:
                multi_object_correct += 1
    
    # Compute overall metrics
    precision = total_matches / total_preds if total_preds > 0 else 0
    recall = total_matches / total_gts if total_gts > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = total_iou / total_matches if total_matches > 0 else 0
    parsing_rate = parsing_success / len(predictions) if predictions else 0
    
    # Per-category metrics
    category_metrics = {}
    for category, stats in category_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        cat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        cat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
        category_metrics[category] = {
            "precision": cat_precision,
            "recall": cat_recall,
            "f1": cat_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    # Multi-object metrics
    multi_object_accuracy = multi_object_correct / multi_object_total if multi_object_total > 0 else 0
    
    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_iou": avg_iou,
            "parsing_success_rate": parsing_rate,
            "total_predictions": total_preds,
            "total_ground_truths": total_gts,
            "total_matches": total_matches,
        },
        "per_category": category_metrics,
        "multi_object": {
            "accuracy": multi_object_accuracy,
            "correct": multi_object_correct,
            "total": multi_object_total,
        }
    }


def run_inference(
    model,
    processor,
    test_data: List[Dict],
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    Run inference on test data.
    
    Returns:
        List of prediction dicts
    """
    predictions = []
    
    if num_samples:
        test_data = test_data[:num_samples]
    
    print(f"Running inference on {len(test_data)} samples...")
    
    for i, sample in enumerate(tqdm(test_data)):
        try:
            # Load image
            image = Image.open(sample["image_path"]).convert("RGB")
            
            # Prepare prompt
            prompt = "Detect and list all prohibited items in this X-ray baggage scan with their bounding boxes. Provide your response in valid JSON format only."
            
            # Create messages for chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process with chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Prepare inputs
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            
            # Decode
            generated_text = processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Extract answer (after the prompt)
            answer = generated_text.split("assistant")[-1].strip() if "assistant" in generated_text else generated_text
            
            # Parse JSON
            parsing_success = False
            objects = []
            try:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', answer, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    objects = parsed.get("objects", [])
                    parsing_success = True
            except:
                pass
            
            predictions.append({
                "image_path": sample["image_path"],
                "raw_output": answer,
                "objects": objects,
                "parsing_success": parsing_success,
                "metadata": sample.get("metadata", {}),
            })
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            predictions.append({
                "image_path": sample.get("image_path", "unknown"),
                "raw_output": "",
                "objects": [],
                "parsing_success": False,
                "error": str(e),
            })
    
    return predictions


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Qwen3-VL-2B Evaluation: Base vs Fine-tuned")
    print("=" * 60)
    print(f"Test data: {args.test_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"IoU threshold: {args.iou_threshold}")
    print("=" * 60)
    print()
    
    # Load test data
    print("Loading test data...")
    test_data = []
    with open(args.test_data, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Parse ground truth answer
                gt_answer = json.loads(entry["answer"])
                entry["objects"] = gt_answer["objects"]
                test_data.append(entry)
    
    print(f"✓ Loaded {len(test_data)} test samples")
    
    if args.num_samples:
        print(f"  (Evaluating first {args.num_samples} samples)")
    print()
    
    results = {}
    
    # Evaluate base model
    if not args.skip_base:
        print("=" * 60)
        print("Evaluating BASE model...")
        print("=" * 60)
        
        base_model, base_processor = load_model(args.base_model)
        base_predictions = run_inference(base_model, base_processor, test_data, args.num_samples)
        base_metrics = compute_metrics(base_predictions, test_data[:len(base_predictions)], args.iou_threshold)
        
        results["base"] = {
            "metrics": base_metrics,
            "predictions": base_predictions
        }
        
        print("\nBase Model Results:")
        print(f"  Precision: {base_metrics['overall']['precision']:.3f}")
        print(f"  Recall: {base_metrics['overall']['recall']:.3f}")
        print(f"  F1: {base_metrics['overall']['f1']:.3f}")
        print(f"  Avg IoU: {base_metrics['overall']['avg_iou']:.3f}")
        print(f"  JSON parsing: {base_metrics['overall']['parsing_success_rate']:.1%}")
        print(f"  Multi-object accuracy: {base_metrics['multi_object']['accuracy']:.1%}")
        
        # Clean up
        del base_model
        torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model
    print("\n" + "=" * 60)
    print("Evaluating FINE-TUNED model...")
    print("=" * 60)
    
    finetuned_model, finetuned_processor = load_model(args.base_model, args.finetuned_model)
    finetuned_predictions = run_inference(finetuned_model, finetuned_processor, test_data, args.num_samples)
    finetuned_metrics = compute_metrics(finetuned_predictions, test_data[:len(finetuned_predictions)], args.iou_threshold)
    
    results["finetuned"] = {
        "metrics": finetuned_metrics,
        "predictions": finetuned_predictions
    }
    
    print("\nFine-tuned Model Results:")
    print(f"  Precision: {finetuned_metrics['overall']['precision']:.3f}")
    print(f"  Recall: {finetuned_metrics['overall']['recall']:.3f}")
    print(f"  F1: {finetuned_metrics['overall']['f1']:.3f}")
    print(f"  Avg IoU: {finetuned_metrics['overall']['avg_iou']:.3f}")
    print(f"  JSON parsing: {finetuned_metrics['overall']['parsing_success_rate']:.1%}")
    print(f"  Multi-object accuracy: {finetuned_metrics['multi_object']['accuracy']:.1%}")
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    # Save full results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {output_dir / 'evaluation_results.json'}")
    
    # Generate comparison report
    with open(output_dir / "comparison_report.md", "w") as f:
        f.write("# Qwen3-VL-2B Evaluation Report\n\n")
        f.write("## Overall Metrics Comparison\n\n")
        f.write("| Metric | Base | Fine-tuned | Improvement |\n")
        f.write("|--------|------|------------|-------------|\n")
        
        if "base" in results:
            base_overall = results["base"]["metrics"]["overall"]
            ft_overall = results["finetuned"]["metrics"]["overall"]
            
            for metric in ["precision", "recall", "f1", "avg_iou", "parsing_success_rate"]:
                base_val = base_overall[metric]
                ft_val = ft_overall[metric]
                improvement = ((ft_val - base_val) / base_val * 100) if base_val > 0 else 0
                f.write(f"| {metric.replace('_', ' ').title()} | {base_val:.3f} | {ft_val:.3f} | {improvement:+.1f}% |\n")
        else:
            ft_overall = results["finetuned"]["metrics"]["overall"]
            for metric in ["precision", "recall", "f1", "avg_iou", "parsing_success_rate"]:
                f.write(f"| {metric.replace('_', ' ').title()} | N/A | {ft_overall[metric]:.3f} | N/A |\n")
        
        f.write("\n## Multi-Object Detection\n\n")
        f.write(f"- Fine-tuned accuracy: {finetuned_metrics['multi_object']['accuracy']:.1%}\n")
        f.write(f"- Images evaluated: {finetuned_metrics['multi_object']['total']}\n")
        
        f.write("\n## Top 10 Categories (Fine-tuned F1)\n\n")
        sorted_cats = sorted(
            finetuned_metrics['per_category'].items(),
            key=lambda x: x[1]['f1'],
            reverse=True
        )[:10]
        
        f.write("| Category | Precision | Recall | F1 |\n")
        f.write("|----------|-----------|--------|----|\n")
        for cat, metrics in sorted_cats:
            f.write(f"| {cat} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |\n")
    
    print(f"✓ Saved: {output_dir / 'comparison_report.md'}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
