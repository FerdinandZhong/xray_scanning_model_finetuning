#!/usr/bin/env python3
"""
Test YOLO API on validation dataset with custom confidence threshold.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import requests
from collections import Counter
import yaml

# Class names from data.yaml
CLASS_NAMES = ['blade', 'Cans', 'CartonDrinks', 'dagger', 'GlassBottle', 'knife', 
               'PlasticBottle', 'scissors', 'SprayCans', 'SwissArmyKnife', 'Tin', 'VacuumCup']

THREATS = ['blade', 'dagger', 'knife', 'scissors', 'SwissArmyKnife']


def test_api_health(api_url: str) -> bool:
    """Test API health."""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()
        health = response.json()
        return health.get("status") == "healthy" and health.get("model_loaded")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def call_yolo_api(api_url: str, image_path: Path, conf_threshold: float = 0.25) -> Dict:
    """Call YOLO API for detection with custom confidence threshold."""
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            params = {"conf_threshold": conf_threshold}
            response = requests.post(f"{api_url}/v1/detect", files=files, params=params, timeout=30)
            response.raise_for_status()
        
        result = response.json()
        items = result.get("items", [])
        
        return {
            "detections": items,
            "num_detections": result.get("total_count", 0),
            "threats": [item['name'] for item in items if item['name'] in THREATS],
            "success": True
        }
    except Exception as e:
        return {"detections": [], "num_detections": 0, "threats": [], "success": False, "error": str(e)}


def get_ground_truth(label_path: Path) -> str:
    """Get ground truth class from YOLO label file."""
    if not label_path.exists():
        return "unknown"
    
    try:
        with open(label_path) as f:
            lines = f.read().strip().split('\n')
            if not lines or not lines[0]:
                return "unknown"
            
            # Parse first line (class_id x y w h)
            class_id = int(lines[0].split()[0])
            if 0 <= class_id < len(CLASS_NAMES):
                return CLASS_NAMES[class_id]
    except:
        pass
    
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Test YOLO API with custom confidence threshold")
    parser.add_argument('--api-url', default='https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site')
    parser.add_argument('--dataset', default='data/luggage_xray_yolo')
    parser.add_argument('--conf-threshold', type=float, default=0.10, help='Confidence threshold (default: 0.10)')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit number of samples')
    args, _ = parser.parse_known_args()
    
    dataset_path = Path(args.dataset)
    images_dir = dataset_path / "images" / "valid"
    labels_dir = dataset_path / "labels" / "valid"
    
    print(f"\n🔬 YOLO API Validation Test (Confidence: {args.conf_threshold})")
    print(f"   API: {args.api_url}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Threshold: {args.conf_threshold}")
    print()
    
    # Check API health
    print("Checking API health...")
    if not test_api_health(args.api_url):
        return 1
    print("✓ API is healthy\n")
    
    # Get all image files
    image_files = sorted(list(images_dir.glob("*.jpg")))
    if args.max_samples:
        image_files = image_files[:args.max_samples]
    
    total = len(image_files)
    print(f"Testing {total} images...\n")
    
    # Test all images
    results = []
    start_time = time.time()
    
    for i, image_path in enumerate(image_files):
        label_path = labels_dir / (image_path.stem + ".txt")
        ground_truth = get_ground_truth(label_path)
        
        # Call API with custom threshold
        api_result = call_yolo_api(args.api_url, image_path, args.conf_threshold)
        
        # Get predicted class (highest confidence)
        predicted = "unknown"
        if api_result["detections"]:
            best = max(api_result["detections"], key=lambda d: d.get('confidence', 0))
            predicted = best.get('name', 'unknown')
        
        correct = predicted.lower() == ground_truth.lower()
        
        results.append({
            "image": image_path.name,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "num_detections": api_result["num_detections"],
            "threats": api_result["threats"]
        })
        
        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"[{i+1}/{total}] Progress... ({rate:.1f} img/s, ~{remaining/60:.1f}min remaining)")
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / total * 100
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / total
    images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
    detection_rate = images_with_detections / total * 100
    threats_found = sum(1 for r in results if r['threats'])
    
    # Class distribution
    gt_counter = Counter(r['ground_truth'] for r in results)
    pred_counter = Counter(r['predicted'] for r in results)
    
    # Per-class accuracy
    class_accuracy = {}
    for gt_class in gt_counter.keys():
        class_samples = [r for r in results if r['ground_truth'] == gt_class]
        class_correct = sum(1 for r in class_samples if r['correct'])
        class_accuracy[gt_class] = {
            'correct': class_correct,
            'total': len(class_samples),
            'accuracy': class_correct / len(class_samples) * 100 if class_samples else 0
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary (Confidence Threshold: {args.conf_threshold})")
    print(f"{'='*60}\n")
    print(f"Overall Accuracy:  {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Detection Rate:    {detection_rate:.2f}% ({images_with_detections}/{total} images)")
    print(f"Avg Detections:    {avg_detections:.2f}")
    print(f"Threats Found:     {threats_found}")
    print(f"Test Duration:     {elapsed_time/60:.1f} minutes")
    print(f"Processing Rate:   {total/elapsed_time:.1f} images/second")
    print()
    
    # Top predictions
    print("Top 10 Predicted Classes:")
    for pred_class, count in pred_counter.most_common(10):
        print(f"  {pred_class:20s} : {count:4d}")
    print()
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    for gt_class in sorted(class_accuracy.keys(), key=lambda x: gt_counter[x], reverse=True):
        stats = class_accuracy[gt_class]
        print(f"  {gt_class:20s} : {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")
    print()
    
    # Save results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"yolo_trained_conf{args.conf_threshold:.2f}_{total}samples.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "api_url": args.api_url,
            "confidence_threshold": args.conf_threshold,
            "total_samples": total,
            "metrics": {
                "accuracy": accuracy,
                "correct": correct_count,
                "detection_rate": detection_rate,
                "avg_detections": avg_detections,
                "threats_found": threats_found,
                "test_duration_seconds": elapsed_time
            },
            "class_distribution": dict(gt_counter),
            "predictions_distribution": dict(pred_counter.most_common(20)),
            "per_class_accuracy": class_accuracy,
            "samples": results
        }, f, indent=2)
    
    print(f"✓ Results saved: {results_file}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
