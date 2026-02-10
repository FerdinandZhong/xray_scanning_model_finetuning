#!/usr/bin/env python3
"""
Test PaddleOCR model on Luggage X-ray dataset and compute confusion matrix.

This script:
1. Selects N random samples from Luggage X-ray validation set
2. Tests PaddleOCR on each image
3. Computes confusion matrix and metrics
4. Saves results and visualizations
"""

import argparse
import base64
import json
import mimetypes
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


# Category mapping (12 classes)
CATEGORIES = [
    'blade',
    'Cans',
    'CartonDrinks',
    'dagger',
    'GlassBottle',
    'knife',
    'PlasticBottle',
    'scissors',
    'SprayCans',
    'SwissArmyKnife',
    'Tin',
    'VacuumCup',
]

# Normalization mapping for PaddleOCR text detections
NORMALIZE_MAP = {
    'swiss army knife': 'SwissArmyKnife',
    'swiss knife': 'SwissArmyKnife',
    'army knife': 'SwissArmyKnife',
    'carton drinks': 'CartonDrinks',
    'carton drink': 'CartonDrinks',
    'carton': 'CartonDrinks',
    'glass bottle': 'GlassBottle',
    'bottle glass': 'GlassBottle',
    'plastic bottle': 'PlasticBottle',
    'bottle plastic': 'PlasticBottle',
    'spray cans': 'SprayCans',
    'spray can': 'SprayCans',
    'spray': 'SprayCans',
    'vacuum cup': 'VacuumCup',
    'thermos': 'VacuumCup',
    'can': 'Cans',
    'cans': 'Cans',
    'scissor': 'scissors',
    'knives': 'knife',
    'blades': 'blade',
    'tin': 'Tin',
    'tins': 'Tin',
    'bottle': 'PlasticBottle',  # Default to plastic
}


def parse_loc_bbox(loc_string: str) -> List[Tuple[str, List[float]]]:
    """
    Parse bounding boxes from <loc####> format.
    
    Returns: [(category, [x1, y1, x2, y2]), ...]
    """
    import re
    pattern = r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s+(\w+)'
    matches = re.findall(pattern, loc_string)
    
    results = []
    for x1, y1, x2, y2, category in matches:
        bbox = [
            int(x1) / 1000.0,
            int(y1) / 1000.0,
            int(x2) / 1000.0,
            int(y2) / 1000.0
        ]
        results.append((category, bbox))
    
    return results


def normalize_category(name: str) -> str:
    """Normalize category name to match our standard categories."""
    name_lower = name.lower().strip().replace('_', ' ')
    
    # Check direct mapping
    if name_lower in NORMALIZE_MAP:
        return NORMALIZE_MAP[name_lower]
    
    # Check if it's already a standard category
    if name in CATEGORIES:
        return name
    
    # Check with underscores
    name_normalized = name_lower.replace(' ', '_')
    if name_normalized in CATEGORIES:
        return name_normalized
    
    # Partial matching
    for standard in CATEGORIES:
        standard_lower = standard.lower()
        if standard_lower in name_lower or name_lower in standard_lower:
            return standard
    
    return 'unknown'


def select_samples(
    jsonl_file: Path,
    images_dir: Path,
    num_samples: int = 10,
    seed: int = 42
) -> List[Tuple[Path, str, int]]:
    """
    Select random samples from Luggage X-ray validation set.
    
    Returns: List of (image_path, ground_truth_category, index)
    """
    random.seed(seed)
    
    # Load all annotations
    annotations = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))
    
    # Sample random annotations
    if num_samples > len(annotations):
        print(f"Warning: Only {len(annotations)} images available, using all")
        sampled = annotations
    else:
        sampled = random.sample(annotations, num_samples)
    
    # Prepare samples
    samples = []
    for idx, ann in enumerate(sampled):
        # Get assistant response with bounding boxes
        assistant_msg = ann['messages'][-1]['content']
        
        # Parse ground truth objects
        objects = parse_loc_bbox(assistant_msg)
        
        if not objects:
            continue
        
        # Get primary category (first object)
        primary_category = objects[0][0]
        
        # Image is stored as valid_XXXXXX.jpg
        image_name = f"valid_{idx:06d}.jpg"
        image_path = images_dir / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        samples.append((image_path, primary_category, idx))
    
    return samples


def encode_image(image_path: Path) -> str:
    """Encode image to base64 data URL."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"  # Default
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{base64_image}"


def query_paddleocr(
    image_path: Path,
    api_endpoint: str,
    api_key: str
) -> Dict:
    """
    Query PaddleOCR with an X-ray image.
    
    Returns detection results from OCR model.
    """
    try:
        # Encode image
        image_data_url = encode_image(image_path)
        
        # Prepare payload (PaddleOCR format)
        payload = {
            "input": [{
                "type": "image_url",
                "url": image_data_url,
            }]
        }
        
        # Make request
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    except Exception as e:
        print(f"Error querying PaddleOCR: {e}")
        return {"output": []}


def extract_predicted_category(result: Dict, ground_truth: str) -> str:
    """
    Extract predicted category from PaddleOCR results.
    
    PaddleOCR returns text detections, so we need to:
    1. Extract all detected text
    2. Try to match with our categories
    3. If no match, return 'unknown'
    """
    if not result or 'output' not in result:
        return 'unknown'
    
    # Extract all detected text
    detected_texts = []
    for output_item in result.get('output', []):
        if isinstance(output_item, dict):
            # Extract text from various possible fields
            text = output_item.get('text', output_item.get('label', ''))
            if text:
                detected_texts.append(text)
        elif isinstance(output_item, str):
            detected_texts.append(output_item)
    
    if not detected_texts:
        return 'unknown'
    
    # Try to match detected text with categories
    all_text = ' '.join(detected_texts).lower()
    
    # Check for category names in detected text
    for category in CATEGORIES:
        category_lower = category.lower()
        if category_lower in all_text:
            return category
    
    # Check normalized mappings
    for text in detected_texts:
        normalized = normalize_category(text)
        if normalized != 'unknown':
            return normalized
    
    # If no match found, return ground truth as fallback (OCR model limitation)
    return 'unknown'


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    categories: List[str]
) -> Dict:
    """Compute confusion matrix and classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=categories,
        output_dict=True,
        zero_division=0
    )
    
    # Compute accuracy manually if not in report
    accuracy = report.get('accuracy', sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))
    
    return {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'accuracy': accuracy,
        'macro_avg': report.get('macro avg', {}),
        'weighted_avg': report.get('weighted avg', {})
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    categories: List[str],
    output_path: Path,
    normalize: bool = True
):
    """Plot and save confusion matrix."""
    if normalize:
        # Normalize by row (true labels)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        title = 'Normalized Confusion Matrix (PaddleOCR on Luggage X-ray)'
        fmt = '.2f'
        vmax = 1.0
    else:
        cm_norm = cm
        title = 'Confusion Matrix (PaddleOCR on Luggage X-ray)'
        fmt = 'd'
        vmax = None
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories,
        vmin=0,
        vmax=vmax,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")


def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: Path,
    api_endpoint: str
):
    """Save detailed results to JSON and text files."""
    # Save JSON
    json_path = output_dir / f"paddleocr_luggage_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'model': 'PaddleOCR',
            'api_endpoint': api_endpoint,
            'dataset': 'Luggage X-ray (yolov5xray)',
            'num_samples': len(results),
            'metrics': metrics,
            'samples': results
        }, f, indent=2)
    print(f"✓ Results saved to: {json_path}")
    
    # Save text report
    report_path = output_dir / f"paddleocr_luggage_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PaddleOCR on Luggage X-ray Dataset - Test Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: PaddleOCR\n")
        f.write(f"API Endpoint: {api_endpoint}\n")
        f.write(f"Dataset: Luggage X-ray (yolov5xray)\n")
        f.write(f"Number of samples: {len(results)}\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2%}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("NOTE: PaddleOCR is an OCR (text detection) model, not an object\n")
        f.write("detection model. Results may be limited as X-ray images contain\n")
        f.write("minimal text. This test demonstrates API compatibility.\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        
        report = metrics['classification_report']
        # Print per-class metrics
        f.write(f"{'Category':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-" * 80 + "\n")
        
        for category in CATEGORIES:
            if category in report:
                cat_metrics = report[category]
                f.write(f"{category:<20} {cat_metrics['precision']:>10.2f} "
                       f"{cat_metrics['recall']:>10.2f} "
                       f"{cat_metrics['f1-score']:>10.2f} "
                       f"{int(cat_metrics['support']):>10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Aggregate Metrics\n")
        f.write("=" * 80 + "\n\n")
        
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                avg_metrics = report[avg_type]
                f.write(f"{avg_type}:\n")
                f.write(f"  Precision: {avg_metrics.get('precision', 0.0):.4f}\n")
                f.write(f"  Recall:    {avg_metrics.get('recall', 0.0):.4f}\n")
                f.write(f"  F1-Score:  {avg_metrics.get('f1-score', 0.0):.4f}\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Sample Results (First 10)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results[:10], 1):
            f.write(f"{i}. {result['image_name']}\n")
            f.write(f"   Ground Truth: {result['ground_truth']}\n")
            f.write(f"   Predicted: {result['predicted']}\n")
            f.write(f"   Correct: {'✓' if result['correct'] else '✗'}\n")
            f.write(f"   Detected Texts: {result.get('detected_texts', [])}\n")
            f.write(f"   Raw Output: {str(result.get('raw_output', {}))[:200]}...\n\n")
    
    print(f"✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test PaddleOCR on Luggage X-ray dataset"
    )
    parser.add_argument(
        '--api-endpoint',
        type=str,
        required=True,
        help='PaddleOCR API endpoint URL'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to test'
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        default=Path('data/luggage_xray/_annotations.valid.jsonl'),
        help='Path to validation annotations'
    )
    parser.add_argument(
        '--images-dir',
        type=Path,
        default=Path('data/luggage_xray_yolo/images/valid'),
        help='Path to validation images directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('test_results/paddleocr_luggage'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample selection'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PaddleOCR Luggage X-ray Testing")
    print("=" * 80)
    print(f"API Endpoint: {args.api_endpoint}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Annotations: {args.annotations}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    print()
    print("NOTE: PaddleOCR is an OCR model designed for text detection.")
    print("X-ray images may not contain readable text, so results may be limited.")
    print("=" * 80)
    print()
    
    # Get API key
    api_key = os.getenv("JWT_TOKEN")
    if not api_key:
        raise ValueError("JWT_TOKEN environment variable not set")
    
    # Load annotations
    print("Loading annotations...")
    if not args.annotations.exists():
        print(f"❌ Error: Annotations not found: {args.annotations}")
        return 1
    
    print(f"✓ Found annotations: {args.annotations}")
    print()
    
    # Select samples
    print(f"Selecting {args.num_samples} random samples...")
    samples = select_samples(args.annotations, args.images_dir, args.num_samples, args.seed)
    print(f"✓ Selected {len(samples)} valid samples")
    print()
    
    # Test each sample
    print("Testing samples...")
    results = []
    y_true = []
    y_pred = []
    
    for image_path, ground_truth, idx in tqdm(samples):
        # Query model
        ocr_result = query_paddleocr(image_path, args.api_endpoint, api_key)
        
        # Extract detected texts
        detected_texts = []
        for output_item in ocr_result.get('output', []):
            if isinstance(output_item, dict):
                text = output_item.get('text', output_item.get('label', ''))
                if text:
                    detected_texts.append(text)
            elif isinstance(output_item, str):
                detected_texts.append(output_item)
        
        # Extract predicted category
        predicted = extract_predicted_category(ocr_result, ground_truth)
        
        # Record results
        y_true.append(ground_truth)
        y_pred.append(predicted)
        
        results.append({
            'image_name': image_path.name,
            'image_path': str(image_path),
            'index': idx,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': ground_truth == predicted,
            'detected_texts': detected_texts,
            'raw_output': ocr_result
        })
    
    print()
    print("=" * 80)
    print("Computing Metrics")
    print("=" * 80)
    print()
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, CATEGORIES)
    
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    if metrics['macro_avg']:
        print(f"Macro Avg F1: {metrics['macro_avg'].get('f1-score', 0.0):.4f}")
    if metrics['weighted_avg']:
        print(f"Weighted Avg F1: {metrics['weighted_avg'].get('f1-score', 0.0):.4f}")
    print()
    
    # Plot confusion matrix
    print("Generating visualizations...")
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        CATEGORIES,
        args.output_dir / 'confusion_matrix.png',
        normalize=True
    )
    plot_confusion_matrix(
        cm,
        CATEGORIES,
        args.output_dir / 'confusion_matrix_raw.png',
        normalize=False
    )
    print()
    
    # Save results
    print("Saving results...")
    save_results(results, metrics, args.output_dir, args.api_endpoint)
    print()
    
    print("=" * 80)
    print("✅ Testing Complete!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print()
    
    # Print summary
    correct = sum(1 for r in results if r['correct'])
    print(f"Correct: {correct}/{len(results)} ({correct/len(results):.1%})")
    print()
    
    # Show sample results
    print("Sample Results:")
    print("-" * 80)
    for i, result in enumerate(results[:5], 1):
        status = "✓" if result['correct'] else "✗"
        print(f"{i}. {status} {result['image_name']}")
        print(f"   GT: {result['ground_truth']:<15} Pred: {result['predicted']:<15}")
        print(f"   Texts: {result['detected_texts']}")
        print()
    
    return 0


if __name__ == '__main__':
    exit(main())
