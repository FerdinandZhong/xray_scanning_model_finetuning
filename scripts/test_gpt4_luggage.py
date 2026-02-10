#!/usr/bin/env python3
"""
Test GPT-4.1 model on Luggage X-ray dataset and compute confusion matrix.

This script:
1. Selects N random samples from Luggage X-ray validation set
2. Tests GPT-4.1 on each image
3. Computes confusion matrix and metrics
4. Saves results and visualizations
"""

import argparse
import base64
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


# Pydantic models for structured output
class DetectedItem(BaseModel):
    """Single detected item in X-ray scan."""
    name: str
    confidence: float = 1.0
    location: str = "center"


class XrayDetectionResult(BaseModel):
    """Complete detection result for an X-ray image."""
    items: List[DetectedItem]
    has_threats: bool = False
    summary: str = ""


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

# Normalization mapping for predictions
NORMALIZE_MAP = {
    'swiss army knife': 'SwissArmyKnife',
    'carton drinks': 'CartonDrinks',
    'carton drink': 'CartonDrinks',
    'glass bottle': 'GlassBottle',
    'plastic bottle': 'PlasticBottle',
    'spray cans': 'SprayCans',
    'spray can': 'SprayCans',
    'vacuum cup': 'VacuumCup',
    'can': 'Cans',
    'scissor': 'scissors',
    'knives': 'knife',
    'blades': 'blade',
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
    name_lower = name.lower().strip()
    
    # Check direct mapping
    if name_lower in NORMALIZE_MAP:
        return NORMALIZE_MAP[name_lower]
    
    # Check if it's already a standard category
    if name in CATEGORIES:
        return name
    
    # Partial matching
    for standard in CATEGORIES:
        if standard.lower() in name_lower or name_lower in standard.lower():
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
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def query_gpt4(
    client: OpenAI,
    image_path: Path,
    model_id: str = "gpt-4o"
) -> XrayDetectionResult:
    """
    Query GPT-4 with an X-ray image using structured output.
    """
    # Encode image
    base64_image = encode_image(image_path)
    
    # Create prompt
    prompt = f"""You are an expert X-ray luggage screening assistant. Analyze this X-ray image and identify all objects present.

Classify objects into these categories:
{', '.join(CATEGORIES)}

Provide:
1. List of detected items with confidence scores (0.0-1.0)
2. A brief summary of the image contents

Be specific and accurate. If you see multiple items, list the most prominent one first."""

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "xray_detection",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "location": {"type": "string"}
                                    },
                                    "required": ["name", "confidence", "location"],
                                    "additionalProperties": False
                                }
                            },
                            "has_threats": {"type": "boolean"},
                            "summary": {"type": "string"}
                        },
                        "required": ["items", "has_threats", "summary"],
                        "additionalProperties": False
                    }
                }
            },
            max_tokens=1000,
            temperature=0.0
        )
        
        # Parse response
        result_json = json.loads(response.choices[0].message.content)
        return XrayDetectionResult(**result_json)
    
    except Exception as e:
        print(f"Error querying GPT-4: {e}")
        # Return empty result on error
        return XrayDetectionResult(
            items=[DetectedItem(name="unknown", confidence=0.0, location="center")],
            has_threats=False,
            summary="Error during inference"
        )


def extract_predicted_category(result: XrayDetectionResult) -> str:
    """Extract and normalize the primary predicted category."""
    if not result.items:
        return 'unknown'
    
    # Get the first (primary) detected item
    primary_item = result.items[0].name
    
    # Normalize it
    return normalize_category(primary_item)


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
        title = 'Normalized Confusion Matrix (GPT-4.1 on Luggage X-ray)'
        fmt = '.2f'
        vmax = 1.0
    else:
        cm_norm = cm
        title = 'Confusion Matrix (GPT-4.1 on Luggage X-ray)'
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
    model_id: str
):
    """Save detailed results to JSON and text files."""
    # Save JSON
    json_path = output_dir / f"gpt4_luggage_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'model': model_id,
            'dataset': 'Luggage X-ray (yolov5xray)',
            'num_samples': len(results),
            'metrics': metrics,
            'samples': results
        }, f, indent=2)
    print(f"✓ Results saved to: {json_path}")
    
    # Save text report
    report_path = output_dir / f"gpt4_luggage_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GPT-4.1 on Luggage X-ray Dataset - Test Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {model_id}\n")
        f.write(f"Dataset: Luggage X-ray (yolov5xray)\n")
        f.write(f"Number of samples: {len(results)}\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2%}\n\n")
        
        f.write("=" * 80 + "\n")
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
            f.write(f"   Detected Items: {', '.join([item['name'] for item in result['detected_items']])}\n")
            f.write(f"   Summary: {result['summary']}\n\n")
    
    print(f"✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test GPT-4.1 on Luggage X-ray dataset"
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='https://api.openai.com/v1',
        help='OpenAI API base URL'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default='gpt-4.1',
        help='Model ID to use'
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
        default=Path('test_results/gpt4_luggage'),
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
    print("GPT-4.1 Luggage X-ray Testing")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    print(f"Base URL: {args.base_url}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Annotations: {args.annotations}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    print()
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    
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
        detection_result = query_gpt4(client, image_path, args.model_id)
        
        # Extract predicted category
        predicted = extract_predicted_category(detection_result)
        
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
            'detected_items': [item.dict() for item in detection_result.items],
            'summary': detection_result.summary
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
    save_results(results, metrics, args.output_dir, args.model_id)
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
        print()
    
    return 0


if __name__ == '__main__':
    exit(main())
