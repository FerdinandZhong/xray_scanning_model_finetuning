#!/usr/bin/env python3
"""
Test RolmOCR model on CargoXray dataset and compute confusion matrix.

This script:
1. Selects N random samples from CargoXray test set
2. Tests RolmOCR on each image
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


# Category mapping (from convert_cargoxray_to_yolo.py)
CARGO_CATEGORIES = [
    'auto_parts', 'bags', 'bicycle', 'car_wheels', 'clothes', 'fabrics',
    'lamps', 'office_supplies', 'shoes', 'spare_parts', 'tableware',
    'textiles', 'tools', 'toys', 'unknown', 'xray_objects'
]

# Normalization mapping for RolmOCR predictions
NORMALIZE_MAP = {
    'auto parts': 'auto_parts',
    'car wheels': 'car_wheels',
    'car wheel': 'car_wheels',
    'office supplies': 'office_supplies',
    'spare parts': 'spare_parts',
    'table ware': 'tableware',
    'tableware': 'tableware',
    'textile': 'textiles',
    'tool': 'tools',
    'toy': 'toys',
    'xray objects': 'xray_objects',
    'xrayobjects': 'xray_objects',
}


def load_coco_annotations(annotation_file: Path) -> Dict:
    """Load COCO format annotations."""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def normalize_category(name: str) -> str:
    """Normalize category name to match our standard categories."""
    name_lower = name.lower().strip().replace('_', ' ')
    
    # Check direct mapping
    if name_lower in NORMALIZE_MAP:
        return NORMALIZE_MAP[name_lower]
    
    # Check if it's already a standard category
    name_normalized = name_lower.replace(' ', '_')
    if name_normalized in CARGO_CATEGORIES:
        return name_normalized
    
    # Partial matching
    for standard in CARGO_CATEGORIES:
        if standard.replace('_', '') in name_lower.replace(' ', ''):
            return standard
        if name_lower.replace(' ', '') in standard.replace('_', ''):
            return standard
    
    return 'unknown'


def select_samples(
    coco_data: Dict,
    images_dir: Path,
    num_samples: int = 100,
    seed: int = 42
) -> List[Tuple[Path, str, int]]:
    """
    Select random samples from CargoXray dataset.
    
    Returns: List of (image_path, ground_truth_category, coco_category_id)
    """
    random.seed(seed)
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create image info mapping
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Filter images that have annotations
    valid_image_ids = [img_id for img_id in images.keys() if img_id in annotations_by_image]
    
    # Sample random images
    if num_samples > len(valid_image_ids):
        print(f"Warning: Only {len(valid_image_ids)} images available, using all")
        sampled_ids = valid_image_ids
    else:
        sampled_ids = random.sample(valid_image_ids, num_samples)
    
    # Prepare samples
    samples = []
    for image_id in sampled_ids:
        img_info = images[image_id]
        image_path = images_dir / img_info['file_name']
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get primary category (first annotation)
        if annotations_by_image[image_id]:
            category_id = annotations_by_image[image_id][0]['category_id']
            category_name = categories[category_id]
            # Normalize category name
            normalized_name = normalize_category(category_name)
            samples.append((image_path, normalized_name, category_id))
    
    print(f"Selected {len(samples)} samples")
    return samples


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def query_rolmocr(
    client: OpenAI,
    image_path: Path,
    model_id: str = "stepfun-ai/GOT-OCR2_0",
    temperature: float = 0.1,
    max_tokens: int = 2048
) -> XrayDetectionResult:
    """Query RolmOCR model with structured output."""
    
    # Encode image
    base64_image = encode_image_base64(image_path)
    
    # Define JSON schema
    json_schema = XrayDetectionResult.model_json_schema()
    
    try:
        # Call OpenAI-compatible API with guided JSON
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert X-ray image analyzer for cargo screening. Identify all objects visible in the X-ray scan."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this cargo X-ray image and list all visible items. Focus on major objects like textiles, auto parts, tools, toys, shoes, bags, etc."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={
                "guided_json": json_schema
            }
        )
        
        # Parse response
        content = response.choices[0].message.content
        result = XrayDetectionResult.model_validate_json(content)
        return result
        
    except Exception as e:
        print(f"Error querying RolmOCR: {e}")
        return XrayDetectionResult(items=[], summary="Error")


def extract_predicted_category(result: XrayDetectionResult) -> str:
    """Extract primary predicted category from RolmOCR result."""
    if not result.items:
        return 'unknown'
    
    # Get first item (highest confidence / most prominent)
    item_name = result.items[0].name
    
    # Normalize to our categories
    normalized = normalize_category(item_name)
    return normalized


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    categories: List[str]
) -> Dict:
    """Compute confusion matrix and classification metrics."""
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        labels=categories,
        output_dict=True,
        zero_division=0
    )
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy,
        'total_samples': len(y_true)
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    categories: List[str],
    output_path: Path,
    normalize: bool = True
):
    """Plot and save confusion matrix."""
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
    else:
        cm_norm = cm
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Plot heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title('RolmOCR Confusion Matrix on CargoXray Dataset', fontsize=16, pad=20)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('True Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")
    plt.close()


def save_results(
    results: Dict,
    y_true: List[str],
    y_pred: List[str],
    sample_paths: List[Path],
    output_dir: Path
):
    """Save detailed results to JSON."""
    
    # Prepare detailed results
    detailed_results = {
        'summary': {
            'total_samples': results['total_samples'],
            'accuracy': results['accuracy'],
            'macro_avg': results['classification_report']['macro avg'],
            'weighted_avg': results['classification_report']['weighted avg']
        },
        'per_category': {
            cat: results['classification_report'][cat]
            for cat in CARGO_CATEGORIES
            if cat in results['classification_report']
        },
        'predictions': [
            {
                'image': str(path.name),
                'ground_truth': true,
                'predicted': pred,
                'correct': true == pred
            }
            for path, true, pred in zip(sample_paths, y_true, y_pred)
        ]
    }
    
    # Save to JSON
    output_file = output_dir / 'rolmocr_cargoxray_results.json'
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Saved detailed results to {output_file}")
    
    # Also save as text report
    report_file = output_dir / 'rolmocr_cargoxray_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RolmOCR Performance on CargoXray Dataset\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Samples: {results['total_samples']}\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.2%}\n\n")
        f.write("="*80 + "\n")
        f.write("Per-Category Performance\n")
        f.write("="*80 + "\n\n")
        
        for cat in CARGO_CATEGORIES:
            if cat in results['classification_report']:
                metrics = results['classification_report'][cat]
                f.write(f"{cat}:\n")
                f.write(f"  Precision: {metrics['precision']:.2%}\n")
                f.write(f"  Recall: {metrics['recall']:.2%}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.2%}\n")
                f.write(f"  Support: {metrics['support']}\n\n")
    
    print(f"Saved text report to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test RolmOCR on CargoXray dataset with confusion matrix"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.stepfun.com/v1",
        help="Base URL for OpenAI-compatible API"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stepfun-ai/GOT-OCR2_0",
        help="Model ID for RolmOCR"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/cargoxray"),
        help="Path to CargoXray dataset directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Which split to test on"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results/rolmocr_cargoxray"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    # Get JWT token from environment
    jwt_token = os.getenv("JWT_TOKEN")
    if not jwt_token:
        print("Error: JWT_TOKEN environment variable not set")
        print("Please set it with: export JWT_TOKEN='your_token_here'")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    client = OpenAI(
        base_url=args.base_url,
        api_key=jwt_token
    )
    
    print(f"\n{'='*80}")
    print("RolmOCR Testing on CargoXray Dataset")
    print(f"{'='*80}\n")
    
    # Load annotations
    annotation_file = args.dataset_dir / args.split / "_annotations.coco.json"
    images_dir = args.dataset_dir / args.split
    
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return 1
    
    print(f"Loading annotations from {annotation_file}...")
    coco_data = load_coco_annotations(annotation_file)
    
    # Select samples
    print(f"Selecting {args.num_samples} random samples from {args.split} split...")
    samples = select_samples(coco_data, images_dir, args.num_samples, args.seed)
    
    if not samples:
        print("Error: No valid samples found")
        return 1
    
    # Test RolmOCR on each sample
    print(f"\nTesting RolmOCR on {len(samples)} samples...")
    y_true = []
    y_pred = []
    sample_paths = []
    
    for image_path, ground_truth, _ in tqdm(samples, desc="Processing"):
        # Query RolmOCR
        result = query_rolmocr(client, image_path, args.model_id)
        
        # Extract predicted category
        predicted = extract_predicted_category(result)
        
        # Store results
        y_true.append(ground_truth)
        y_pred.append(predicted)
        sample_paths.append(image_path)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(y_true, y_pred, CARGO_CATEGORIES)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}\n")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}\n")
    
    print("Macro Average:")
    macro_avg = metrics['classification_report']['macro avg']
    print(f"  Precision: {macro_avg['precision']:.2%}")
    print(f"  Recall: {macro_avg['recall']:.2%}")
    print(f"  F1-Score: {macro_avg['f1-score']:.2%}\n")
    
    print("Weighted Average:")
    weighted_avg = metrics['classification_report']['weighted avg']
    print(f"  Precision: {weighted_avg['precision']:.2%}")
    print(f"  Recall: {weighted_avg['recall']:.2%}")
    print(f"  F1-Score: {weighted_avg['f1-score']:.2%}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = args.output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        CARGO_CATEGORIES,
        cm_path,
        normalize=True
    )
    
    # Also save unnormalized version
    cm_raw_path = args.output_dir / 'confusion_matrix_raw.png'
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        CARGO_CATEGORIES,
        cm_raw_path,
        normalize=False
    )
    
    # Save detailed results
    print("\nSaving detailed results...")
    save_results(metrics, y_true, y_pred, sample_paths, args.output_dir)
    
    print(f"\n{'='*80}")
    print("Testing complete!")
    print(f"{'='*80}\n")
    print(f"Results saved to: {args.output_dir}")
    print(f"  - confusion_matrix.png (normalized)")
    print(f"  - confusion_matrix_raw.png (counts)")
    print(f"  - rolmocr_cargoxray_results.json (detailed)")
    print(f"  - rolmocr_cargoxray_report.txt (summary)")
    
    return 0


if __name__ == "__main__":
    exit(main())
