#!/usr/bin/env python3
"""
Test RolmOCR model with OpenAI-compatible API.

Picks a random image from STCray dataset and queries the RolmOCR endpoint.
"""

import os
import json
import random
import argparse
import base64
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from pydantic import BaseModel


# Define Pydantic models for structured output schema
class DetectedItem(BaseModel):
    """Single detected item in X-ray scan."""
    category: str  # Item category (e.g., "Knife", "Gun", "Explosive")
    item_name: str  # Specific item name
    confidence: Optional[float] = None  # Confidence score (0.0-1.0)
    location: Optional[str] = None  # Location description (e.g., "center", "upper-left")


class XrayDetectionResult(BaseModel):
    """Complete X-ray detection result with structured items."""
    detected_items: List[DetectedItem]  # List of detected items
    total_count: int  # Total number of items detected
    has_threats: bool  # Whether any threats were detected
    summary: Optional[str] = None  # Brief summary of findings


def get_random_stcray_image(image_dir: str):
    """Get a random X-ray image from STCray dataset."""
    image_path = Path(image_dir)
    
    if not image_path.exists():
        print(f"Error: Image directory not found: {image_dir}")
        exit(1)
    
    # Find all JPG images recursively
    all_images = list(image_path.rglob("*.jpg")) + list(image_path.rglob("*.JPG"))
    
    if not all_images:
        print(f"Error: No images found in {image_dir}")
        exit(1)
    
    # Pick random image
    random_image = random.choice(all_images)
    return random_image


def load_ground_truth(image_filename: str, annotations_file: str = None) -> Optional[Dict]:
    """
    Load ground truth annotations for a specific image.
    
    Supports multiple annotation formats:
    - STCray annotations.json (array of annotations)
    - Single annotation file (dict for one image)
    - Custom test annotations
    
    Args:
        image_filename: Name of the image file
        annotations_file: Path to annotations JSON file
    
    Returns:
        Ground truth annotation dict or None if not found
    """
    if annotations_file is None:
        # Try STCray annotations first
        for split in ['train', 'test']:
            annotations_file = f"data/stcray_processed/{split}/annotations.json"
            if Path(annotations_file).exists():
                try:
                    with open(annotations_file) as f:
                        annotations = json.load(f)
                    
                    # Find annotation for this image
                    if isinstance(annotations, list):
                        for ann in annotations:
                            if ann.get('image_filename') == image_filename:
                                return ann
                    elif isinstance(annotations, dict) and annotations.get('image_filename') == image_filename:
                        return annotations
                except Exception as e:
                    print(f"Warning: Error loading annotations from {annotations_file}: {e}")
                    continue
        
        # Try test_xrays annotations
        test_annotations_dir = Path("data/test_xrays/annotations")
        if test_annotations_dir.exists():
            # Look for matching annotation file
            annotation_path = test_annotations_dir / f"{Path(image_filename).stem}_annotation.json"
            if annotation_path.exists():
                try:
                    with open(annotation_path) as f:
                        annotation = json.load(f)
                    if annotation.get('image_filename') == image_filename:
                        return annotation
                except Exception as e:
                    print(f"Warning: Error loading annotation from {annotation_path}: {e}")
    else:
        # Use specified annotations file
        if not Path(annotations_file).exists():
            print(f"Warning: Annotations file not found: {annotations_file}")
            return None
        
        try:
            with open(annotations_file) as f:
                data = json.load(f)
            
            # Handle both array and single dict formats
            if isinstance(data, list):
                for ann in data:
                    if ann.get('image_filename') == image_filename:
                        return ann
            elif isinstance(data, dict):
                if data.get('image_filename') == image_filename:
                    return data
                else:
                    print(f"Warning: Annotation file is for different image: {data.get('image_filename')}")
                    return None
        except Exception as e:
            print(f"Warning: Error loading annotations: {e}")
            return None
    
    return None


def compare_with_ground_truth(predicted: XrayDetectionResult, ground_truth: Dict) -> Dict:
    """
    Compare predicted results with ground truth annotations.
    
    Args:
        predicted: Predicted detection result from model
        ground_truth: Ground truth annotation from STCray
    
    Returns:
        Comparison metrics and analysis
    """
    gt_categories = set(ground_truth.get('categories', []))
    predicted_categories = set(item.category for item in predicted.detected_items)
    
    # Calculate metrics
    true_positives = gt_categories.intersection(predicted_categories)
    false_positives = predicted_categories - gt_categories
    false_negatives = gt_categories - predicted_categories
    
    precision = len(true_positives) / len(predicted_categories) if predicted_categories else 0
    recall = len(true_positives) / len(gt_categories) if gt_categories else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'ground_truth_categories': list(gt_categories),
        'predicted_categories': list(predicted_categories),
        'true_positives': list(true_positives),
        'false_positives': list(false_positives),
        'false_negatives': list(false_negatives),
        'ground_truth_count': len(gt_categories),
        'predicted_count': len(predicted_categories),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'exact_match': gt_categories == predicted_categories
    }


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_rolmocr(base_url: str, api_key: str, model_id: str, image_path: Path, 
                 temperature: float = 0.2, top_p: float = 0.7, max_tokens: int = 1024):
    """
    Test RolmOCR model using OpenAI client with image upload.
    
    Args:
        base_url: Base URL of the API server
        api_key: JWT token for authentication
        model_id: Model ID (e.g., "reducto/RolmOCR")
        image_path: Path to X-ray image file
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
    """
    print(f"\n{'='*70}")
    print(f"Testing RolmOCR Model")
    print(f"{'='*70}")
    print(f"API Base URL: {base_url}")
    print(f"Model ID: {model_id}")
    print(f"Image: {image_path}")
    print(f"Image size: {image_path.stat().st_size / 1024:.1f} KB")
    print(f"\nParameters:")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Max tokens: {max_tokens}")
    
    # Initialize OpenAI client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    
    # Encode image to base64
    print(f"\nEncoding image to base64...")
    base64_image = encode_image_to_base64(image_path)
    print(f"✓ Image encoded ({len(base64_image)} characters)")
    
    # Determine image MIME type
    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in ['.jpg', '.jpeg'] else "image/png"
    
    # Create structured prompt for X-ray analysis
    prompt = """Analyze this X-ray baggage scan image and identify all items present.

For each item you detect, provide:
- category: The general category (e.g., "Knife", "Gun", "Explosive", "Battery", "Scissors", "Blade", "Hammer", "Tool")
- item_name: A specific name for the item
- confidence: Your confidence level (0.0 to 1.0)
- location: Where the item is located in the scan (e.g., "center", "upper-left", "lower-right")

Focus on identifying prohibited or dangerous items, but also note other significant objects.
Provide a structured JSON response with all detected items."""
    
    print(f"\n{'='*70}")
    print(f"Sending request with structured output...")
    print(f"{'='*70}")
    print(f"Prompt: {prompt[:150]}...")
    print()
    
    try:
        # Get JSON schema from Pydantic model for vLLM guided decoding
        # vLLM uses extra_body with guided_json parameter
        json_schema = XrayDetectionResult.model_json_schema()
        
        print(f"JSON Schema for guided decoding:")
        print(json.dumps(json_schema, indent=2)[:500] + "...")
        print()
        
        # Create completion with vision and structured output (vLLM format)
        # vLLM v0.8.5 uses extra_body={"guided_json": schema} for structured outputs
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert X-ray security scanner analyst. Provide structured JSON output with detected items."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            # vLLM structured output via extra_body
            extra_body={"guided_json": json_schema}
        )
        
        print(f"\n{'='*70}")
        print(f"RolmOCR Response (Structured JSON):")
        print(f"{'='*70}\n")
        
        # Get response content (non-streaming for structured output)
        response_content = completion.choices[0].message.content
        print(response_content)
        
        # Parse JSON response
        try:
            result_dict = json.loads(response_content)
            result = XrayDetectionResult(**result_dict)
            
            print(f"\n{'='*70}")
            print(f"Parsed Detection Results")
            print(f"{'='*70}")
            print(f"Total items detected: {result.total_count}")
            print(f"Has threats: {result.has_threats}")
            
            if result.detected_items:
                print(f"\nDetected Items:")
                for i, item in enumerate(result.detected_items, 1):
                    print(f"\n  {i}. Category: {item.category}")
                    print(f"     Item: {item.item_name}")
                    if item.confidence is not None:
                        print(f"     Confidence: {item.confidence:.1%}")
                    if item.location:
                        print(f"     Location: {item.location}")
            
            if result.summary:
                print(f"\nSummary: {result.summary}")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"\n❌ Error parsing JSON response: {e}")
            print(f"   Raw response: {response_content}")
            return None
        
    except Exception as e:
        print(f"\n❌ Error calling RolmOCR API: {e}")
        print(f"   Make sure:")
        print(f"   1. JWT_TOKEN is set correctly")
        print(f"   2. API endpoint is accessible")
        print(f"   3. Model ID is correct")
        exit(1)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test RolmOCR model with random STCray X-ray image"
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default="https://ml-9132483a-8f3.gr-docpr.a465-9q4k.cloudera.site/namespaces/serving-default/endpoints/rolmocr/openai/v1",
        help='Base URL of the API server (default: Cloudera AI RolmOCR endpoint)'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default="reducto/RolmOCR",
        help='Model ID (default: reducto/RolmOCR)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/stcray_raw/STCray_TestSet/Images',
        help='Directory containing STCray images (default: data/stcray_raw/STCray_TestSet/Images)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Specific image path (if provided, ignores --image-dir and uses this image)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Sampling temperature (default: 0.2)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.7,
        help='Top-p sampling parameter (default: 0.7)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Maximum tokens to generate (default: 1024)'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default=None,
        help='Path to annotations JSON file (default: auto-detect from data/stcray_processed/)'
    )
    
    args = parser.parse_args()
    
    # Get JWT token from environment
    api_key = os.environ.get("JWT_TOKEN")
    if not api_key:
        print("❌ Error: JWT_TOKEN environment variable not set")
        print("  Set it with: export JWT_TOKEN=your-token-here")
        exit(1)
    
    print(f"\n{'='*70}")
    print(f"RolmOCR X-ray Image Test")
    print(f"{'='*70}")
    
    # Get image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"\n❌ Error: Image not found: {args.image}")
            exit(1)
        print(f"Using specified image: {image_path}")
    else:
        # Try primary image directory
        if not Path(args.image_dir).exists():
            # Fallback to test dataset
            fallback_dir = "data/yolo_dataset_test/images/val"
            if Path(fallback_dir).exists():
                print(f"⚠ Primary image directory not found: {args.image_dir}")
                print(f"  Using fallback: {fallback_dir}")
                args.image_dir = fallback_dir
            else:
                print(f"\n❌ Error: Image directory not found: {args.image_dir}")
                print(f"  Fallback directory also not found: {fallback_dir}")
                print(f"\n  Please specify a valid directory with --image-dir or a specific image with --image")
                exit(1)
        
        image_path = get_random_stcray_image(args.image_dir)
    
    # Test RolmOCR
    predicted_result = test_rolmocr(
        base_url=args.base_url,
        api_key=api_key,
        model_id=args.model_id,
        image_path=image_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    if not predicted_result:
        print(f"\n{'='*70}")
        print(f"✗ Test completed with errors")
        print(f"{'='*70}\n")
        exit(1)
    
    # Load and compare with ground truth
    print(f"\n{'='*70}")
    print(f"Ground Truth Comparison")
    print(f"{'='*70}")
    
    ground_truth = load_ground_truth(image_path.name, args.annotations)
    
    if ground_truth:
        print(f"\nGround Truth:")
        print(f"  Image: {ground_truth['image_filename']}")
        print(f"  Class: {ground_truth['class_name']}")
        print(f"  Categories: {', '.join(ground_truth['categories'])}")
        print(f"  Number of items: {ground_truth['num_annotations']}")
        print(f"  Bounding boxes: {len(ground_truth['bboxes'])}")
        
        # Compare results
        comparison = compare_with_ground_truth(predicted_result, ground_truth)
        
        print(f"\n{'='*70}")
        print(f"Performance Metrics")
        print(f"{'='*70}")
        print(f"Precision: {comparison['precision']:.1%}")
        print(f"Recall: {comparison['recall']:.1%}")
        print(f"F1 Score: {comparison['f1_score']:.3f}")
        print(f"Exact Match: {'✓ Yes' if comparison['exact_match'] else '✗ No'}")
        
        print(f"\nDetailed Comparison:")
        print(f"  True Positives: {comparison['true_positives']}")
        print(f"  False Positives: {comparison['false_positives']}")
        print(f"  False Negatives: {comparison['false_negatives']}")
        
        print(f"\n{'='*70}")
        print(f"Comparison Summary")
        print(f"{'='*70}")
        print(f"Ground Truth: {comparison['ground_truth_categories']}")
        print(f"Predicted:    {comparison['predicted_categories']}")
        
    else:
        print(f"\n⚠ Warning: Could not load ground truth annotations for {image_path.name}")
        print(f"  Comparison skipped.")
    
    print(f"\n{'='*70}")
    print(f"✓ Test completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
