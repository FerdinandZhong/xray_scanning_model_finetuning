#!/usr/bin/env python3
"""
Test YOLO inference on sample X-ray images.

Validates model predictions and JSON output format.
"""

import argparse
import json
from pathlib import Path
from typing import List
import sys

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


def test_inference(
    model_path: str,
    image_paths: List[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    save_dir: str = 'test_results',
    show_plots: bool = True
):
    """
    Test YOLO inference on images.
    
    Args:
        model_path: Path to trained model (.pt)
        image_paths: List of image paths to test
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold for NMS
        save_dir: Directory to save results
        show_plots: Whether to display plots
    """
    print("="*70)
    print("YOLO INFERENCE TEST")
    print("="*70)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded successfully")
    print(f"Classes: {len(model.names)}")
    print(f"Class names: {list(model.names.values())[:5]}... ({len(model.names)} total)")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each image
    results_summary = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n{'-'*70}")
        print(f"Testing image {i+1}/{len(image_paths)}: {image_path}")
        print(f"{'-'*70}")
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"  ✗ Image not found: {image_path}")
            continue
        
        # Load image
        image = Image.open(image_path)
        print(f"  Image size: {image.size}")
        
        # Run inference
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Extract detections
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox_normalized = box.xywhn[0].tolist()  # [x_center, y_center, w, h]
                bbox_pixels = box.xywh[0].tolist()  # [x_center, y_center, w, h] in pixels
                
                detection = {
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": confidence,
                    "bbox_normalized": bbox_normalized,
                    "bbox_pixels": bbox_pixels,
                    "location": bbox_to_location(bbox_normalized)
                }
                detections.append(detection)
                
                print(f"  ✓ Detected: {detection['class_name']} "
                      f"(conf: {detection['confidence']:.3f}, "
                      f"loc: {detection['location']})")
        
        print(f"\n  Total detections: {len(detections)}")
        
        # Save results
        result_summary = {
            "image_path": str(image_path),
            "image_size": image.size,
            "num_detections": len(detections),
            "detections": detections
        }
        results_summary.append(result_summary)
        
        # Save JSON
        json_path = save_dir / f"result_{i+1}.json"
        with open(json_path, 'w') as f:
            json.dump(result_summary, f, indent=2)
        print(f"  Saved JSON: {json_path}")
        
        # Visualize and save
        if detections:
            fig = visualize_detections(image, detections, model.names)
            plot_path = save_dir / f"result_{i+1}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization: {plot_path}")
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        else:
            print(f"  No detections to visualize")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_detections = sum(r['num_detections'] for r in results_summary)
    avg_detections = total_detections / len(results_summary) if results_summary else 0
    
    print(f"\nTested images: {len(results_summary)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    
    # Print detection distribution
    if total_detections > 0:
        class_counts = {}
        for result in results_summary:
            for det in result['detections']:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nDetection distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
    
    # Save summary
    summary_path = save_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "model": model_path,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "num_images": len(results_summary),
            "total_detections": total_detections,
            "avg_detections": avg_detections,
            "results": results_summary
        }, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")
    print(f"All results saved to: {save_dir}")
    print("="*70)


def bbox_to_location(bbox_normalized: List[float]) -> str:
    """Convert normalized bbox to location string."""
    x_center, y_center, _, _ = bbox_normalized
    
    LEFT_THRESHOLD = 0.33
    RIGHT_THRESHOLD = 0.67
    UPPER_THRESHOLD = 0.33
    LOWER_THRESHOLD = 0.67
    
    if x_center < LEFT_THRESHOLD:
        h_pos = "left"
    elif x_center > RIGHT_THRESHOLD:
        h_pos = "right"
    else:
        h_pos = "center"
    
    if y_center < UPPER_THRESHOLD:
        v_pos = "upper"
    elif y_center > LOWER_THRESHOLD:
        v_pos = "lower"
    else:
        v_pos = "center"
    
    if v_pos == "center" and h_pos == "center":
        return "center"
    elif v_pos == "center":
        return h_pos
    elif h_pos == "center":
        return v_pos
    else:
        return f"{v_pos}-{h_pos}"


def visualize_detections(image: Image.Image, detections: List[dict], class_names: dict) -> plt.Figure:
    """Visualize detections on image."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    img_width, img_height = image.size
    
    # Draw bounding boxes
    for det in detections:
        bbox_pixels = det['bbox_pixels']
        x_center, y_center, width, height = bbox_pixels
        
        # Convert center to top-left corner
        x = x_center - width / 2
        y = y_center - height / 2
        
        # Create rectangle
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{det['class_name']}\n{det['confidence']:.2f}"
        ax.text(
            x, y - 5,
            label,
            color='white',
            fontsize=10,
            bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.3')
        )
    
    ax.axis('off')
    plt.title(f"Detections: {len(detections)}", fontsize=14, pad=10)
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO inference on X-ray images"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pt)'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to test images'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.45,
        help='IOU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='test_results',
        help='Directory to save results (default: test_results)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots'
    )
    
    args = parser.parse_args()
    
    test_inference(
        model_path=args.model,
        image_paths=args.images,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        save_dir=args.save_dir,
        show_plots=not args.no_show
    )


if __name__ == '__main__':
    main()
