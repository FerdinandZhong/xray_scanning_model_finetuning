#!/usr/bin/env python3
"""
Download pre-trained YOLO model for benchmarking.

This script downloads a pre-trained YOLOv8 model that can be used
for initial benchmarking and testing the deployment system.

For X-ray detection, you should train on X-ray datasets, but this
provides a quick way to test the infrastructure.
"""

import argparse
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def download_pretrained_model(model_name: str = 'yolov8n.pt', output_dir: str = 'models/pretrained'):
    """
    Download pre-trained YOLO model from Ultralytics.
    
    Args:
        model_name: Model variant (yolov8n.pt, yolov8s.pt, yolov8m.pt)
        output_dir: Directory to save model
    """
    print(f"📥 Downloading pre-trained model: {model_name}")
    print(f"This model is trained on COCO dataset (80 classes)")
    print(f"For X-ray detection, you should train on X-ray datasets")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download model (Ultralytics will download automatically)
    print(f"Downloading {model_name}...")
    model = YOLO(model_name)
    
    # Save to output directory
    model_file = output_path / model_name
    print(f"✅ Model ready at: {model_file}")
    print()
    
    # Print model info
    print("Model Information:")
    print(f"  Model: {model_name}")
    print(f"  Classes: 80 (COCO dataset)")
    print(f"  Architecture: YOLOv8")
    print()
    
    print("⚠️  Important Notes:")
    print("  - This model is NOT trained on X-ray images")
    print("  - It won't detect X-ray specific threats accurately")
    print("  - Use this only for testing deployment infrastructure")
    print("  - For production, train on X-ray datasets (luggage_xray, stcray)")
    print()
    
    print("Next Steps:")
    print(f"  1. Deploy this model for testing:")
    print(f"     python cai_integration/deploy_yolo_application.py \\")
    print(f"       --model {model_name}")
    print()
    print(f"  2. Train on X-ray dataset:")
    print(f"     python training/train_yolo.py \\")
    print(f"       --data data/luggage_xray_yolo/data.yaml")
    
    return model_file


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained YOLO model for benchmarking"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='Model variant to download (default: yolov8n.pt - fastest)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/pretrained',
        help='Output directory (default: models/pretrained)'
    )
    
    args = parser.parse_args()
    
    download_pretrained_model(args.model, args.output_dir)


if __name__ == '__main__':
    main()
