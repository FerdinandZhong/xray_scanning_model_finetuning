#!/usr/bin/env python3
"""
YOLO X-ray Detection Training Job for CAI.

This script trains a YOLO model for X-ray baggage threat detection.
Supports both STCray and CargoXray datasets.

Environment Variables:
- DATASET: Dataset to use (cargoxray or stcray, default: stcray)
- MODEL_NAME: YOLO model name (default: yolov8n.pt)
- EPOCHS: Number of training epochs (default: 100)
- BATCH_SIZE: Batch size (default: 16)
- IMG_SIZE: Input image size (default: 640)
- EXPORT_ONNX: Export to ONNX after training (default: false)
- VAL_SPLIT: Validation split ratio (default: 0.2, only used for stcray)
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Execute YOLO model training."""
    print("=" * 60)
    print("YOLO X-ray Detection Training Job")
    print("=" * 60)
    
    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"
    
    # Verify venv exists
    if not venv_python.exists():
        print(f"❌ Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)
    
    # Get configuration from environment
    dataset = os.getenv("DATASET", "stcray").lower()
    model_name = os.getenv("MODEL_NAME", "yolov8n.pt")
    epochs = int(os.getenv("EPOCHS", "100"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    img_size = int(os.getenv("IMG_SIZE", "640"))
    export_onnx = os.getenv("EXPORT_ONNX", "false").lower() == "true"
    val_split = float(os.getenv("VAL_SPLIT", "0.2"))
    
    print(f"✓ Using Python: {venv_python}")
    print(f"✓ Working directory: {project_root}")
    print()
    print(f"YOLO Training Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    if dataset == "stcray":
        print(f"  Validation Split: {val_split}")
    print(f"  Export ONNX: {export_onnx}")
    print()
    
    # Verify dataset exists based on type
    if dataset == "cargoxray":
        # CargoXray is already in YOLO format from Git LFS
        data_yaml = project_root / "data/cargoxray_yolo/data.yaml"
        if not data_yaml.exists():
            print(f"❌ Error: CargoXray data.yaml not found at {data_yaml}")
            print("   Ensure Git LFS files are checked out properly")
            sys.exit(1)
        print(f"✓ CargoXray YOLO data verified: {data_yaml}")
        print("  (Using pre-converted YOLO format from Git LFS)")
        print()
    else:  # stcray
        # STCray needs to be converted to YOLO format
        train_annotations = project_root / "data/stcray_processed/train/annotations.json"
        test_annotations = project_root / "data/stcray_processed/test/annotations.json"
        
        if not train_annotations.exists():
            print(f"❌ Error: Training annotations not found at {train_annotations}")
            print("   Ensure download_dataset job has completed successfully")
            sys.exit(1)
        
        if not test_annotations.exists():
            print(f"❌ Error: Test annotations not found at {test_annotations}")
            print("   Ensure download_dataset job has completed successfully")
            sys.exit(1)
        
        print(f"✓ Training annotations verified: {train_annotations}")
        print(f"✓ Test annotations verified: {test_annotations}")
        print()
    
    # Check GPU availability
    try:
        gpu_check = subprocess.run(
            [str(venv_python), "-c", "import torch; print(f'GPUs: {torch.cuda.device_count()}')"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        print(gpu_check.stdout.strip())
    except Exception as e:
        print(f"⚠ Warning: Could not check GPU availability: {e}")
    
    print()
    
    # Determine data.yaml path based on dataset
    if dataset == "cargoxray":
        data_yaml_path = "data/cargoxray_yolo/data.yaml"
        print("=" * 60)
        print("Using Pre-Converted CargoXray Data")
        print("=" * 60)
        print(f"✓ Data YAML: {data_yaml_path}")
        print("  (No conversion needed - already in YOLO format)")
        print()
    else:  # stcray
        print("=" * 60)
        print("Step 1: Convert STCray to YOLO Format")
        print("=" * 60)
        print()
        
        # Convert data to YOLO format
        convert_cmd = [
            str(venv_python),
            "data/convert_to_yolo_format.py",
            "--annotations-dir", "data/stcray_processed",
            "--output-dir", "data/yolo_dataset",
            "--val-split", str(val_split),
        ]
        
        print(f"Running: {' '.join(convert_cmd)}")
        print()
        
        result = subprocess.run(
            convert_cmd,
            cwd=str(project_root),
            env=os.environ.copy()
        )
        
        if result.returncode != 0:
            print()
            print("❌ Data conversion failed!")
            sys.exit(1)
        
        print()
        print("✓ Data conversion completed")
        
        # Verify YOLO dataset was created
        data_yaml = project_root / "data/yolo_dataset/data.yaml"
        if not data_yaml.exists():
            print(f"❌ Error: data.yaml not found at {data_yaml}")
            sys.exit(1)
        
        print(f"✓ YOLO dataset created: {data_yaml}")
        print()
        
        data_yaml_path = "data/yolo_dataset/data.yaml"
    
    print("=" * 60)
    print("Step 2: Train YOLO Model")
    print("=" * 60)
    print(f"This may take {epochs * 2 // 60}-{epochs * 5 // 60} hours depending on GPU configuration...")
    print("Checkpoints will be saved periodically")
    print()
    
    # Build training command
    train_cmd = [
        str(venv_python),
        "training/train_yolo.py",
        "--data", data_yaml_path,
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--imgsz", str(img_size),
        "--device", "0",  # Use first GPU
        "--project", "runs/detect",
        "--name", f"xray_detection_{dataset}",
    ]
    
    if export_onnx:
        train_cmd.append("--export-onnx")
    
    print(f"Running: {' '.join(train_cmd)}")
    print()
    
    result = subprocess.run(
        train_cmd,
        cwd=str(project_root),
        env=os.environ.copy()
    )
    
    if result.returncode != 0:
        print()
        print("❌ Training failed!")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✓ YOLO Training Completed Successfully!")
    print("=" * 60)
    
    # Print results location
    weights_dir = project_root / f"runs/detect/xray_detection_{dataset}/weights"
    best_weights = weights_dir / "best.pt"
    
    if best_weights.exists():
        print(f"\nBest model weights: {best_weights}")
        
        # Print model size
        size_mb = best_weights.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.1f} MB")
        
        if export_onnx:
            onnx_path = best_weights.parent / "best.onnx"
            if onnx_path.exists():
                onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
                print(f"ONNX model: {onnx_path} ({onnx_size_mb:.1f} MB)")
    
    print()
    print("Next steps:")
    print("  1. Test the model locally with scripts/test_yolo_inference.py")
    print("  2. Deploy API server with scripts/serve_yolo_api.sh")
    print("  3. Integrate with your agentic workflow via OpenAI-compatible API")
    print()


if __name__ == "__main__":
    main()
