#!/usr/bin/env python3
"""
YOLO Training on CargoXray for CAI.

Quick baseline training on CargoXray dataset (659 images, 16 categories).
Trains in ~30 minutes on 1x GPU.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Execute YOLO training on CargoXray."""
    print("=" * 60)
    print("YOLO CargoXray Training Job")
    print("=" * 60)
    
    project_root = Path("/home/cdsw")
    venv_python = project_root / ".venv/bin/python"
    
    # Verify venv exists
    if not venv_python.exists():
        print(f"❌ Error: Virtual environment not found at {venv_python}")
        print("   Ensure setup_environment job has completed successfully")
        sys.exit(1)
    
    # Get configuration from environment
    model_name = os.getenv("MODEL_NAME", "yolov8n.pt")
    epochs = int(os.getenv("EPOCHS", "100"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    img_size = int(os.getenv("IMG_SIZE", "640"))
    data_yaml = os.getenv("DATA_YAML", "data/cargoxray_yolo/data.yaml")
    export_onnx = os.getenv("EXPORT_ONNX", "false").lower() == "true"
    
    print(f"✓ Using Python: {venv_python}")
    print(f"✓ Working directory: {project_root}")
    print()
    print(f"YOLO Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Export ONNX: {export_onnx}")
    print()
    
    # Verify dataset exists
    data_yaml_path = project_root / data_yaml
    if not data_yaml_path.exists():
        print(f"❌ Error: Dataset config not found: {data_yaml_path}")
        print("   Ensure CargoXray dataset has been uploaded")
        sys.exit(1)
    
    print(f"✓ Dataset config found: {data_yaml_path}")
    print()
    
    # Build training command
    train_script = project_root / "training" / "train_yolo.py"
    
    cmd = [
        str(venv_python),
        str(train_script),
        "--data", str(data_yaml_path),
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--imgsz", str(img_size),
        "--device", "0",  # Use first GPU
        "--name", "cargoxray_v1",
        "--project", str(project_root / "runs" / "detect")
    ]
    
    if export_onnx:
        cmd.append("--export-onnx")
    
    print("=" * 60)
    print("Starting YOLO Training")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            env=os.environ.copy()
        )
        
        print()
        print("=" * 60)
        print("✅ YOLO Training Completed Successfully!")
        print("=" * 60)
        print()
        print("Results saved to:")
        print(f"  Model: runs/detect/cargoxray_v1/weights/best.pt")
        print(f"  Metrics: runs/detect/cargoxray_v1/results.csv")
        print(f"  Plots: runs/detect/cargoxray_v1/*.png")
        
        if export_onnx:
            print(f"  ONNX: runs/detect/cargoxray_v1/weights/best.onnx")
        
        print()
        print("Next steps:")
        print("1. Download model: scp cai:/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt .")
        print("2. Test locally: python inference/yolo_api_server.py --model best.pt")
        print("3. Deploy to CAI Application for production")
        print()
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"❌ YOLO Training Failed (exit code {e.returncode})")
        print("=" * 60)
        print()
        print("Check logs above for error details")
        return e.returncode
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
