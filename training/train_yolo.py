#!/usr/bin/env python3
"""
Train YOLO model for X-ray baggage detection.

Supports YOLOv8 and YOLOv11 with X-ray specific augmentations.
Optional ONNX export for production deployment.
"""

# Fix matplotlib backend for CAI/Jupyter environment
import os
if 'MPLBACKEND' in os.environ:
    # Replace Jupyter inline backend with non-interactive Agg backend
    os.environ['MPLBACKEND'] = 'Agg'

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml


def train_yolo(
    data_yaml: str,
    model_name: str = 'yolov8n.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = '0',
    project: str = 'runs/detect',
    name: str = 'xray_detection',
    patience: int = 10,  # Early stopping: terminate if no improvement for 10 epochs
    save_period: int = 10,
    export_onnx: bool = False,
    **kwargs
):
    """
    Train YOLO model on X-ray baggage dataset with early stopping.
    
    Early Stopping: Training automatically terminates if validation mAP50 
    doesn't improve for 'patience' consecutive epochs (default: 10).
    Best model checkpoint is saved before termination.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model_name: Pretrained model name or path (.pt file)
        epochs: Number of training epochs
        imgsz: Input image size (square)
        batch: Batch size
        device: Device to use ('0', 'cpu', or '0,1,2,3' for multi-GPU)
        project: Project directory for saving runs
        name: Run name
        patience: Early stopping patience (epochs)
        save_period: Save checkpoint every N epochs
        export_onnx: Export best model to ONNX after training
        **kwargs: Additional training arguments
    """
    
    print("="*70)
    print("YOLO X-RAY BAGGAGE DETECTION TRAINING")
    print("="*70)
    
    # Load pretrained model
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Print model info
    print(f"Model architecture: {model_name}")
    print(f"Device: {device}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Epochs: {epochs}")
    
    # Load data.yaml to show dataset info
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    print(f"\nDataset: {data_config['path']}")
    print(f"Classes: {data_config['nc']}")
    print(f"Train: {data_config['train']}")
    print(f"Val: {data_config['val']}")
    
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    # Train the model with X-ray specific augmentations
    # Configuration optimized for convergence (addresses mAP50=0.2 plateau issue)
    # Early stopping enabled: training terminates if no mAP50 improvement for 10 epochs
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,  # Early stopping: stop if no improvement for this many epochs
        save_period=save_period,
        
        # X-ray specific augmentations (REDUCED from aggressive values)
        # Previous aggressive augmentation may have confused the model
        degrees=10.0,       # Reduced from 15° (less rotation)
        translate=0.05,     # Reduced from 0.1 (less translation)
        scale=0.3,          # Reduced from 0.5 (less scaling)
        shear=3.0,          # Reduced from 5° (less shearing)
        perspective=0.0003, # Reduced from 0.0005 (less perspective)
        flipud=0.5,         # Keep (baggage orientation varies)
        fliplr=0.5,         # Keep (horizontal flip)
        
        # Mosaic and mixup - REDUCED to prevent overfitting
        mosaic=0.8,         # Reduced from 1.0 (less mosaic)
        mixup=0.0,          # Disabled from 0.1 (can confuse on small details)
        
        # Color augmentations - REDUCED (X-ray contrast is important)
        hsv_h=0.01,         # Reduced from 0.015 (less hue change)
        hsv_s=0.5,          # Reduced from 0.7 (less saturation change)
        hsv_v=0.3,          # Reduced from 0.4 (less brightness change)
        
        # IMPROVED: Optimizer settings for better convergence
        optimizer='AdamW',  # Changed from 'auto' - AdamW better for large models
        lr0=0.002,          # Reduced from 0.01 (was too high, causing poor convergence)
        lrf=0.001,          # Reduced from 0.01 (lower final LR for fine-tuning)
        momentum=0.95,      # Increased from 0.937 (more momentum for escaping local minima)
        weight_decay=0.0001,# Reduced from 0.0005 (less regularization)
        warmup_epochs=5.0,  # Increased from 3.0 (more gradual warmup)
        warmup_momentum=0.9,# Increased from 0.8 (smoother start)
        
        # Loss weights
        box=7.5,            # Box loss weight
        cls=0.5,            # Class loss weight
        dfl=1.5,            # DFL loss weight
        
        # Other settings
        plots=False,        # Disable plots to avoid matplotlib backend issues in CAI
        save=True,          # Save checkpoints
        val=True,           # Validate during training
        cache=False,        # Cache images (set to True if enough RAM)
        workers=8,          # Number of dataloader workers (requires sufficient shared memory)
        
        # Override with any additional kwargs
        **kwargs
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Print results
    best_weights = Path(project) / name / 'weights' / 'best.pt'
    last_weights = Path(project) / name / 'weights' / 'last.pt'
    
    print(f"\nBest weights: {best_weights}")
    print(f"Last weights: {last_weights}")
    
    # Print metrics
    if results.results_dict:
        print("\nFinal Metrics:")
        print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    
    # Export to ONNX if requested
    if export_onnx:
        print("\n" + "="*70)
        print("EXPORTING TO ONNX")
        print("="*70)
        export_to_onnx(str(best_weights), imgsz)
    
    print("\n" + "="*70)
    print(f"Training artifacts saved to: {Path(project) / name}")
    print("="*70)
    
    return results


def export_to_onnx(model_path: str, imgsz: int = 640, simplify: bool = True, dynamic: bool = False):
    """
    Export trained YOLO model to ONNX format.
    
    Args:
        model_path: Path to trained .pt model
        imgsz: Input image size
        simplify: Simplify ONNX model
        dynamic: Dynamic input shape (False for better optimization)
    """
    print(f"\nExporting model to ONNX...")
    print(f"  Model: {model_path}")
    print(f"  Image size: {imgsz}")
    print(f"  Simplify: {simplify}")
    print(f"  Dynamic: {dynamic}")
    
    # Load model
    model = YOLO(model_path)
    
    # Export to ONNX
    onnx_path = model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
        opset=12,  # ONNX opset version
    )
    
    print(f"\n✓ ONNX model exported to: {onnx_path}")
    
    # Test ONNX inference
    try:
        import onnxruntime as ort
        print("\nTesting ONNX inference...")
        
        session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"  Input: {input_name} {input_shape}")
        print(f"  Outputs: {output_names}")
        print("  ✓ ONNX model loaded successfully")
        
        # Test with dummy input
        import numpy as np
        dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        print(f"  ✓ Test inference successful (output shape: {outputs[0].shape})")
        
    except ImportError:
        print("\n  Note: onnxruntime not installed. Skipping ONNX validation.")
        print("  Install with: pip install onnxruntime-gpu")
    except Exception as e:
        print(f"\n  Warning: ONNX validation failed: {e}")
    
    return onnx_path


def validate_model(model_path: str, data_yaml: str, imgsz: int = 640, batch: int = 16, device: str = '0'):
    """Run validation on trained model."""
    print("\n" + "="*70)
    print("VALIDATING MODEL")
    print("="*70)
    
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )
    
    print("\nValidation Metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for X-ray baggage detection"
    )
    
    # Required arguments
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data.yaml configuration file'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Model name or path (yolov8n, yolov8s, yolov8m, yolov11n, etc.)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use: 0, cpu, or 0,1,2,3 for multi-GPU (default: 0)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience in epochs (default: 50)'
    )
    
    # Output arguments
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory (default: runs/detect)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='xray_detection',
        help='Run name (default: xray_detection)'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs (default: 10)'
    )
    
    # Export arguments
    parser.add_argument(
        '--export-onnx',
        action='store_true',
        help='Export best model to ONNX after training'
    )
    
    # Validation only mode
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation on existing model'
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Validation only mode
    if args.validate_only:
        if not Path(args.model).exists():
            print(f"Error: Model not found: {args.model}")
            return
        validate_model(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
        return
    
    # Train model
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        export_onnx=args.export_onnx,
    )


if __name__ == '__main__':
    main()
