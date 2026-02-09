#!/usr/bin/env bash
set -euo pipefail

# Training script for X-ray VQA model
# Supports both Phase 1 (local) and Phase 2 (Ray) training

echo "=========================================="
echo "X-ray VQA Model Training"
echo "=========================================="

# Parse arguments
CONFIG="${1:-configs/train_stcray.yaml}"
RESUME_FROM="${2:-}"

# Activate environment
source .venv/bin/activate

# Check if dataset exists
TRAIN_FILE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train_file'])")
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    echo ""
    echo "Please generate VQA dataset first:"
    echo "  bash scripts/test_llm_generation.sh      # Test with 100 images"
    echo "  bash scripts/generate_full_vqa.sh        # Full generation"
    exit 1
fi

# Check GPU availability
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "Warning: No GPUs detected!"
    echo "Training will be very slow on CPU."
    read -p "Continue anyway? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        exit 1
    fi
else
    echo "Detected $GPU_COUNT GPU(s)"
fi

echo ""
echo "Training configuration:"
echo "  Config: $CONFIG"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resuming from: $RESUME_FROM"
fi
echo "  GPUs: $GPU_COUNT"
echo ""

# Start training
echo "Starting training..."
echo "Monitor with: tensorboard --logdir outputs/*/logs"
echo ""

if [ -n "$RESUME_FROM" ]; then
    python training/train_local.py \
      --config "$CONFIG" \
      --resume-from-checkpoint "$RESUME_FROM"
else
    python training/train_local.py \
      --config "$CONFIG"
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Evaluate: bash scripts/evaluate.sh"
echo "2. Deploy: bash scripts/deploy_inference.sh"
