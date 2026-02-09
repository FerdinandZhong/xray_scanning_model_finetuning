#!/usr/bin/env bash
set -euo pipefail

# Generate VQA dataset using text-only small LLM (cost-effective alternative)
# Uses ground truth annotations instead of vision-capable LLMs

echo "=========================================="
echo "Text-Based VQA Generation (Small LLM)"
echo "=========================================="
echo ""
echo "This uses a small text-only LLM (Qwen2.5-3B) running locally."
echo "Cost: FREE (local inference)"
echo "Time: ~1-2 hours for full dataset"
echo ""

# Configuration
MODEL="${MODEL:-qwen2.5-3b-instruct}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-3}"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠ Warning: nvidia-smi not found. Will use CPU (slower)"
else
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ Found $GPU_COUNT GPU(s)"
fi

# Activate environment
source .venv/bin/activate

echo ""
echo "Step 1: Generate training VQA dataset"
echo "======================================"
python data/llm_vqa_generator_text.py \
  --annotations data/stcray/train/annotations.json \
  --output data/stcray_vqa_train.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --batch-save 100

echo ""
echo "Step 2: Generate validation VQA dataset"
echo "========================================"
python data/llm_vqa_generator_text.py \
  --annotations data/stcray/test/annotations.json \
  --output data/stcray_vqa_val.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --batch-save 100

echo ""
echo "=========================================="
echo "VQA Generation Complete!"
echo "=========================================="
echo ""
echo "Files created:"
echo "  - data/stcray_vqa_train.jsonl (~90k pairs)"
echo "  - data/stcray_vqa_val.jsonl (~48k pairs)"
echo ""
echo "Cost: FREE (local inference)"
echo ""
echo "Next step: Start training"
echo "  python training/train_local.py --config configs/train_stcray.yaml"
