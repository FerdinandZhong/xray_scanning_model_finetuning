#!/usr/bin/env bash
set -euo pipefail

# Generate VQA dataset using local Qwen2.5-VL model
# FREE (no API costs) + High Quality (vision-capable)

echo "=========================================="
echo "VQA Generation with Qwen2.5-VL (Local)"
echo "=========================================="
echo ""
echo "This uses Qwen2.5-VL vision model running locally."
echo "Cost: FREE (no API costs)"
echo "Quality: High (actually sees images)"
echo "Time: ~2-4 hours for full dataset"
echo ""

# Configuration
MODEL="${MODEL:-qwen2.5-vl-7b}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-3}"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ Error: nvidia-smi not found. GPU required for Qwen2.5-VL"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $GPU_COUNT GPU(s)"

# Check VRAM
if [[ "$MODEL" == *"7b"* ]]; then
    echo "Model: Qwen2.5-VL-7B (requires 16-24GB VRAM)"
elif [[ "$MODEL" == *"2b"* ]]; then
    echo "Model: Qwen2.5-VL-2B (requires 8-12GB VRAM)"
else
    echo "Model: $MODEL"
fi

# Confirm with user
echo ""
read -p "Continue with local VQA generation? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted by user"
    exit 0
fi

# Activate environment
source .venv/bin/activate

echo ""
echo "Step 1: Test with 10 images"
echo "============================"
python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_test.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --max-images 10 \
  --use-local

echo ""
echo "Test complete! Review output:"
echo "  head -3 data/stcray_vqa_test.jsonl | python -m json.tool"
echo ""
read -p "Quality looks good? Continue with full generation? (yes/no): " confirm_full
if [ "$confirm_full" != "yes" ]; then
    echo "Stopped. Adjust settings and try again."
    exit 0
fi

echo ""
echo "Step 2: Generate training VQA dataset"
echo "======================================"
python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --use-local \
  --batch-save 100

echo ""
echo "Step 3: Generate validation VQA dataset"
echo "========================================"
python data/llm_vqa_generator.py \
  --annotations data/stcray/test/annotations.json \
  --images-dir data/stcray/test/images \
  --output data/stcray_vqa_val.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --use-local \
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
echo "Quality: High (vision-capable model)"
echo ""
echo "Next step: Start training"
echo "  python training/train_local.py --config configs/train_stcray.yaml"
