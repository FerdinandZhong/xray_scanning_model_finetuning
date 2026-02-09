#!/usr/bin/env bash
set -euo pipefail

# Generate VQA dataset using Gemini 2.0 Flash via Cloudera AI Gateway
# CHEAP + Good Quality (vision-capable)

echo "=========================================="
echo "VQA Generation with Gemini 2.0 Flash"
echo "=========================================="
echo ""
echo "This uses Gemini 2.0 Flash via Cloudera AI Gateway."
echo "Cost: Very low (~\$0.0001-0.0003 per image)"
echo "Quality: Good (vision-capable model)"
echo "Time: ~1-2 hours for full dataset"
echo ""

# Configuration
MODEL="${MODEL:-gemini-2.0-flash-exp}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-3}"
API_BASE="${API_BASE:-https://ai-gateway.dev.cloudops.cloudera.com}"

# Check if API key is set
if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "✗ Error: GOOGLE_API_KEY environment variable not set"
    echo ""
    echo "Please set your API key:"
    echo "  export GOOGLE_API_KEY='your-api-key'"
    exit 1
fi

# Check if dataset exists
if [ ! -f "data/stcray/train/annotations.json" ]; then
    echo "✗ Error: STCray dataset not found"
    echo ""
    echo "Please download the dataset first:"
    echo "  python data/download_stcray.py --output-dir data/stcray"
    exit 1
fi

echo "✓ API key configured"
echo "✓ Dataset found"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  API Base: $API_BASE"
echo "  Samples per image: $SAMPLES_PER_IMAGE"
echo ""

# Estimate cost (very rough)
TRAIN_COST=$(echo "scale=2; 30000 * 0.0002" | bc)
VAL_COST=$(echo "scale=2; 16000 * 0.0002" | bc)
TOTAL_COST=$(echo "scale=2; $TRAIN_COST + $VAL_COST" | bc)

echo "Estimated Cost:"
echo "  Training set (~30k images): \$$TRAIN_COST"
echo "  Validation set (~16k images): \$$VAL_COST"
echo "  Total estimated: \$$TOTAL_COST"
echo ""

# Confirm with user
read -p "Continue with VQA generation? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted by user"
    exit 0
fi

# Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: No .venv found, using system Python"
fi

echo ""
echo "Step 1: Test with 10 images"
echo "============================"
python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_gemini_test.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --max-images 10 \
  --api-base "$API_BASE"

echo ""
echo "Test complete! Review output:"
echo "  head -3 data/stcray_vqa_gemini_test.jsonl | python -m json.tool"
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
  --api-base "$API_BASE" \
  --rate-limit-delay 0.2 \
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
  --api-base "$API_BASE" \
  --rate-limit-delay 0.2 \
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
echo "Cost: ~\$$TOTAL_COST (very cheap with Gemini Flash)"
echo "Quality: Good (vision-capable model)"
echo ""
echo "Next step: Start training"
echo "  python training/train_local.py --config configs/train_stcray.yaml"
