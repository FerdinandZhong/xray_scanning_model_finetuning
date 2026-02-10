#!/usr/bin/env bash
set -euo pipefail

# Generate VQA dataset using Gemini 2.0 Flash via OpenAI-compatible endpoint
# CHEAP + Good Quality (vision-capable)

echo "=========================================="
echo "VQA Generation with Gemini 2.0 Flash"
echo "=========================================="
echo ""
echo "This uses Gemini 2.0 Flash via OpenAI-compatible AI Gateway."
echo "Cost: Very low (~\$0.0001-0.0003 per image)"
echo "Quality: Good (vision-capable model)"
echo "Time: ~1-2 hours for full dataset"
echo ""

# Check if running on macOS and warn about sleep
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Running on macOS detected"
    echo ""
    echo "⚠️  IMPORTANT: Prevent laptop sleep during generation"
    echo "   Option 1 (RECOMMENDED): This script will use 'caffeinate' to prevent sleep"
    echo "   Option 2: Manually: System Settings → Energy Saver → Prevent sleep"
    echo "   Option 3: Use 'caffeinate -i' in terminal before running this script"
    echo ""
    USE_CAFFEINATE="${USE_CAFFEINATE:-yes}"
    if [ "$USE_CAFFEINATE" = "yes" ]; then
        echo "✓ Will use 'caffeinate' to prevent sleep during generation"
        CAFFEINATE_CMD="caffeinate -i"
    else
        echo "⚠️ Sleep prevention disabled (set USE_CAFFEINATE=yes to enable)"
        CAFFEINATE_CMD=""
    fi
    echo ""
else
    CAFFEINATE_CMD=""
fi

# Configuration
MODEL="${MODEL:-gemini-2.0-flash}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"              # Sample 1000 random images
STRUCTURED_RATIO="${STRUCTURED_RATIO:-0.3}"      # 30% JSON, 70% natural language
RANDOM_SEED="${RANDOM_SEED:-42}"                 # For reproducible sampling
API_BASE="${API_BASE:-https://ai-gateway.dev.cloudops.cloudera.com/v1}"

# Calculate samples per image based on structured_ratio
# Need at least 5 samples to get meaningful structured/natural mix
if (( $(echo "$STRUCTURED_RATIO > 0" | bc -l) )); then
    SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-5}"  # Increase to 5 for mixed format
else
    SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-3}"  # Default 3 for natural only
fi

# Dataset paths - using processed STCray data
ANNOTATIONS_FILE="data/stcray_processed/train/annotations.json"
IMAGES_DIR="data/stcray_raw/STCray_TrainSet/Images"
OUTPUT_FILE="data/stcray_vqa_1k_mixed.jsonl"

# Check if API key is set
if [ -z "${API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "✗ Error: API_KEY environment variable not set"
    echo ""
    echo "Please set your API key:"
    echo "  export API_KEY='your-api-key'"
    echo ""
    echo "Or use OPENAI_API_KEY if you prefer:"
    echo "  export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Check if dataset exists
if [ ! -f "$ANNOTATIONS_FILE" ]; then
    echo "✗ Error: STCray annotations not found at: $ANNOTATIONS_FILE"
    echo ""
    echo "Please process the extracted dataset first:"
    echo "  python data/process_stcray_extracted.py \\"
    echo "    --input-dir data/stcray_raw/STCray_TrainSet \\"
    echo "    --output-dir data/stcray_processed/train"
    echo ""
    echo "If dataset is not extracted yet:"
    echo "  1. Download: ./scripts/download_stcray_rar.sh"
    echo "  2. Extract: cd data/stcray_raw && unar STCray_TrainSet.rar"
    echo "  3. Process: Run command above"
    exit 1
fi

echo "✓ API key configured"
echo "✓ Dataset found: $ANNOTATIONS_FILE"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  API Base: $API_BASE"
echo "  Samples per image: $SAMPLES_PER_IMAGE"
echo "  Random sampling: $NUM_SAMPLES images"
echo "  Structured ratio: ${STRUCTURED_RATIO} (~$((${STRUCTURED_RATIO%.*}*10))% JSON, ~$((100-${STRUCTURED_RATIO%.*}*10))% natural)"
echo "  Random seed: $RANDOM_SEED"
echo "  Output: $OUTPUT_FILE"
echo "  OpenAI-compatible: Yes"
echo ""
if (( $(echo "$STRUCTURED_RATIO > 0" | bc -l) )); then
    echo "ℹ️  Note: Using $SAMPLES_PER_IMAGE samples/image to ensure structured questions are generated"
    echo "   With 3 samples/image, 30% = 0.9 → rounds to 0 (no structured questions)"
    echo "   With 5 samples/image, 30% = 1.5 → includes structured questions"
    echo ""
fi

# Estimate cost (very rough, for 1000 samples)
COST_ESTIMATE=$(echo "scale=2; $NUM_SAMPLES * 0.0002 * $SAMPLES_PER_IMAGE" | bc)

echo "Estimated Cost:"
echo "  $NUM_SAMPLES images × $SAMPLES_PER_IMAGE samples × \$0.0002 ≈ \$$COST_ESTIMATE"
echo "  (Very cheap with Gemini Flash!)"
echo ""

# Confirm with user
read -p "Continue with VQA generation? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted by user"
    exit 0
fi

# Set Python command (use venv_vqa if available)
if [ -x "venv_vqa/bin/python" ]; then
    PYTHON_CMD="venv_vqa/bin/python"
    echo "✓ Using venv_vqa Python"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_CMD="python"
    echo "✓ Virtual environment activated"
else
    PYTHON_CMD="python"
    echo "⚠ Warning: No venv found, using system Python"
fi

echo ""
echo "Step 1: Test with 10 images"
echo "============================"

# Set environment for OpenAI-compatible endpoint
export OPENAI_API_BASE="$API_BASE"
export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"

$CAFFEINATE_CMD $PYTHON_CMD data/llm_vqa_generator.py \
  --annotations "$ANNOTATIONS_FILE" \
  --images-dir "$IMAGES_DIR" \
  --output data/stcray_vqa_test_10.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --max-images 10 \
  --structured-ratio "$STRUCTURED_RATIO" \
  --api-base "$API_BASE"

echo ""
echo "Test complete! Review output:"
echo "  head -3 data/stcray_vqa_test_10.jsonl | $PYTHON_CMD -m json.tool"
echo ""
read -p "Quality looks good? Continue with 1000-sample generation? (yes/no): " confirm_full
if [ "$confirm_full" != "yes" ]; then
    echo "Stopped. Adjust settings and try again."
    exit 0
fi

echo ""
echo "Step 2: Generate 1000-sample VQA dataset (mixed format)"
echo "========================================================"
echo "  • Sampling: $NUM_SAMPLES random images"
echo "  • Format: $((100-${STRUCTURED_RATIO%.*}0))% natural language + ${STRUCTURED_RATIO}0% structured JSON"
echo "  • Random seed: $RANDOM_SEED (reproducible)"
if [ -n "$CAFFEINATE_CMD" ]; then
    echo "  • 🔋 Keeping MacBook awake during generation..."
fi
echo ""

$CAFFEINATE_CMD $PYTHON_CMD data/llm_vqa_generator.py \
  --annotations "$ANNOTATIONS_FILE" \
  --images-dir "$IMAGES_DIR" \
  --output "$OUTPUT_FILE" \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --num-samples "$NUM_SAMPLES" \
  --structured-ratio "$STRUCTURED_RATIO" \
  --random-seed "$RANDOM_SEED" \
  --api-base "$API_BASE" \
  --rate-limit-delay 0.2 \
  --batch-save 100

echo ""
echo "=========================================="
echo "VQA Generation Complete!"
echo "=========================================="
echo ""
echo "Generated dataset:"
echo "  File: $OUTPUT_FILE"
echo "  Samples: $NUM_SAMPLES images × $SAMPLES_PER_IMAGE questions = ~$((NUM_SAMPLES * SAMPLES_PER_IMAGE)) VQA pairs"
echo "  Format: $((100-${STRUCTURED_RATIO%.*}0))% natural + ${STRUCTURED_RATIO}0% JSON"
echo ""
echo "Cost: ~\$$COST_ESTIMATE (very cheap with Gemini Flash)"
echo ""
echo "Next steps:"
echo "  1. Split into train/val:"
echo "     python data/split_vqa_dataset.py \\"
echo "       --input $OUTPUT_FILE \\"
echo "       --output-dir data \\"
echo "       --output-prefix stcray_vqa_1k \\"
echo "       --train-ratio 0.8 \\"
echo "       --val-ratio 0.2"
echo ""
echo "  2. Start training:"
echo "     python training/train_qwen_vl.py \\"
echo "       --train_file data/stcray_vqa_1k_train.jsonl \\"
echo "       --eval_file data/stcray_vqa_1k_val.jsonl \\"
echo "       --output_dir models/qwen2.5-vl-7b-xray-1k"
echo ""
