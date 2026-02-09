#!/usr/bin/env bash
set -euo pipefail

# Generate complete VQA dataset with LLM
# WARNING: This will incur API costs (~$300-2,800 depending on model)

echo "=========================================="
echo "Full VQA Dataset Generation with LLM"
echo "=========================================="

# Configuration
MODEL="${MODEL:-claude-3-5-sonnet-20241022}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-3}"

# Estimate cost
echo ""
echo "Cost Estimation:"
if [[ "$MODEL" == *"opus"* ]]; then
    COST_PER_IMAGE=0.03
elif [[ "$MODEL" == *"sonnet"* ]]; then
    COST_PER_IMAGE=0.02
elif [[ "$MODEL" == "gpt-4o" ]]; then
    COST_PER_IMAGE=0.02
elif [[ "$MODEL" == "gpt-4o-mini" ]]; then
    COST_PER_IMAGE=0.002
else
    COST_PER_IMAGE=0.02
fi

# Calculate for ~30k train + 16k val images
TRAIN_COST=$(echo "scale=2; 30000 * $COST_PER_IMAGE" | bc)
VAL_COST=$(echo "scale=2; 16000 * $COST_PER_IMAGE" | bc)
TOTAL_COST=$(echo "scale=2; $TRAIN_COST + $VAL_COST" | bc)

echo "  Model: $MODEL"
echo "  Cost per image: \$$COST_PER_IMAGE"
echo "  Training set (~30k): \$$TRAIN_COST"
echo "  Validation set (~16k): \$$VAL_COST"
echo "  Total estimated: \$$TOTAL_COST"
echo ""

# Confirm with user
read -p "Continue with full generation? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted by user"
    exit 0
fi

# Activate environment
source .venv/bin/activate

echo ""
echo "Step 1: Generate training VQA dataset"
echo "======================================"
echo "This may take several hours..."
echo ""

python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --rate-limit-delay 1.0 \
  --batch-save 100

echo ""
echo "Step 2: Generate validation VQA dataset"
echo "========================================"
echo ""

python data/llm_vqa_generator.py \
  --annotations data/stcray/test/annotations.json \
  --images-dir data/stcray/test/images \
  --output data/stcray_vqa_val.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --rate-limit-delay 1.0 \
  --batch-save 100

echo ""
echo "=========================================="
echo "VQA Generation Complete!"
echo "=========================================="
echo ""
echo "Files created:"
echo "  - data/stcray_vqa_train.jsonl"
echo "  - data/stcray_vqa_val.jsonl"
echo ""
echo "Next step: Start training"
echo "  python training/train_local.py --config configs/train_stcray.yaml"
