#!/usr/bin/env bash
set -euo pipefail

# Test LLM VQA generation with small sample
# This script helps validate LLM output before full generation

echo "=========================================="
echo "Test LLM VQA Generation"
echo "=========================================="

# Configuration
MODEL="${MODEL:-claude-3-5-sonnet-20241022}"
MAX_IMAGES="${MAX_IMAGES:-100}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-3}"

# Check API key
if [[ "$MODEL" == claude* ]] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    echo "Set it with: export ANTHROPIC_API_KEY=your_key"
    exit 1
elif [[ "$MODEL" == gpt* ]] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Set it with: export OPENAI_API_KEY=your_key"
    exit 1
fi

# Check dataset exists
if [ ! -f "data/stcray/train/annotations.json" ]; then
    echo "Error: Dataset not found"
    echo "Run: python data/download_stcray.py"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Max images: $MAX_IMAGES"
echo "  Samples per image: $SAMPLES_PER_IMAGE"
echo "  Estimated cost: \$$(echo "scale=2; $MAX_IMAGES * $SAMPLES_PER_IMAGE * 0.02" | bc)"

# Activate environment
source .venv/bin/activate

echo ""
echo "Step 1: Generate test VQA pairs"
echo "================================"
python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train_test.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --max-images "$MAX_IMAGES"

echo ""
echo "Step 2: Validate quality"
echo "========================"
python data/llm_vqa_generator.py \
  --output data/stcray_vqa_train_test.jsonl \
  --validate

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Review the output in: data/stcray_vqa_train_test.jsonl"
echo ""
echo "If quality is good, run full generation with:"
echo "  bash scripts/generate_full_vqa.sh"
echo ""
echo "Or manually:"
echo "  python data/llm_vqa_generator.py \\"
echo "    --annotations data/stcray/train/annotations.json \\"
echo "    --images-dir data/stcray/train/images \\"
echo "    --output data/stcray_vqa_train.jsonl \\"
echo "    --model $MODEL \\"
echo "    --samples-per-image $SAMPLES_PER_IMAGE"
