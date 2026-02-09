#!/usr/bin/env bash
set -euo pipefail

# Generate VQA dataset using Qwen2.5-VL via OpenAI-compatible API
# Assumes vLLM server is running (start with: bash scripts/start_qwen_vllm_server.sh)

echo "=========================================="
echo "VQA Generation with Qwen2.5-VL (vLLM API)"
echo "=========================================="

# Configuration
API_BASE="${API_BASE:-http://localhost:8000/v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-3}"

echo ""
echo "Configuration:"
echo "  API Base: $API_BASE"
echo "  Model: $MODEL"
echo "  Samples per image: $SAMPLES_PER_IMAGE"
echo ""

# Check if vLLM server is running
echo "Checking vLLM server status..."
if ! curl -s "${API_BASE}/models" > /dev/null 2>&1; then
    echo "❌ Error: vLLM server not accessible at $API_BASE"
    echo ""
    echo "Start the server first:"
    echo "  bash scripts/start_qwen_vllm_server.sh"
    exit 1
fi

echo "✓ vLLM server is running"

# Activate environment
source .venv/bin/activate

# Set OpenAI API configuration
export OPENAI_API_KEY="EMPTY"  # vLLM doesn't require a key
export OPENAI_API_BASE="$API_BASE"

echo ""
echo "Step 1: Test generation (100 images)"
echo "====================================="
echo "This will cost: FREE (local inference)"
echo "Estimated time: 5-10 minutes"
echo ""

python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train_test.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --max-images 100 \
  --rate-limit-delay 0.1

echo ""
echo "Step 2: Validate quality"
echo "========================"
python data/llm_vqa_generator.py \
  --output data/stcray_vqa_train_test.jsonl \
  --validate

echo ""
read -p "Quality looks good? Continue with full generation? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted by user"
    exit 0
fi

echo ""
echo "Step 3: Generate training VQA dataset"
echo "======================================"
echo "Estimated time: 2-3 hours"
echo ""

python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --rate-limit-delay 0.1 \
  --batch-save 100

echo ""
echo "Step 4: Generate validation VQA dataset"
echo "========================================"
echo ""

python data/llm_vqa_generator.py \
  --annotations data/stcray/test/annotations.json \
  --images-dir data/stcray/test/images \
  --output data/stcray_vqa_val.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --rate-limit-delay 0.1 \
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
