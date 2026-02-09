#!/usr/bin/env bash
set -euo pipefail

# Complete pipeline execution script for STCray dataset
# This script runs the entire pipeline from dataset download to deployment

echo "=========================================="
echo "X-ray VQA Fine-tuning - Complete Pipeline"
echo "=========================================="

# Configuration
DATASET="stcray"
MODEL="claude-3-5-sonnet-20241022"
SAMPLES_PER_IMAGE=3
MAX_TEST_IMAGES=100

# Check environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Run: bash scripts/setup_venv.sh"
    exit 1
fi

source .venv/bin/activate

# Check API key
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: No API key found"
    echo "Set: export ANTHROPIC_API_KEY=your_key"
    echo "Or: export OPENAI_API_KEY=your_key"
    exit 1
fi

echo ""
echo "Step 1: Download STCray dataset from HuggingFace"
echo "================================================"
python data/download_stcray.py --output-dir data/stcray

echo ""
echo "Step 2: Test LLM generation (100 images)"
echo "========================================"
python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train_test.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --max-images "$MAX_TEST_IMAGES"

echo ""
echo "Step 3: Validate quality of test generation"
echo "==========================================="
python data/llm_vqa_generator.py \
  --output data/stcray_vqa_train_test.jsonl \
  --validate

# Ask user to confirm before full generation
echo ""
echo "========================================"
echo "Test generation complete!"
echo "Review the test dataset and press Enter to continue with full generation,"
echo "or Ctrl+C to abort."
read -p "Continue? " 

echo ""
echo "Step 4: Generate full training VQA dataset"
echo "==========================================="
python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE"

echo ""
echo "Step 5: Generate validation VQA dataset"
echo "========================================"
python data/llm_vqa_generator.py \
  --annotations data/stcray/test/annotations.json \
  --images-dir data/stcray/test/images \
  --output data/stcray_vqa_val.jsonl \
  --model "$MODEL" \
  --samples-per-image "$SAMPLES_PER_IMAGE"

echo ""
echo "Step 6: Start training"
echo "====================="
python training/train_local.py --config configs/train_stcray.yaml

echo ""
echo "Step 7: Evaluate model"
echo "======================"
python evaluation/eval_vqa.py \
  --model outputs/qwen25vl_stcray_lora \
  --test-file data/stcray_vqa_val.jsonl \
  --output results/stcray_eval.json

python evaluation/eval_operational.py \
  --model outputs/qwen25vl_stcray_lora \
  --test-file data/stcray_vqa_val.jsonl \
  --output results/stcray_operational.json

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Model: outputs/qwen25vl_stcray_lora"
echo "Results: results/stcray_eval.json"
echo ""
echo "Next steps:"
echo "1. Review evaluation results"
echo "2. Deploy inference: python inference/api_server.py --model outputs/qwen25vl_stcray_lora"
