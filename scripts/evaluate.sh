#!/usr/bin/env bash
set -euo pipefail

# Evaluation script for trained model
# Runs VQA metrics and operational benchmarks

echo "=========================================="
echo "Model Evaluation"
echo "=========================================="

# Configuration
MODEL_PATH="${1:-outputs/qwen25vl_stcray_lora}"
TEST_FILE="${2:-data/stcray_vqa_val.jsonl}"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    echo ""
    echo "Train model first:"
    echo "  bash scripts/train.sh"
    exit 1
fi

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

# Activate environment
source .venv/bin/activate

# Create results directory
mkdir -p results

echo ""
echo "Model: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo ""

# VQA Evaluation
echo "Step 1: VQA Metrics (Accuracy, F1, BLEU, ROUGE)"
echo "================================================"
python evaluation/eval_vqa.py \
  --model "$MODEL_PATH" \
  --test-file "$TEST_FILE" \
  --output results/eval_vqa_results.json \
  --max-samples 500  # Limit for faster evaluation

echo ""
echo "Step 2: Operational Benchmarks (Latency, Throughput)"
echo "===================================================="
python evaluation/eval_operational.py \
  --model "$MODEL_PATH" \
  --test-file "$TEST_FILE" \
  --batch-sizes 1,2,4,8,16 \
  --num-runs 20 \
  --output results/operational_benchmarks.json

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - VQA metrics: results/eval_vqa_results.json"
echo "  - Operational: results/operational_benchmarks.json"
echo ""
echo "View results:"
echo "  cat results/eval_vqa_results.json | python -m json.tool"
echo ""
echo "Next step: Deploy inference"
echo "  bash scripts/deploy_inference.sh"
