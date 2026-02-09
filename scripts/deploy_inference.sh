#!/usr/bin/env bash
set -euo pipefail

# Deploy inference pipeline
# Starts vLLM engine and FastAPI server

echo "=========================================="
echo "Deploy Inference Pipeline"
echo "=========================================="

# Configuration
MODEL_PATH="${1:-outputs/qwen25vl_stcray_lora}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
API_PORT="${API_PORT:-8080}"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    echo "Train model first: bash scripts/train.sh"
    exit 1
fi

# Activate environment
source .venv/bin/activate

# Check GPU availability
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -lt "$TENSOR_PARALLEL" ]; then
    echo "Warning: Requested $TENSOR_PARALLEL GPUs, but only $GPU_COUNT available"
    TENSOR_PARALLEL=$GPU_COUNT
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Tensor parallel size: $TENSOR_PARALLEL"
echo "  API port: $API_PORT"
echo "  GPUs available: $GPU_COUNT"
echo ""

# Start API server (includes vLLM)
echo "Starting API server..."
echo ""
echo "Endpoints:"
echo "  - Health: http://localhost:$API_PORT/health"
echo "  - API docs: http://localhost:$API_PORT/docs"
echo "  - Inspect: POST http://localhost:$API_PORT/api/v1/inspect"
echo ""
echo "Press Ctrl+C to stop server"
echo ""

python inference/api_server.py \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$API_PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL"
