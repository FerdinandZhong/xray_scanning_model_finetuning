#!/usr/bin/env bash
set -euo pipefail

# Start vLLM server with Qwen2.5-VL model
# This creates an OpenAI-compatible API endpoint for VQA generation

echo "=========================================="
echo "Starting Qwen2.5-VL vLLM Server"
echo "=========================================="

# Configuration
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. GPU is required for vLLM."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $GPU_COUNT GPU(s)"

if [ "$TENSOR_PARALLEL_SIZE" -gt "$GPU_COUNT" ]; then
    echo "❌ Error: Tensor parallel size ($TENSOR_PARALLEL_SIZE) > available GPUs ($GPU_COUNT)"
    exit 1
fi

# Activate environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check vLLM installation
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ Error: vLLM not installed"
    echo "Install with: pip install vllm>=0.6.0"
    exit 1
fi

echo ""
echo "Starting vLLM server..."
echo "API will be available at: http://localhost:$PORT/v1"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    --dtype bfloat16

# Note: For Qwen2.5-VL, you may need additional flags:
# --limit-mm-per-prompt image=10  # Max images per prompt
# --max-num-seqs 16  # Batch size
