#!/usr/bin/env bash
set -euo pipefail

# Comprehensive setup verification before VQA generation
# This script checks all prerequisites to avoid costly failures

echo "=========================================="
echo "Setup Verification Script"
echo "=========================================="
echo ""

EXIT_CODE=0

# Check 1: Virtual environment
echo "[1/8] Checking virtual environment..."
if [ -d ".venv" ]; then
    echo "  ✓ Virtual environment exists"
else
    echo "  ✗ Virtual environment not found"
    echo "    Run: bash scripts/setup_venv.sh"
    EXIT_CODE=1
fi
echo ""

# Check 2: Python dependencies
echo "[2/8] Checking Python dependencies..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    
    MISSING_DEPS=()
    for dep in anthropic openai transformers datasets torch pillow tqdm; do
        if ! python -c "import $dep" 2>/dev/null; then
            MISSING_DEPS+=("$dep")
        fi
    done
    
    if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
        echo "  ✓ All required dependencies installed"
    else
        echo "  ✗ Missing dependencies: ${MISSING_DEPS[*]}"
        echo "    Run: pip install -r setup/requirements.txt"
        EXIT_CODE=1
    fi
else
    echo "  ⚠ Skipping (venv not found)"
fi
echo ""

# Check 3: API Keys
echo "[3/8] Checking API keys..."
HAS_ANTHROPIC=false
HAS_OPENAI=false

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "  ✓ ANTHROPIC_API_KEY is set"
    HAS_ANTHROPIC=true
else
    echo "  ⚠ ANTHROPIC_API_KEY not set"
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "  ✓ OPENAI_API_KEY is set"
    HAS_OPENAI=true
else
    echo "  ⚠ OPENAI_API_KEY not set"
fi

if [ "$HAS_ANTHROPIC" = false ] && [ "$HAS_OPENAI" = false ]; then
    echo "  ✗ No API keys found"
    echo "    Set one of:"
    echo "      export ANTHROPIC_API_KEY=your_key"
    echo "      export OPENAI_API_KEY=your_key"
    EXIT_CODE=1
fi
echo ""

# Check 4: Dataset download
echo "[4/8] Checking dataset..."
if [ -d "data/stcray" ]; then
    if [ -f "data/stcray/train/annotations.json" ] && [ -f "data/stcray/test/annotations.json" ]; then
        TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('data/stcray/train/annotations.json'))))")
        TEST_COUNT=$(python -c "import json; print(len(json.load(open('data/stcray/test/annotations.json'))))")
        echo "  ✓ STCray dataset found"
        echo "    Training samples: $TRAIN_COUNT"
        echo "    Test samples: $TEST_COUNT"
    else
        echo "  ✗ Dataset annotations missing"
        echo "    Run: python data/download_stcray.py"
        EXIT_CODE=1
    fi
else
    echo "  ✗ Dataset not found"
    echo "    Run: python data/download_stcray.py"
    EXIT_CODE=1
fi
echo ""

# Check 5: GPU availability
echo "[5/8] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "  ✓ $GPU_COUNT GPU(s) detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while IFS=, read -r name memory; do
            echo "    - $name ($memory)"
        done
    else
        echo "  ⚠ No GPUs found"
        echo "    Training will be slow on CPU"
    fi
else
    echo "  ⚠ nvidia-smi not found"
    echo "    GPU check skipped"
fi
echo ""

# Check 6: Disk space
echo "[6/8] Checking disk space..."
REQUIRED_GB=50
AVAILABLE_KB=$(df -k . | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))

if [ "$AVAILABLE_GB" -ge "$REQUIRED_GB" ]; then
    echo "  ✓ Sufficient disk space: ${AVAILABLE_GB}GB available"
else
    echo "  ✗ Insufficient disk space: ${AVAILABLE_GB}GB available (need ${REQUIRED_GB}GB)"
    EXIT_CODE=1
fi
echo ""

# Check 7: Test data loading (dry run)
echo "[7/8] Testing data loading..."
if [ -f "data/stcray/train/annotations.json" ]; then
    python data/llm_vqa_generator.py \
        --annotations data/stcray/train/annotations.json \
        --images-dir data/stcray/train/images \
        --output data/test_output.jsonl \
        --dry-run \
        2>&1 | tail -10
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Data loading test passed"
    else
        echo "  ✗ Data loading test failed"
        EXIT_CODE=1
    fi
else
    echo "  ⚠ Skipping (dataset not found)"
fi
echo ""

# Check 8: Configuration files
echo "[8/8] Checking configuration files..."
CONFIG_FILES=(
    "configs/train_stcray.yaml"
    "setup/requirements.txt"
)

ALL_CONFIGS_OK=true
for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        echo "  ✓ $config"
    else
        echo "  ✗ $config missing"
        ALL_CONFIGS_OK=false
        EXIT_CODE=1
    fi
done
echo ""

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All checks passed!"
    echo ""
    echo "Next steps:"
    echo "  1. Test LLM generation (100 images):"
    echo "     bash scripts/test_llm_generation.sh"
    echo ""
    echo "  2. Review test output quality"
    echo ""
    echo "  3. Run full generation (~\$300-900):"
    echo "     bash scripts/generate_full_vqa.sh"
else
    echo "✗ Some checks failed"
    echo ""
    echo "Please fix the issues above before proceeding."
    exit $EXIT_CODE
fi
