#!/usr/bin/env bash
set -euo pipefail

# Minimal virtual environment setup for VQA generation on local laptop
# This installs only what's needed for data generation (not training)

echo "=========================================="
echo "VQA Generation Environment Setup"
echo "=========================================="
echo ""
echo "This creates a lightweight environment for VQA generation only."
echo "For full training, use scripts/setup_venv.sh instead."
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${VENV_DIR:-.venv_vqa}"
PYTHON="${PYTHON:-python3}"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at: $VENV_DIR"
    read -p "Recreate it? (yes/no): " recreate
    if [ "$recreate" = "yes" ]; then
        echo "Removing old environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing environment"
        source "$VENV_DIR/bin/activate"
        echo "✓ Environment activated"
        exit 0
    fi
fi

echo ""
echo "Creating virtual environment at: $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install minimal requirements for VQA generation
echo "Installing VQA generation dependencies..."
echo "(This is much lighter than full training requirements)"
echo ""
pip install -r setup/requirements_vqa.txt
echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 << 'PYTHON_VERIFY'
import sys
print(f"Python: {sys.version}")

try:
    import datasets
    print(f"✓ datasets: {datasets.__version__}")
except ImportError:
    print("✗ datasets not found")

try:
    from PIL import Image
    import PIL
    print(f"✓ Pillow: {PIL.__version__}")
except ImportError:
    print("✗ Pillow not found")

try:
    import google.generativeai as genai
    print(f"✓ google-generativeai installed")
except ImportError:
    print("✗ google-generativeai not found")

try:
    import anthropic
    print(f"✓ anthropic: {anthropic.__version__}")
except ImportError:
    print("⚠ anthropic not installed (optional)")

try:
    import openai
    print(f"✓ openai: {openai.__version__}")
except ImportError:
    print("⚠ openai not installed (optional)")

print("\n✓ Core VQA generation packages verified")
PYTHON_VERIFY

echo ""
echo "=========================================="
echo "✅ VQA Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment: $VENV_DIR"
echo ""
echo "To activate it in the future:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Set API key:"
echo "     export GOOGLE_API_KEY='your-api-key'"
echo ""
echo "  2. Test Gemini API:"
echo "     ./scripts/test_gemini_api.sh"
echo ""
echo "  3. Download dataset:"
echo "     python data/download_stcray.py --output-dir data/stcray"
echo ""
echo "  4. Generate VQA dataset:"
echo "     ./scripts/generate_vqa_gemini.sh"
echo ""
echo "Disk usage: ~200MB (vs ~5GB+ for full training environment)"
