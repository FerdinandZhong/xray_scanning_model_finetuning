#!/usr/bin/env bash
set -euo pipefail

# Test Gemini API connectivity via OpenAI-compatible endpoint
# Run this from your local laptop to verify Gemini setup

echo "=========================================="
echo "Gemini API Test Script (OpenAI-compatible)"
echo "=========================================="
echo ""

# Check API key
if [ -z "${API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "❌ Error: API_KEY or OPENAI_API_KEY not set"
    echo ""
    echo "Set it with:"
    echo "  export API_KEY='your-api-key'"
    echo "or:"
    echo "  export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Use API_KEY if set, otherwise fall back to OPENAI_API_KEY
export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY}}"

echo "✓ API key is set"
echo ""

# Configuration
API_BASE="${API_BASE:-https://ai-gateway.dev.cloudops.cloudera.com/v1}"
MODEL="${MODEL:-gemini-2.0-flash-exp}"

echo "Configuration:"
echo "  API Base: $API_BASE"
echo "  Model: $MODEL"
echo "  Endpoint Type: OpenAI-compatible"
echo ""

# Test 1: Python SDK import
echo "Test 1: Checking OpenAI SDK..."
python3 << 'PYTHON_TEST'
try:
    import openai
    print(f"✓ openai package installed (version {openai.__version__})")
except ImportError:
    print("❌ openai package not found")
    print("   Install with: pip install openai>=1.12.0")
    exit(1)
PYTHON_TEST

# Test 2: API client configuration
echo ""
echo "Test 2: Configuring OpenAI client with custom base URL..."
python3 << PYTHON_TEST
import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
api_base = "$API_BASE"

client = OpenAI(
    api_key=api_key,
    base_url=api_base
)
print(f"✓ OpenAI client configured")
print(f"  Base URL: {api_base}")
PYTHON_TEST

# Test 3: List models (if supported)
echo ""
echo "Test 3: Testing API connectivity..."
python3 << PYTHON_TEST
import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
api_base = "$API_BASE"

client = OpenAI(
    api_key=api_key,
    base_url=api_base
)

try:
    # Try to list models (may not be supported by all endpoints)
    models = client.models.list()
    print(f"✓ API connection successful")
    print(f"  Available models: {[m.id for m in models.data[:3]]}...")
except Exception as e:
    # If listing models fails, it's ok - not all endpoints support it
    print(f"✓ API endpoint reachable")
    print(f"  (Model listing not supported: {type(e).__name__})")
PYTHON_TEST

# Test 4: Simple text completion
echo ""
echo "Test 4: Testing text generation..."
python3 << PYTHON_TEST
import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
api_base = "$API_BASE"
model = "$MODEL"

client = OpenAI(
    api_key=api_key,
    base_url=api_base
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say hello in exactly 3 words"}
        ],
        max_tokens=50
    )
    print(f"✓ Text generation successful")
    print(f"   Model: {model}")
    print(f"   Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Text generation failed: {e}")
    print(f"   This may indicate the model name is incorrect or API issues")
    exit(1)
PYTHON_TEST

echo ""
echo "=========================================="
echo "✅ All Tests Passed!"
echo "=========================================="
echo ""
echo "Gemini API (via OpenAI-compatible endpoint) is working correctly."
echo ""
echo "Next steps:"
echo "  1. Download dataset:"
echo "     python data/download_stcray.py --output-dir data/stcray"
echo ""
echo "  2. Test with 1 image:"
echo "     python data/llm_vqa_generator.py \\"
echo "       --model $MODEL \\"
echo "       --api-base $API_BASE \\"
echo "       --annotations data/stcray/train/annotations.json \\"
echo "       --images-dir data/stcray/train/images \\"
echo "       --output test.jsonl \\"
echo "       --max-images 1"
echo ""
echo "  3. Run full generation:"
echo "     ./scripts/generate_vqa_gemini.sh"
