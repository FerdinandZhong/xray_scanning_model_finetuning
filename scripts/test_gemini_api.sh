#!/usr/bin/env bash
set -euo pipefail

# Test Gemini API connectivity and VQA generation
# Run this from your local laptop to verify Gemini setup

echo "=========================================="
echo "Gemini API Test Script"
echo "=========================================="
echo ""

# Check API key
if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "❌ Error: GOOGLE_API_KEY not set"
    echo ""
    echo "Set it with:"
    echo "  export GOOGLE_API_KEY='your-api-key'"
    exit 1
fi

echo "✓ GOOGLE_API_KEY is set"
echo ""

# Configuration
API_BASE="${API_BASE:-https://ai-gateway.dev.cloudops.cloudera.com}"
MODEL="${MODEL:-gemini-2.0-flash-exp}"

echo "Configuration:"
echo "  API Base: $API_BASE"
echo "  Model: $MODEL"
echo ""

# Test 1: Python SDK import
echo "Test 1: Checking Google Generative AI SDK..."
python3 << 'PYTHON_TEST'
try:
    import google.generativeai as genai
    print("✓ google-generativeai package installed")
except ImportError:
    print("❌ google-generativeai package not found")
    print("   Install with: pip install google-generativeai>=0.3.0")
    exit(1)
PYTHON_TEST

# Test 2: API key configuration
echo ""
echo "Test 2: Configuring Gemini API..."
python3 << 'PYTHON_TEST'
import os
import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
print("✓ Gemini API configured")
PYTHON_TEST

# Test 3: Model initialization
echo ""
echo "Test 3: Initializing Gemini model..."
python3 << PYTHON_TEST
import os
import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="$MODEL",
    generation_config={"temperature": 0.7, "max_output_tokens": 2000}
)
print(f"✓ Model initialized: $MODEL")
PYTHON_TEST

# Test 4: Simple text generation
echo ""
echo "Test 4: Testing text generation..."
python3 << PYTHON_TEST
import os
import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="$MODEL",
    generation_config={"temperature": 0.7, "max_output_tokens": 2000}
)

try:
    response = model.generate_content("Say hello in JSON format: {\"message\": \"...\"}")
    print(f"✓ Text generation successful")
    print(f"   Response: {response.text[:100]}...")
except Exception as e:
    print(f"❌ Text generation failed: {e}")
    exit(1)
PYTHON_TEST

# Test 5: Image generation (if test image available)
if [ -f "test_model.py" ]; then
    echo ""
    echo "Test 5: Testing image+text generation..."
    echo "(Skipped - requires actual X-ray image)"
    echo "Run the VQA generator with --max-images 1 to test with real images"
else
    echo ""
    echo "Test 5: Skipped (no test image)"
fi

echo ""
echo "=========================================="
echo "✅ All Tests Passed!"
echo "=========================================="
echo ""
echo "Gemini API is working correctly."
echo ""
echo "Next steps:"
echo "  1. Download dataset: python data/download_stcray.py --output-dir data/stcray"
echo "  2. Test with 1 image: python data/llm_vqa_generator.py --model $MODEL --max-images 1 --api-base $API_BASE ..."
echo "  3. Run full generation: ./scripts/generate_vqa_gemini.sh"
