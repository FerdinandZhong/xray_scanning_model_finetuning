# Using Gemini 2.0 Flash via OpenAI-Compatible API

This guide explains how to use Gemini 2.0 Flash through the OpenAI-compatible AI Gateway endpoint.

## Overview

**Gemini 2.0 Flash is accessed via OpenAI-compatible API** - no Google SDK needed!

- ✅ Uses standard OpenAI client
- ✅ Simple configuration
- ✅ Very cheap (~$0.0002 per image)
- ✅ Good quality for VQA generation

---

## Setup

### 1. Get API Key

Your Gemini access is through the Cloudera AI Gateway. Get your API key from your organization.

### 2. Set Environment Variable

```bash
# Set API key
export API_KEY="your-api-key-here"

# Or use OPENAI_API_KEY if you prefer
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Configure Endpoint

The AI Gateway endpoint is: `https://ai-gateway.dev.cloudops.cloudera.com/v1`

This is **automatically configured** in the VQA generation script.

---

## Usage

### Generate VQA Dataset (1000 samples)

```bash
# Set API key
export API_KEY="your-api-key"

# Run VQA generation
./scripts/generate_vqa_gemini.sh
```

**Configuration in script:**
```bash
MODEL="gemini-2.0-flash"
API_BASE="https://ai-gateway.dev.cloudops.cloudera.com/v1"
NUM_SAMPLES=1000
STRUCTURED_RATIO=0.3  # 30% JSON, 70% natural
```

### Direct Python Usage

```bash
python data/llm_vqa_generator.py \
  --annotations data/stcray_processed/train/annotations.json \
  --images-dir data/stcray_raw/STCray_TrainSet/Images \
  --output data/stcray_vqa_1k.jsonl \
  --model gemini-2.0-flash \
  --samples-per-image 3 \
  --num-samples 1000 \
  --structured-ratio 0.3 \
  --api-base https://ai-gateway.dev.cloudops.cloudera.com/v1
```

---

## How It Works

### 1. OpenAI-Compatible Client

The code uses the **standard OpenAI Python client**:

```python
import openai

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://ai-gateway.dev.cloudops.cloudera.com/v1"
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ],
    max_tokens=2000,
    temperature=0.7
)
```

### 2. No Google SDK Required

**Removed dependencies:**
- ❌ `google-generativeai` - NOT needed
- ❌ `google-cloud` - NOT needed
- ❌ Gemini-specific code paths

**Only need:**
- ✅ `openai` package (already in requirements)

### 3. Automatic Detection

The code automatically detects Gemini and uses OpenAI-compatible API:

```python
if "claude" in model.lower():
    # Use Anthropic client
    provider = "anthropic"
else:
    # Use OpenAI-compatible client (for GPT, Gemini, vLLM, etc.)
    provider = "openai"
```

---

## Cost Comparison

| Model | Cost per Image | 1000 Images | 10k Images |
|-------|----------------|-------------|------------|
| **Gemini 2.0 Flash** | ~$0.0002 | **~$0.20** | **~$2.00** |
| Claude 3.5 Sonnet | ~$0.015 | ~$15.00 | ~$150.00 |
| GPT-4o | ~$0.01 | ~$10.00 | ~$100.00 |

**Gemini 2.0 Flash is 50-75x cheaper!** 🎉

---

## API Key Sources

The code checks for API keys in this order:

1. `--api-key` command line argument (if provided)
2. `API_KEY` environment variable
3. `OPENAI_API_KEY` environment variable

**Recommended:** Use `API_KEY` for consistency:

```bash
export API_KEY="your-api-key"
```

---

## Troubleshooting

### Error: "API key not set"

**Solution:**
```bash
export API_KEY="your-api-key"
# or
export OPENAI_API_KEY="your-api-key"
```

### Error: "Connection refused" or "404 Not Found"

**Solution:** Check API base URL is correct:
```bash
echo $API_BASE
# Should be: https://ai-gateway.dev.cloudops.cloudera.com/v1
```

### Error: "Invalid API key"

**Solution:**
1. Verify your API key with your organization
2. Check the key is for the AI Gateway, not direct Google API

### Slow generation

**Normal behavior:** Gemini Flash is fast, but:
- ~1-2 seconds per image with API calls
- 1000 images ≈ 30-45 minutes
- Script prevents MacBook sleep automatically (`caffeinate`)

---

## Testing

### Quick API Test

```bash
# Test Gemini API connectivity
./scripts/test_gemini_api.sh
```

This verifies:
- ✅ API key is valid
- ✅ Endpoint is accessible
- ✅ Model responds correctly

---

## Files Modified

**Simplified for OpenAI-compatible API:**

1. `data/llm_vqa_generator.py`
   - Removed Google SDK imports
   - Removed `_call_gemini()` method
   - Unified under OpenAI-compatible path
   - Added `API_KEY` env variable support

2. `setup/requirements_vqa.txt`
   - Removed `google-generativeai` (not needed)
   - Only requires `openai` package

3. `scripts/generate_vqa_gemini.sh`
   - Uses `gemini-2.0-flash` model name
   - Sets `API_BASE` to AI Gateway
   - Checks for `API_KEY` env variable

---

## Summary

**Before (Complex):**
- Separate Google SDK path
- Gemini-specific code
- `google-generativeai` dependency
- Manual endpoint configuration

**After (Simple):**
- ✅ Single OpenAI-compatible path
- ✅ No Google SDK needed
- ✅ Works with `openai` package only
- ✅ Automatic AI Gateway integration
- ✅ Supports `API_KEY` or `OPENAI_API_KEY`

**Result:** Simpler, cleaner, more maintainable code! 🚀
