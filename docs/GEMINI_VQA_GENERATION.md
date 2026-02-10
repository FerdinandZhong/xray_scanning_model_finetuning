# VQA Generation with Gemini 2.0 Flash

## Overview

Generate VQA pairs locally using Google's Gemini 2.0 Flash model via OpenAI-compatible endpoint at Cloudera AI Gateway. This is the most cost-effective cloud-based option while maintaining good quality vision understanding.

**Key Benefits:**
- Very low cost (~$9 for full dataset)
- Vision-capable (actually sees the X-ray images)
- Fast generation (1-2 hours total)
- Good quality output
- Runs locally on your laptop
- Uses standard OpenAI-compatible API (no special SDK needed)

## Prerequisites

### System Requirements
- Local laptop with internet access
- Access to Cloudera AI Gateway
- Python 3.10+
- ~10GB free disk space for dataset

### API Access
- Google API key configured for Cloudera AI Gateway
- Must run from local laptop (AI Gateway is only accessible from laptop)

### Dataset
- STCray dataset downloaded locally
- Annotations in JSON format
- Images available

## Cost Comparison

| Model | Cost per Image | Full Dataset (~46k) | Quality | Location |
|-------|----------------|---------------------|---------|----------|
| **Gemini 2.0 Flash** | ~$0.0002 | **~$9** | Good | Cloud (local laptop) |
| GPT-4o-mini | ~$0.002 | ~$92 | Good | Cloud |
| Claude Sonnet | ~$0.02 | ~$920 | Excellent | Cloud |
| Qwen vLLM (local) | $0 | **$0** | Excellent | Local GPU |
| Text-only LLM | $0 | **$0** | Fair | Local GPU |

**Gemini 2.0 Flash provides the best balance for cloud-based generation when:**
- You don't have GPU for local vLLM
- You want vision capability (vs text-only)
- Budget is limited

## Setup

### 1. Install Dependencies

```bash
# Use minimal VQA setup (recommended)
./scripts/setup_venv_vqa.sh
source .venv_vqa/bin/activate

# Or install manually
pip install openai>=1.12.0 pillow datasets tqdm pyyaml
```

### 2. Set API Key

```bash
export API_KEY="your-api-key-here"
# Or use OPENAI_API_KEY if you prefer
export OPENAI_API_KEY="your-api-key-here"
```

To make it persistent across sessions, add to `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Download Dataset (if not already done)

```bash
python data/download_stcray.py --output-dir data/stcray
```

This downloads ~46k X-ray images with annotations (~5GB).

## Usage

### Quick Start (Recommended)

```bash
# Set API key
export API_KEY="your-api-key"

# Run generation script
./scripts/generate_vqa_gemini.sh
```

The script will:
1. Verify API key and dataset
2. Show cost estimate
3. Test with 10 images
4. Ask for confirmation
5. Generate full training set
6. Generate validation set

### Advanced Usage

**Custom configuration:**
```bash
MODEL=gemini-2.0-flash-exp \
SAMPLES_PER_IMAGE=5 \
API_BASE=https://ai-gateway.dev.cloudops.cloudera.com/v1 \
API_KEY="your-key" \
./scripts/generate_vqa_gemini.sh
```

**Direct Python call:**
```bash
export API_KEY="your-key"
export OPENAI_API_BASE="https://ai-gateway.dev.cloudops.cloudera.com/v1"
export OPENAI_API_KEY="$API_KEY"

python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train.jsonl \
  --model gemini-2.0-flash-exp \
  --samples-per-image 3 \
  --api-base "$OPENAI_API_BASE" \
  --rate-limit-delay 0.2 \
  --batch-save 100
```

**Resume interrupted generation:**
```bash
# If generation was interrupted, just re-run the same command
# The script automatically resumes from the last checkpoint
./scripts/generate_vqa_gemini.sh
```

## Workflow

### Step 1: Test Run (10 images)
- Duration: ~30 seconds
- Cost: ~$0.002
- Purpose: Verify API connectivity and output quality

**Output:** `data/stcray_vqa_gemini_test.jsonl`

Review the test output:
```bash
head -3 data/stcray_vqa_gemini_test.jsonl | python -m json.tool
```

### Step 2: Training Set (~30k images)
- Duration: 45-60 minutes
- Cost: ~$6
- Output: ~90k VQA pairs

**Output:** `data/stcray_vqa_train.jsonl`

### Step 3: Validation Set (~16k images)
- Duration: 25-35 minutes
- Cost: ~$3
- Output: ~48k VQA pairs

**Output:** `data/stcray_vqa_val.jsonl`

### Total
- **Time:** 1-2 hours
- **Cost:** ~$9
- **Output:** ~138k VQA pairs

## Output Format

Same format as other generators - JSONL with Qwen2.5-VL conversation format:

```json
{
  "image_id": "train_00001",
  "image_path": "data/stcray/train/images/00001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat items are visible in this X-ray scan?"
    },
    {
      "from": "gpt",
      "value": "The scan shows a laptop, mobile phone, and keys. The laptop is in the center with its screen and keyboard visible. The phone is partially obscured behind the laptop, and the keys are in the lower left corner."
    }
  ],
  "ground_truth": {
    "category": "positive",
    "items": ["laptop", "phone", "keys"]
  }
}
```

## Monitoring Progress

The generator provides real-time progress tracking:

```
Processing images: 45%|████████████          | 13500/30000 [0:45:23<0:55:12, 4.99 images/s]
Successful: 13487 | Failed: 13 | Checkpoint saved
```

**Checkpoints:**
- Saved every 100 images
- Allows resumption if interrupted
- Files: `data/stcray_vqa_train.jsonl.checkpoint`

## Troubleshooting

### Issue: API Key Error

**Error:**
```
Error: API_KEY environment variable not set
```

**Solution:**
```bash
export API_KEY="your-key"
# Or
export OPENAI_API_KEY="your-key"
```

### Issue: Connection Error to AI Gateway

**Error:**
```
Error: Failed to connect to https://ai-gateway.dev.cloudops.cloudera.com
```

**Possible causes:**
1. Not running from local laptop (AI Gateway only accessible from laptop)
2. Network connectivity issues
3. API Gateway is down

**Solution:**
- Ensure you're on your local laptop (not CAI workspace or remote server)
- Check network connectivity
- Verify API Gateway URL is correct

### Issue: Rate Limit Errors

**Error:**
```
Error 429: Too Many Requests
```

**Solution:** Increase delay between requests
```bash
# Edit generate_vqa_gemini.sh, change:
--rate-limit-delay 0.5  # 500ms between requests (instead of 200ms)
```

### Issue: API Authentication Error

**Error:**
```
Error 401: Unauthorized
```

**Solution:**
- Verify API key is correct
- Check API key has access to Gemini models
- Ensure API key is configured for the AI Gateway endpoint

### Issue: Model Not Found

**Error:**
```
Error: Model gemini-2.0-flash-exp not found
```

**Solution:** Use the correct model name for your AI Gateway:
```bash
# Try different model names
MODEL=gemini-2.0-flash \
./scripts/generate_vqa_gemini.sh

# Or check available models
MODEL=gemini-1.5-flash \
./scripts/generate_vqa_gemini.sh
```

### Issue: Generation Interrupted

**Solution:** Just re-run the script - it automatically resumes from checkpoint:
```bash
./scripts/generate_vqa_gemini.sh
```

The generator will:
1. Load existing checkpoint
2. Skip already processed images
3. Continue from where it stopped

## Validation

After generation completes, validate the output:

```bash
# Validate VQA dataset
python data/llm_vqa_generator.py \
  --output data/stcray_vqa_train.jsonl \
  --validate

# Check statistics
python -c "
import json
with open('data/stcray_vqa_train.jsonl') as f:
    lines = f.readlines()
print(f'Total samples: {len(lines)}')
print(f'First sample:')
print(json.dumps(json.loads(lines[0]), indent=2))
"
```

Expected output:
```
✓ Validated 90,000 VQA pairs
✓ All images exist
✓ All conversations have valid format
✓ Dataset ready for training
```

## Next Steps

### 1. Validate Output Quality

```bash
python data/llm_vqa_generator.py \
  --output data/stcray_vqa_train.jsonl \
  --validate
```

### 2. Start Training

```bash
python training/train_local.py --config configs/train_stcray.yaml
```

### 3. Upload to HuggingFace (Optional)

If you want to share or back up the VQA dataset:

```bash
huggingface-cli upload \
  your-username/stcray-vqa-gemini \
  data/stcray_vqa_train.jsonl \
  data/stcray_vqa_val.jsonl
```

## Comparison with Other VQA Generation Options

### When to Use Gemini 2.0 Flash

**Best for:**
- Running locally from laptop (not in CAI)
- Budget-conscious projects (~$9 vs $100-1000)
- Need vision capability (actually sees images)
- Want reasonable quality without GPU

**Advantages:**
- Very cheap (~$9 for full dataset)
- Vision-capable (understands image content)
- Fast (1-2 hours)
- No GPU required
- Good quality

**Disadvantages:**
- Requires local laptop (can't run in CAI)
- Requires API key
- Not free (but very cheap)
- Quality slightly lower than Claude/GPT-4

### When to Use Qwen vLLM (Remote)

**Best for:**
- Running in CAI workspace
- Zero cost
- Have GPU access (24GB+ VRAM)

See: [`docs/QWEN_VL_VLLM_GUIDE.md`](QWEN_VL_VLLM_GUIDE.md)

### When to Use Qwen vLLM (Local)

**Best for:**
- Have local GPU (24GB+ VRAM)
- Want zero cost
- Want highest quality

See: [`docs/QWEN_VL_VLLM_GUIDE.md`](QWEN_VL_VLLM_GUIDE.md)

### When to Use Text-only LLM

**Best for:**
- No GPU available
- Want zero cost
- Acceptable to generate from text only (no vision)

See: [`docs/TEXT_LLM_VQA_GENERATION.md`](TEXT_LLM_VQA_GENERATION.md)

### When to Use Claude/GPT-4

**Best for:**
- Need highest quality
- Budget allows ($100-1000)
- Want enterprise support

See: [`docs/AI_AGENT_VQA_GENERATION.md`](AI_AGENT_VQA_GENERATION.md)

## Cost Breakdown (Gemini 2.0 Flash)

Assuming:
- ~30,000 training images
- ~16,000 validation images
- 3 VQA pairs per image
- ~$0.0002 per image (rough estimate)

```
Training Set:
  30,000 images × $0.0002 = $6.00

Validation Set:
  16,000 images × $0.0002 = $3.20

Total: ~$9.20
```

**Note:** Actual cost may vary based on:
- Prompt length
- Response length
- API Gateway pricing
- Rate limits and retries

## Technical Details

### API Integration

The generator uses OpenAI-compatible API with custom base URL:

```python
from openai import OpenAI

# Configure client for Gemini via AI Gateway
client = OpenAI(
    api_key=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://ai-gateway.dev.cloudops.cloudera.com/v1"
)

# Generate with image (using OpenAI chat completion format)
response = client.chat.completions.create(
    model="gemini-2.0-flash-exp",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    }],
    max_tokens=2000
)
```

**Benefits of OpenAI-compatible approach:**
- Standard API format (works with multiple providers)
- No special SDK needed
- Same code works for OpenAI, Gemini, vLLM, etc.
- Easy to switch between providers

### Rate Limiting

Default settings:
- 200ms delay between requests
- 5 images per second
- ~3 retries on failure

Adjust if needed:
```bash
--rate-limit-delay 0.5  # 500ms = 2 images/sec
```

### Checkpointing

Automatic checkpointing every 100 images:
- File: `<output_file>.checkpoint`
- Contains progress metadata
- Enables automatic resumption

## Performance

**Expected throughput:**
- ~5 images per second (with 200ms delay)
- ~18,000 images per hour
- ~2.5 hours for full dataset (46k images)

**Actual timing may vary based on:**
- Network latency
- API Gateway load
- Image complexity
- Response length

### Running on MacBook

**Sleep Prevention (Automatic):**

The script automatically prevents your MacBook from sleeping during generation using `caffeinate`. No manual intervention needed.

**Just plug into AC power and run:**
```bash
./scripts/generate_vqa_gemini.sh
```

**Advanced options:**
```bash
# Disable automatic sleep prevention (not recommended)
USE_CAFFEINATE=no ./scripts/generate_vqa_gemini.sh

# Use tmux for extra resilience
tmux new -s vqa
./scripts/generate_vqa_gemini.sh
# Detach: Ctrl+B, then D
```

See [`docs/MACOS_LONG_RUNNING.md`](MACOS_LONG_RUNNING.md) for more details.

## Quality Assessment

Gemini 2.0 Flash produces good quality VQA pairs:

**Strengths:**
- Understands visual content (vision-capable)
- Good at object detection and localization
- Reasonable detail in descriptions
- Follows JSON format well

**Limitations compared to Claude/GPT-4:**
- Less detailed descriptions
- May miss subtle details
- Occasional formatting issues

**Recommendation:** Good enough for fine-tuning, especially considering the 100x cost savings over Claude.

## Example Output

### Sample VQA Pair

**Image:** X-ray scan with laptop and phone

**Question:** "What electronic devices are visible in this X-ray scan?"

**Answer:** "The scan shows a laptop computer in the center and a mobile phone to the right. The laptop's keyboard, screen, and internal components are clearly visible. The phone appears as a rectangular object with visible circuitry."

**Ground Truth Categories:** laptop, phone

## Advanced Configuration

### Use Different Gemini Model

```bash
# Use Gemini 1.5 Flash (if 2.0 not available)
MODEL=gemini-1.5-flash ./scripts/generate_vqa_gemini.sh

# Use Gemini Pro (higher quality, higher cost)
MODEL=gemini-1.5-pro ./scripts/generate_vqa_gemini.sh
```

### Generate More Pairs Per Image

```bash
# Generate 5 pairs per image instead of 3
SAMPLES_PER_IMAGE=5 ./scripts/generate_vqa_gemini.sh
```

This creates ~230k VQA pairs but costs ~$15 instead of $9.

### Custom API Endpoint

```bash
# Use different AI Gateway or direct Google API
API_BASE=https://generativelanguage.googleapis.com/v1beta \
./scripts/generate_vqa_gemini.sh
```

## Monitoring

### Real-time Progress

The script shows:
```
Processing images: 45%|████████████          | 13500/30000 [0:45:23<0:55:12, 4.99 images/s]
Successful: 13487 | Failed: 13
```

### Check Intermediate Results

```bash
# Count lines (= number of samples)
wc -l data/stcray_vqa_train.jsonl

# View latest samples
tail -5 data/stcray_vqa_train.jsonl | python -m json.tool

# Check for errors
grep -i error data/stcray_vqa_train.jsonl || echo "No errors found"
```

### Estimated Time Remaining

Based on current throughput, the script estimates remaining time:
```
[0:45:23<0:55:12, 4.99 images/s]
         ^^^^^^^  ^^^^^^^^^^^^
         elapsed  remaining   throughput
```

## Integration with Training Pipeline

After VQA generation completes, proceed with training:

```bash
# 1. Validate VQA dataset
python data/llm_vqa_generator.py \
  --output data/stcray_vqa_train.jsonl \
  --validate

# 2. Start local training (multi-GPU)
python training/train_local.py --config configs/train_stcray.yaml

# Or train in CAI workspace
# (Upload VQA files to CAI, then trigger finetune_model job)
```

## FAQ

### Q: Can I run this in CAI workspace?

**A:** No, the AI Gateway is only accessible from your local laptop. You must run this locally.

For CAI execution, use:
- Remote Qwen vLLM server (external endpoint)
- Pre-generated VQA dataset (upload to CAI)

### Q: How does quality compare to Qwen vLLM?

**A:** Gemini 2.0 Flash is slightly lower quality than Qwen2.5-VL but still good enough for fine-tuning. The main advantage is cost (~$9 vs $0 but no GPU needed).

### Q: Can I use this with OpenAI or Claude instead?

**A:** Yes, but they're much more expensive:
- Gemini 2.0 Flash: ~$9
- GPT-4o-mini: ~$92
- Claude Sonnet: ~$920

Use the existing scripts:
- `./scripts/generate_full_vqa.sh` for Claude/GPT

### Q: What if I don't have API access to AI Gateway?

**A:** Options:
1. Use text-only VQA generation (free): `./scripts/generate_vqa_text_llm.sh`
2. Set up local Qwen vLLM server: See `docs/QWEN_VL_VLLM_GUIDE.md`
3. Request API Gateway access from your Cloudera admin

### Q: Can I stop and resume generation?

**A:** Yes! The generator saves checkpoints every 100 images. Just re-run the same command and it will automatically resume.

## Support

For issues:
1. Check GOOGLE_API_KEY is set correctly
2. Verify you're on local laptop (not CAI)
3. Test with small sample (`--max-images 1`)
4. Check AI Gateway connectivity
5. Review error logs in output

## References

- [Main README](../README.md)
- [Complete Workflow](COMPLETE_WORKFLOW.md)
- [Qwen vLLM Guide](QWEN_VL_VLLM_GUIDE.md)
- [Text-only LLM Generation](TEXT_LLM_VQA_GENERATION.md)
