# Local VQA Generation Setup Guide

Quick setup guide for generating VQA datasets on your local laptop with minimal dependencies.

## Overview

This setup is optimized for **VQA generation only** (not training). It installs only what's needed to run `llm_vqa_generator.py` with Gemini/Claude/GPT APIs.

**Disk Usage:**
- Minimal setup: ~200MB (VQA generation only)
- Full setup: ~5GB+ (includes PyTorch, training libraries)

**For local laptop VQA generation, use the minimal setup.**

## Quick Start

### 1. Setup Environment (One-time)

```bash
# Create minimal virtual environment
./scripts/setup_venv_vqa.sh

# This creates .venv_vqa/ with only VQA dependencies
```

### 2. Configure API Key

```bash
# For Gemini via OpenAI-compatible endpoint (RECOMMENDED - cheapest at ~$9)
export API_KEY="your-api-key"
# Or use OPENAI_API_KEY if you prefer
export OPENAI_API_KEY="your-api-key"

# Or for Claude (expensive, ~$900)
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Make it persistent:
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Test API Connection

```bash
# Activate environment
source .venv_vqa/bin/activate

# Test Gemini API
./scripts/test_gemini_api.sh
```

### 4. Download Dataset

```bash
# Download STCray dataset (~5GB)
python data/download_stcray.py --output-dir data/stcray
```

### 5. Generate VQA Dataset

```bash
# Generate with Gemini 2.0 Flash (RECOMMENDED)
./scripts/generate_vqa_gemini.sh

# Cost: ~$9
# Time: 1-2 hours
# Output: ~138k VQA pairs
```

## What's Included

### Minimal Setup (`requirements_vqa.txt`)

**Core packages** (~150MB):
- `datasets` - HuggingFace datasets for STCray
- `pillow` - Image processing
- `tqdm` - Progress bars
- `openai` - OpenAI-compatible client (for Gemini, GPT-4V, vLLM)
- `anthropic` - Claude API (optional)
- `pyyaml` - Config files

**NOT included** (saves ~5GB):
- PyTorch
- Transformers
- Training libraries (PEFT, DeepSpeed)
- Local model inference

### Full Setup (`requirements.txt`)

**Everything** (~5GB+):
- All VQA packages above
- PyTorch + CUDA
- Transformers + PEFT
- Training tools
- Evaluation tools
- Deployment tools

Use this for training in CAI or on GPU workstation.

## Environment Comparison

| Feature | Minimal (VQA only) | Full (Training) |
|---------|-------------------|-----------------|
| **Disk Space** | ~150MB | ~5GB+ |
| **Install Time** | 1-2 minutes | 10-20 minutes |
| **Use Case** | VQA generation | Training + VQA |
| **GPU Required** | No | Yes (for training) |
| **Location** | Local laptop | CAI/GPU server |

## VQA Generation Options

### Option 1: Gemini 2.0 Flash via OpenAI-compatible endpoint (RECOMMENDED)

**Best for local laptop:**
- Cost: ~$9 for full dataset
- Quality: Good (vision-capable)
- Speed: 1-2 hours
- Requirements: API key only
- Uses: OpenAI-compatible API endpoint

```bash
export API_KEY="your-key"
./scripts/generate_vqa_gemini.sh
```

See: [`docs/GEMINI_VQA_GENERATION.md`](GEMINI_VQA_GENERATION.md)

### Option 2: Claude Sonnet (Expensive)

**Best for highest quality:**
- Cost: ~$900 for full dataset
- Quality: Excellent
- Speed: 2-3 hours

```bash
export ANTHROPIC_API_KEY="your-key"
python data/llm_vqa_generator.py \
  --model claude-3-5-sonnet-20241022 \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train.jsonl \
  --samples-per-image 3
```

See: [`docs/AI_AGENT_VQA_GENERATION.md`](AI_AGENT_VQA_GENERATION.md)

### Option 3: GPT-4V (Moderate cost)

**Good balance:**
- Cost: ~$100 for full dataset
- Quality: Good
- Speed: 1-2 hours

```bash
export OPENAI_API_KEY="your-key"
python data/llm_vqa_generator.py \
  --model gpt-4o-mini \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train.jsonl \
  --samples-per-image 3
```

### Option 4: Qwen vLLM (FREE, requires GPU)

**Best for free + quality:**
- Cost: $0
- Quality: Excellent
- Speed: 1-2 hours
- Requirements: 24GB+ GPU, vLLM server

See: [`docs/QWEN_VL_VLLM_GUIDE.md`](QWEN_VL_VLLM_GUIDE.md)

*Note: Can't run on laptop without GPU. Use in CAI or GPU server.*

## Workflow

### Complete Local VQA Generation Flow

```bash
# 1. One-time setup
./scripts/setup_venv_vqa.sh

# 2. Activate environment (every session)
source .venv_vqa/bin/activate

# 3. Set API key (every session or add to ~/.bashrc)
export API_KEY="your-key"

# 4. Download dataset (one-time, ~5GB)
python data/download_stcray.py --output-dir data/stcray

# 5. Generate VQA dataset (1-2 hours, ~$9)
./scripts/generate_vqa_gemini.sh

# 6. Deactivate when done
deactivate
```

### Upload VQA Dataset to CAI

After generating VQA locally, upload to CAI for training:

```bash
# Option 1: Push to git (if files are small)
git add data/stcray_vqa_*.jsonl
git commit -m "Add VQA dataset"
git push

# Option 2: Upload to HuggingFace
huggingface-cli upload \
  your-username/stcray-vqa \
  data/stcray_vqa_train.jsonl \
  data/stcray_vqa_val.jsonl

# Then download in CAI:
# python -c "from datasets import load_dataset; ds = load_dataset('your-username/stcray-vqa')"

# Option 3: Use CAI file upload
# Upload via CAI UI: Files → Upload
```

## Troubleshooting

### Issue: Environment Too Large

**Problem:** Even minimal setup uses too much space

**Solution:** Install only essential packages
```bash
python3 -m venv .venv_vqa
source .venv_vqa/bin/activate
pip install openai pillow datasets tqdm pyyaml
```

This is ~100MB instead of 150MB.

### Issue: API Key Not Working

**Problem:** `API_KEY` or `OPENAI_API_KEY` not recognized

**Solution:**
```bash
# Check if set
echo $API_KEY
echo $OPENAI_API_KEY

# Set temporarily
export API_KEY="your-key"

# Set permanently
echo 'export API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

### Issue: Connection to AI Gateway Fails

**Problem:** Can't reach `https://ai-gateway.dev.cloudops.cloudera.com`

**Solution:**
- Ensure running on local laptop (not remote server)
- Check VPN connection
- Verify API Gateway is accessible

### Issue: Out of Memory

**Problem:** Laptop crashes during generation

**Solution:**
- Close other applications
- Reduce batch size: `--batch-save 10` (default: 100)
- Process in smaller chunks: `--max-images 5000`

## Resource Requirements

### Minimal Laptop Requirements

- **RAM:** 8GB+ (16GB recommended)
- **Disk:** 10GB free (5GB dataset + 5GB outputs)
- **CPU:** Any modern processor
- **GPU:** Not required
- **Network:** Stable internet for API calls

### Expected Performance

- **Installation:** 1-2 minutes
- **Dataset download:** 10-30 minutes (depends on network)
- **VQA generation:** 1-2 hours for full dataset
- **Cost:** ~$9 (Gemini) to ~$900 (Claude)

### MacBook Users: Prevent Sleep

**Important:** VQA generation takes 1-2 hours. The script automatically prevents your MacBook from sleeping using `caffeinate`.

**No action needed** - just run the script normally:
```bash
./scripts/generate_vqa_gemini.sh
```

For more details and advanced options, see: [`docs/MACOS_LONG_RUNNING.md`](MACOS_LONG_RUNNING.md)

## Next Steps

After VQA generation completes:

1. **Validate dataset:**
   ```bash
   python data/llm_vqa_generator.py \
     --output data/stcray_vqa_train.jsonl \
     --validate
   ```

2. **Upload to CAI for training:**
   - See "Upload VQA Dataset to CAI" section above

3. **Start training in CAI:**
   ```bash
   # In CAI workspace, trigger fine-tuning job
   # Jobs → finetune_model → Run
   ```

## Cost Summary

| Method | Cost | Time | Quality | Location |
|--------|------|------|---------|----------|
| **Gemini 2.0 Flash** | **~$9** | 1-2h | Good | Local laptop |
| GPT-4o-mini | ~$100 | 1-2h | Good | Local laptop |
| Claude Sonnet | ~$900 | 2-3h | Excellent | Local laptop |
| Qwen vLLM | $0 | 1-2h | Excellent | GPU server/CAI |

**Recommendation:** Use Gemini 2.0 Flash for best cost/quality balance on local laptop.

## FAQ

### Q: Can I generate VQA in CAI instead of locally?

**A:** Yes, but:
- AI Gateway is only accessible from local laptop (Gemini won't work)
- Use remote vLLM endpoint instead (set `VLLM_API_BASE` in CAI job)
- Or generate locally, then upload dataset to CAI

### Q: How do I switch between minimal and full environments?

**A:**
```bash
# Minimal (VQA only)
source .venv_vqa/bin/activate

# Full (training)
source .venv/bin/activate
```

They're separate environments, won't conflict.

### Q: Can I use the minimal setup for training?

**A:** No, training requires PyTorch, transformers, PEFT, etc.
- Use minimal setup for VQA generation
- Use full setup for training

### Q: What if I want to use local Qwen vLLM?

**A:** Add these to `requirements_vqa.txt`:
```bash
torch>=2.1.0
transformers>=4.37.0
vllm>=0.3.0
```

But this increases disk usage to ~3GB+. Better to use full setup for local inference.

## References

- [Main README](../README.md)
- [Gemini VQA Generation](GEMINI_VQA_GENERATION.md)
- [Complete Workflow](COMPLETE_WORKFLOW.md)
- [Qwen vLLM Guide](QWEN_VL_VLLM_GUIDE.md)
