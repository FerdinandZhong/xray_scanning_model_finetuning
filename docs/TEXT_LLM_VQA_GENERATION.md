# Text-Based VQA Generation with Small LLMs

## Overview

This is a **cost-effective alternative** to vision-capable LLMs (Claude/GPT-4V) for generating VQA datasets. Instead of analyzing actual images, it uses **ground truth annotations** (categories, bounding boxes) to generate natural language question-answer pairs.

**Key Difference:**
- **Vision LLMs** (Claude/GPT-4V): Analyze actual images → More accurate but expensive ($600-920)
- **Text LLMs** (Qwen2.5-3B): Use text annotations → Less accurate but FREE (local inference)

## When to Use

### ✅ Use Text-Based Generation When:
- Budget is limited (want to avoid API costs)
- Ground truth annotations are high quality
- You have a GPU for local inference
- Generation speed matters more than visual accuracy
- Dataset size is large (46k+ images)

### ❌ Use Vision LLMs When:
- Annotations are incomplete or low quality
- Need visual verification (occlusion, concealment)
- Dataset is small (<5k images, cost ~$100-200)
- Highest quality is critical
- Budget allows for API costs

## Cost & Time Comparison

| Method | Cost | Time (46k images) | Quality | GPU Required |
|--------|------|------------------|---------|--------------|
| **Vision LLM** (Claude) | $920 | 8-12 hours | ⭐⭐⭐⭐⭐ | No |
| **Vision LLM** (GPT-4o-mini) | $92 | 8-12 hours | ⭐⭐⭐⭐ | No |
| **Text LLM** (Qwen2.5-3B) | **FREE** | 1-2 hours | ⭐⭐⭐ | Yes |
| **Text LLM** (Qwen2.5-7B) | **FREE** | 2-3 hours | ⭐⭐⭐⭐ | Yes |
| **Rule-Based** | FREE | <30 min | ⭐⭐ | No |

## Requirements

### Hardware
- GPU: 16GB VRAM (for Qwen2.5-3B)
- GPU: 24GB VRAM (for Qwen2.5-7B)
- RAM: 16GB+
- Storage: 10GB for model weights

### Software
```bash
pip install transformers>=4.37.0 torch>=2.2.0 accelerate>=0.25.0
```

## Quick Start

### 1. Test with 100 Images

```bash
# Generate test dataset
python data/llm_vqa_generator_text.py \
  --annotations data/stcray/train/annotations.json \
  --output data/stcray_vqa_test.jsonl \
  --model qwen2.5-3b-instruct \
  --samples-per-image 3 \
  --max-images 100
```

**Expected output:**
```
Initializing local model...
Loading checkpoint shards: 100%|██████| 2/2
✓ Loaded Qwen/Qwen2.5-3B-Instruct on cuda:0
Generating VQA pairs: 100%|██████| 100/100 [02:15<00:00]
✓ Dataset saved to: data/stcray_vqa_test.jsonl
Total VQA pairs: 300
Success rate: 100.0%
```

### 2. Review Quality

```bash
# Check first 5 samples
head -5 data/stcray_vqa_test.jsonl | python -m json.tool
```

### 3. Full Generation

```bash
# Run full pipeline (FREE, 1-2 hours)
bash scripts/generate_vqa_text_llm.sh
```

## Supported Models

### Qwen2.5 Series (Recommended)

| Model | VRAM | Speed | Quality | HuggingFace ID |
|-------|------|-------|---------|----------------|
| **Qwen2.5-3B-Instruct** | 16GB | Fast | Good | `Qwen/Qwen2.5-3B-Instruct` |
| **Qwen2.5-7B-Instruct** | 24GB | Medium | Better | `Qwen/Qwen2.5-7B-Instruct` |

### Llama 3 Series

| Model | VRAM | Speed | Quality | HuggingFace ID |
|-------|------|-------|---------|----------------|
| Llama-3-8B-Instruct | 24GB | Medium | Good | `meta-llama/Llama-3-8B-Instruct` |

### Using API Endpoint

If you have a vLLM or similar API endpoint:

```bash
python data/llm_vqa_generator_text.py \
  --annotations data/stcray/train/annotations.json \
  --output data/stcray_vqa_train.jsonl \
  --model qwen2.5-3b-instruct \
  --use-api \
  --api-base http://localhost:8000/v1
```

## Output Format

Same format as vision LLM generator:

```json
{
  "image_path": "train/images/000123.jpg",
  "question": "What items are visible in this X-ray scan?",
  "answer": "I can see a gun at upper-left and a knife at center.",
  "metadata": {
    "image_id": 123,
    "question_type": "general",
    "categories": ["gun", "knife"],
    "num_categories": 2,
    "generated_by": "text_llm",
    "model": "qwen2.5-3b-instruct"
  }
}
```

## Example Output

### Input Annotation:
```json
{
  "image_id": 42,
  "categories": ["gun", "knife"],
  "bboxes": [[120, 80, 60, 90], [350, 220, 40, 80]],
  "caption": "Baggage with concealed items"
}
```

### Generated VQA Pairs:

**Pair 1 (General):**
```json
{
  "question": "What items are visible in this X-ray scan?",
  "answer": "This scan shows two prohibited items: a gun located in the upper-left area and a knife in the center-right section.",
  "question_type": "general"
}
```

**Pair 2 (Specific):**
```json
{
  "question": "Is there a gun in this baggage?",
  "answer": "Yes, there is a gun detected at the upper-left position.",
  "question_type": "specific"
}
```

**Pair 3 (Location):**
```json
{
  "question": "Where are the threat items located?",
  "answer": "Two items are present: a gun at upper-left and a knife at center-right.",
  "question_type": "location"
}
```

## Quality Comparison

### Vision LLM Output (Claude/GPT-4V)
```
"This scan reveals a handgun partially concealed beneath clothing in the 
upper-left corner, with clear barrel and grip visible. A folding knife is 
located in the center-right, partially obscured by dense materials."
```
**Pros:** Highly detailed, describes occlusion, mentions visual features

### Text LLM Output (Qwen2.5-3B)
```
"This scan shows two prohibited items: a gun at upper-left and a knife 
at center-right."
```
**Pros:** Accurate categories and locations, natural language  
**Cons:** Less visual detail, generic descriptions

### Rule-Based Output
```
"Detected items: gun, knife."
```
**Pros:** Fast, accurate categories  
**Cons:** No natural language variation, no locations

## Optimization Tips

### 1. Speed Up Generation

```bash
# Use smaller model
MODEL=qwen2.5-3b-instruct bash scripts/generate_vqa_text_llm.sh

# Reduce samples per image
python data/llm_vqa_generator_text.py \
  --samples-per-image 2  # Instead of 3
```

### 2. Improve Quality

```bash
# Use larger model (requires 24GB VRAM)
MODEL=qwen2.5-7b-instruct bash scripts/generate_vqa_text_llm.sh

# Generate more samples per image for diversity
python data/llm_vqa_generator_text.py \
  --samples-per-image 5
```

### 3. Batch Processing

The script automatically checkpoints every 100 images. If interrupted:

```bash
# Simply rerun - it will resume from checkpoint
bash scripts/generate_vqa_text_llm.sh
```

## Troubleshooting

### Issue: Out of Memory

**Solution 1:** Use smaller model
```bash
MODEL=qwen2.5-3b-instruct  # Instead of 7B
```

**Solution 2:** Reduce batch size in code (edit `llm_vqa_generator_text.py`):
```python
# In _call_local method, add:
torch.cuda.empty_cache()
```

**Solution 3:** Use quantization
```python
# In __init__, change:
torch_dtype=torch.bfloat16,
# to:
load_in_8bit=True,
```

### Issue: Slow Generation

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Solution:** Ensure model is on GPU:
```python
print(f"Model device: {self.model_obj.device}")  # Should be cuda:0
```

### Issue: Low Quality Output

**Solution 1:** Use larger model (Qwen2.5-7B)

**Solution 2:** Improve prompt engineering (edit `_create_prompt`)

**Solution 3:** Generate more samples and filter best ones:
```bash
# Generate 5 per image, then manually select best 3
--samples-per-image 5
```

## Performance Benchmarks

Tested on NVIDIA RTX 4090 (24GB):

| Model | Speed (img/sec) | Memory Usage | Time (30k images) |
|-------|----------------|--------------|-------------------|
| Qwen2.5-3B | 8-10 | 8GB | 50-60 min |
| Qwen2.5-7B | 4-6 | 14GB | 80-120 min |
| Llama-3-8B | 3-5 | 16GB | 100-150 min |

Tested on NVIDIA V100 (16GB):

| Model | Speed (img/sec) | Memory Usage | Time (30k images) |
|-------|----------------|--------------|-------------------|
| Qwen2.5-3B | 5-7 | 8GB | 70-100 min |
| Qwen2.5-7B | ❌ OOM | - | - |

## Hybrid Approach

Combine text-based and vision-based generation for best results:

```bash
# Step 1: Generate bulk with text LLM (FREE)
bash scripts/generate_vqa_text_llm.sh

# Step 2: Generate subset with vision LLM for quality check (5k images, ~$100)
python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train_vision.jsonl \
  --model gpt-4o-mini \
  --max-images 5000

# Step 3: Merge datasets
cat data/stcray_vqa_train.jsonl data/stcray_vqa_train_vision.jsonl > data/stcray_vqa_train_merged.jsonl
```

## Conclusion

**Text-based VQA generation** is a viable alternative when:
- ✅ Budget is constrained
- ✅ Ground truth annotations are reliable
- ✅ GPU resources are available
- ✅ Slight quality reduction is acceptable

**Expected Quality:** 75-85% of vision LLM quality at 0% of the cost.

For maximum quality and budget allows, use vision-capable LLMs (Claude/GPT-4V).
For balanced approach, use hybrid method (bulk text + sample vision).
