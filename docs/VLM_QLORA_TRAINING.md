# VLM QLoRA Training Guide
## Qwen3-VL-2B for Multi-Object X-ray Threat Detection

Complete guide for fine-tuning Qwen3-VL-2B-Instruct on STCray dataset using QLoRA (4-bit quantization + LoRA adapters) for T4 GPU.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Dataset Preparation](#dataset-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [CAI Deployment](#cai-deployment)
8. [Memory Optimization](#memory-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Comparison with YOLO](#comparison-with-yolo)

---

## Overview

### What is QLoRA?

**QLoRA** (Quantized Low-Rank Adaptation) is a memory-efficient fine-tuning technique that combines:

- **4-bit Quantization**: Compress model weights to 4-bit using NormalFloat4 (NF4)
- **LoRA Adapters**: Train small adapter matrices instead of full model weights
- **Double Quantization**: Further compress quantization constants
- **Paged Optimizers**: Prevent OOM by offloading optimizer states

**Result**: Fine-tune 2B models on 16GB GPU (T4) with only ~3-6GB VRAM usage.

### Why Qwen3-VL-2B?

- **Small but powerful**: 2B parameters, fast inference
- **Vision-language**: Native multi-modal understanding
- **Structured output**: JSON generation for object detection
- **T4 compatible**: Fits comfortably with QLoRA on 16GB VRAM

### Dataset: STCray

- **46,642 images** (37,316 train / 9,326 test)
- **21 threat categories**: Explosive, Gun, Knife, Battery, etc.
- **Multi-object scenarios**: Real airport security scans with 1-10+ objects per image
- **Production-ready**: Professional annotations from security experts

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA T4 (16GB VRAM) or better
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space (dataset + checkpoints)

### Software Requirements

```bash
# Python 3.10+
python --version  # Should be 3.10 or higher

# CUDA 11.8+ or 12.1+
nvidia-smi  # Check CUDA version

# Git LFS (for STCray dataset)
git lfs install
```

### Python Dependencies

Already included in [`setup/requirements.txt`](../setup/requirements.txt):

```
torch>=2.2.0
transformers>=4.57.0
accelerate>=1.0.0
peft>=0.14.0
bitsandbytes>=0.44.0
trl>=0.29.0
datasets>=2.16.0
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
cd /path/to/xray_scanning_model_finetuning

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies (fast with uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv pip install -r setup/requirements.txt
```

### 2. Download STCray Dataset

```bash
# Pull STCray archives from Git LFS
git lfs pull

# Extract and process
python cai_integration/download_dataset.py
```

Expected output:
```
✓ Train: 37,316 images
✓ Test: 9,326 images
✓ Annotations processed
```

### 3. Convert to VLM Format

```bash
python data/convert_stcray_to_vlm.py \
  --input-dir data/stcray_processed \
  --output-dir data/stcray_vlm
```

Expected output:
```
✓ stcray_vlm_train.jsonl (37,316 samples)
✓ stcray_vlm_test.jsonl (9,326 samples)
✓ statistics.json
```

### 4. Train with QLoRA

```bash
python training/train_vlm_qlora.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-data data/stcray_vlm/stcray_vlm_train.jsonl \
  --eval-data data/stcray_vlm/stcray_vlm_test.jsonl \
  --output-dir checkpoints/qwen3vl-2b-xray-qlora \
  --num-train-epochs 3 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4
```

Training time: ~36-45 hours on T4 for 3 epochs.

### 5. Evaluate

```bash
python evaluation/eval_vlm_qlora.py \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --finetuned-model checkpoints/qwen3vl-2b-xray-qlora/final \
  --test-data data/stcray_vlm/stcray_vlm_test.jsonl \
  --output-dir test_results/vlm_qlora_eval
```

---

## Dataset Preparation

### Understanding STCray Format

Original STCray structure:
```
data/stcray_processed/
├── train/
│   └── annotations.json      # List of image annotations
└── test/
    └── annotations.json
```

Each annotation:
```json
{
  "image_id": 0,
  "image_path": "data/stcray_raw/STCray_TrainSet/Images/Class 01_Explosive/img_001.jpg",
  "categories": ["Explosive", "Battery"],
  "bboxes": [[10, 20, 50, 60], [100, 150, 30, 40]]
}
```

### VLM Conversion

The conversion script transforms to VQA format:

```json
{
  "image_path": "/path/to/image.jpg",
  "question": "Detect and list all prohibited items in this X-ray baggage scan with their bounding boxes.",
  "answer": "{\"objects\": [{\"category\": \"Explosive\", \"bbox\": [0.01, 0.02, 0.05, 0.06], \"threat_level\": \"critical\"}, {\"category\": \"Battery\", \"bbox\": [0.10, 0.15, 0.13, 0.19], \"threat_level\": \"medium\"}]}",
  "metadata": {
    "question_type": "structured_list",
    "num_objects": 2,
    "num_threats": 2,
    "has_critical_threat": true
  }
}
```

**Key transformations**:
- Bounding boxes normalized to [0, 1] range: [x1, y1, x2, y2]
- All objects from single image included
- Threat levels assigned (critical, high, medium, low, none)
- Rich metadata for evaluation filtering

### Conversion Options

```bash
python data/convert_stcray_to_vlm.py --help
```

Options:
- `--input-dir`: STCray processed annotations (default: `data/stcray_processed`)
- `--output-dir`: Output directory for JSONL (default: `data/stcray_vlm`)
- `--project-root`: Project root for resolving paths

---

## Training

### Training Script Arguments

```bash
python training/train_vlm_qlora.py --help
```

**Model arguments**:
- `--model-name`: Base model (default: `Qwen/Qwen3-VL-2B-Instruct`)

**Data arguments**:
- `--train-data`: Training JSONL file
- `--eval-data`: Evaluation JSONL file
- `--image-root`: Root directory for images (if relative paths)

**Training hyperparameters**:
- `--num-train-epochs`: Number of epochs (default: 3)
- `--per-device-train-batch-size`: Batch size per GPU (default: 2)
- `--gradient-accumulation-steps`: Gradient accumulation (default: 4)
  - **Effective batch size** = batch_size × grad_accum = 2 × 4 = 8
- `--learning-rate`: Learning rate (default: 2e-4)
- `--warmup-steps`: Warmup steps (default: 100)
- `--max-seq-length`: Max sequence length (default: 2048)

**QLoRA parameters**:
- `--lora-r`: LoRA rank (default: 16)
- `--lora-alpha`: LoRA alpha (default: 32)
- `--lora-dropout`: LoRA dropout (default: 0.05)

**Other**:
- `--output-dir`: Output directory for checkpoints
- `--resume-from-checkpoint`: Resume from checkpoint path
- `--logging-steps`, `--save-steps`, `--eval-steps`

### Training Configuration

The script uses these optimizations for T4 GPU:

```python
# 4-bit quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# LoRA adapters
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training arguments
TrainingArguments(
    fp16=False,
    bf16=True,  # Better for quantized models
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_grad_norm=1.0
)
```

### Monitoring Training

```bash
# View logs in real-time
tail -f checkpoints/qwen3vl-2b-xray-qlora/logs/events.out.tfevents.*

# Or use TensorBoard
tensorboard --logdir checkpoints/qwen3vl-2b-xray-qlora/logs
```

### Checkpoints

Checkpoints are saved every 500 steps to:
```
checkpoints/qwen3vl-2b-xray-qlora/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── final/              # Final LoRA adapters
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

---

## Evaluation

### Evaluation Script

```bash
python evaluation/eval_vlm_qlora.py \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --finetuned-model checkpoints/qwen3vl-2b-xray-qlora/final \
  --test-data data/stcray_vlm/stcray_vlm_test.jsonl \
  --output-dir test_results/vlm_qlora_eval \
  --iou-threshold 0.5 \
  --num-samples 100  # Optional: evaluate on first 100 samples
```

Options:
- `--skip-base`: Skip base model evaluation (faster)
- `--iou-threshold`: IoU threshold for matching predictions (default: 0.5)
- `--num-samples`: Limit evaluation to N samples

### Metrics

**Overall metrics**:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **Average IoU**: Mean IoU across all matched predictions
- **JSON Parsing Rate**: Success rate of parsing structured JSON

**Per-category metrics**:
- Precision, Recall, F1 for each of 21 categories

**Multi-object metrics**:
- Accuracy on images with 3+ objects
- Detection rate for complex scenarios

### Results

Evaluation outputs:
```
test_results/vlm_qlora_eval/
├── evaluation_results.json  # Full results
├── comparison_report.md     # Markdown report
└── predictions/             # Per-image predictions
```

Example report:
```markdown
# Qwen3-VL-2B Evaluation Report

## Overall Metrics Comparison

| Metric | Base | Fine-tuned | Improvement |
|--------|------|------------|-------------|
| Precision | 0.234 | 0.856 | +265.8% |
| Recall | 0.189 | 0.823 | +335.4% |
| F1 | 0.209 | 0.839 | +301.4% |
| Avg IoU | 0.312 | 0.721 | +131.1% |
| JSON Parsing | 45.2% | 97.8% | +116.4% |

## Multi-Object Detection
- Fine-tuned accuracy: 87.3%
- Images evaluated: 3,245
```

---

## CAI Deployment

### Deploy via GitHub Actions

1. Go to **Actions** tab → **Deploy X-ray Detection to CAI**
2. Click **Run workflow**
3. Select:
   - **model_type**: `vlm_qlora`
   - **trigger_jobs**: `true`
4. Click **Run workflow**

### Manual CAI Deployment

```bash
# Setup environment
bash cai_integration/setup_environment.sh

# Download STCray
python cai_integration/download_dataset.py

# Convert to VLM format
python data/convert_stcray_to_vlm.py

# Train
python training/train_vlm_qlora.py \
  --train-data data/stcray_vlm/stcray_vlm_train.jsonl \
  --eval-data data/stcray_vlm/stcray_vlm_test.jsonl \
  --output-dir /home/cdsw/checkpoints/qwen3vl-2b-xray-qlora

# Evaluate
python evaluation/eval_vlm_qlora.py \
  --finetuned-model /home/cdsw/checkpoints/qwen3vl-2b-xray-qlora/final \
  --test-data data/stcray_vlm/stcray_vlm_test.jsonl \
  --output-dir /home/cdsw/test_results/vlm_qlora_eval
```

### CAI Job Configuration

Jobs defined in [`cai_integration/jobs_config_vlm.yaml`](../cai_integration/jobs_config_vlm.yaml):

```yaml
1. download_dataset     # 30 min, 2 CPU, 8GB RAM
2. convert_to_vlm       # 15 min, 4 CPU, 16GB RAM
3. train_vlm_qlora      # 48 hours, 8 CPU, 64GB RAM, 1×T4 GPU
4. evaluate_vlm         # 2 hours, 4 CPU, 32GB RAM, 1×T4 GPU
```

---

## Memory Optimization

### Expected VRAM Usage

| Configuration | VRAM |
|---------------|------|
| Base model (4-bit) | ~1 GB |
| LoRA adapters | ~50-100 MB |
| Optimizer states (8-bit) | ~100-200 MB |
| Activations (batch=2) | ~2-3 GB |
| **Total** | **3-6 GB** |

**T4 GPU (16GB)**: Comfortable margin of 10-13GB unused.

### If OOM Occurs

**1. Reduce batch size**:
```bash
--per-device-train-batch-size 1  # Instead of 2
--gradient-accumulation-steps 8  # Double to maintain effective batch
```

**2. Reduce sequence length**:
```bash
--max-seq-length 1024  # Instead of 2048
```

**3. Reduce LoRA rank**:
```bash
--lora-r 8  # Instead of 16
--lora-alpha 16  # Instead of 32
```

**4. Train fewer modules**:
In [`training/train_vlm_qlora.py`](../training/train_vlm_qlora.py), comment out additional target modules:
```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # "gate_proj",  # Comment out
    # "up_proj",    # Comment out
    # "down_proj",  # Comment out
],
```

### Monitor VRAM

```bash
# During training, watch VRAM usage
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Issue: JSON Parsing Failures

**Symptom**: Model outputs text instead of JSON.

**Solution**:
1. Check training data format in JSONL
2. Increase training epochs (model needs more examples)
3. Add explicit JSON instruction in prompt:
   ```python
   prompt = "... Provide your response in valid JSON format only."
   ```

### Issue: Low Detection Recall

**Symptom**: Model misses many objects.

**Solutions**:
1. **Train longer**: Increase `--num-train-epochs` to 5
2. **Increase LoRA rank**: Use `--lora-r 32 --lora-alpha 64`
3. **Reduce learning rate**: Try `--learning-rate 1e-4`
4. **Check data**: Ensure training data has multi-object examples

### Issue: Slow Training

**Expected**: ~12-15 hours per epoch on T4 for 37,316 images.

**If slower**:
1. Check GPU utilization: `nvidia-smi`
2. Reduce logging frequency: `--logging-steps 50`
3. Disable evaluation during training: Remove `--eval-data`

### Issue: Checkpoint Loading Fails

**Symptom**: Can't resume from checkpoint.

**Solution**:
```bash
# Load LoRA adapters with PeftModel
from peft import PeftModel
model = Qwen3VLForConditionalGeneration.from_pretrained(...)
model = PeftModel.from_pretrained(model, "checkpoints/.../final")
```

---

## Comparison with YOLO

| Aspect | YOLO (Recommended) | VLM QLoRA (This Guide) |
|--------|-------------------|------------------------|
| **Speed** | 20-100ms/image | 200-500ms/image |
| **Model Size** | 11-47MB | 1GB (base) + 50MB (LoRA) |
| **VRAM** | 2-8GB | 3-6GB |
| **Training Time** | 2-4 hours | 36-45 hours |
| **Use Case** | Production, real-time | Research, explanation |
| **Output** | Bounding boxes | JSON + natural language |
| **Accuracy** | High (mAP 0.7-0.9) | Good (F1 0.6-0.8) |
| **Multi-object** | Excellent | Good |
| **Explainability** | None | Natural language |

### When to Use VLM QLoRA

- **Research**: Understanding model behavior
- **Explanation needed**: Security audits requiring justification
- **Flexible queries**: Ask natural language questions about scans
- **JSON output**: Structured data for downstream systems

### When to Use YOLO

- **Production**: Real-time screening
- **High throughput**: Processing thousands of bags/hour
- **Edge devices**: Deployment on resource-constrained hardware
- **Simple detection**: Just need bounding boxes

---

## Next Steps

1. **Experiment with hyperparameters**: Try different LoRA ranks, learning rates
2. **Integrate additional datasets**: Combine STCray with PIDray or HiXray
3. **Deploy inference API**: Create FastAPI endpoint for model
4. **Compare with larger models**: Try Qwen3-VL-7B or Qwen3-VL-30B

---

## References

- [Qwen3-VL Paper](https://arxiv.org/abs/2505.09388)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [STCray Dataset](https://huggingface.co/datasets/Naoufel555/STCray-Dataset)
- [PIDray Dataset](https://arxiv.org/abs/2211.10763)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [bitsandbytes Documentation](https://huggingface.co/docs/bitsandbytes)

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review training logs in `checkpoints/*/logs/`
- Monitor VRAM with `nvidia-smi`
- Reduce batch size if OOM

---

**Happy training!** 🚀
