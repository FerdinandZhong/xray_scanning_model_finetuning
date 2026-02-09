# Complete Workflow Guide

End-to-end workflow for X-ray VQA model fine-tuning and deployment.

## Overview

This guide walks through the entire pipeline:
1. Environment setup
2. Dataset download and preparation
3. LLM-based VQA generation
4. Model training
5. Evaluation
6. Deployment

## Prerequisites

- Python 3.10+
- 2-8 NVIDIA GPUs (16GB+ VRAM each)
- 100GB+ storage
- CUDA 12.1+
- API key: Anthropic (Claude) or OpenAI (GPT-4V)

## Step-by-Step Workflow

### 1. Environment Setup

```bash
# Clone repository
cd /path/to/xray_scanning_model_finetuning

# Create virtual environment
bash scripts/setup_venv.sh

# Activate environment
source .venv/bin/activate

# Verify GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 2. Dataset Download

```bash
# Download STCray dataset from HuggingFace
python data/download_stcray.py --output-dir data/stcray

# Verify dataset structure
python data/download_stcray.py --output-dir data/stcray --verify
```

**Expected Output:**
```
data/stcray/
├── train/
│   ├── images/       # ~30,000 X-ray images
│   └── annotations.json
└── test/
    ├── images/       # ~16,000 X-ray images
    └── annotations.json
```

### 3. VQA Dataset Generation (LLM-based)

#### 3.1. Set API Key

```bash
# For Claude (recommended)
export ANTHROPIC_API_KEY="your_key_here"

# Or for GPT-4V
export OPENAI_API_KEY="your_key_here"
```

#### 3.2. Test Generation (100 images)

```bash
# Test with small sample first
bash scripts/test_llm_generation.sh
```

This will:
- Generate VQA pairs for 100 images
- Save to `data/stcray_vqa_train_test.jsonl`
- Display quality metrics
- Estimated cost: ~$2-6

**Review the output:**
```bash
# View samples
head -n 20 data/stcray_vqa_train_test.jsonl | python -m json.tool

# Validate quality
python data/llm_vqa_generator.py \
  --output data/stcray_vqa_train_test.jsonl \
  --validate
```

#### 3.3. Full Dataset Generation

Once quality is validated:

```bash
# Generate full VQA dataset
# WARNING: This costs $300-2,800 depending on model
bash scripts/generate_full_vqa.sh
```

This will:
- Generate ~90,000 training VQA pairs (30k images × 3 pairs/image)
- Generate ~48,000 validation VQA pairs (16k images × 3 pairs/image)
- Time estimate: 8-15 hours
- Cost estimate:
  - Claude Sonnet: ~$600
  - GPT-4o: ~$920
  - GPT-4o-mini: ~$92

**Progress tracking:**
```bash
# Monitor progress (in separate terminal)
watch -n 10 "wc -l data/stcray_vqa_train.jsonl"

# Check for checkpoint files
ls -lh data/*checkpoint*
```

**Resume from checkpoint:**
If interrupted, the script automatically resumes from checkpoint.

### 4. Model Training

#### 4.1. Review Configuration

```bash
# View training config
cat configs/train_stcray.yaml
```

Key parameters:
- `model_name`: Qwen/Qwen2.5-VL-7B-Instruct
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 8
- `learning_rate`: 2e-4
- LoRA rank: 64

#### 4.2. Start Training

```bash
# Start training (multi-GPU automatic)
bash scripts/train.sh
```

**Monitor training:**
```bash
# In separate terminal
tensorboard --logdir outputs/qwen25vl_stcray_lora/logs

# Or watch logs
tail -f outputs/qwen25vl_stcray_lora/train.log
```

**Training time estimates:**
- 2 GPUs (V100): ~18-24 hours
- 4 GPUs (V100): ~9-12 hours
- 8 GPUs (A100): ~4-6 hours

**Resume training:**
```bash
# If interrupted
bash scripts/train.sh configs/train_stcray.yaml outputs/qwen25vl_stcray_lora/checkpoint-1000
```

### 5. Evaluation

#### 5.1. VQA Metrics

```bash
# Evaluate model performance
bash scripts/evaluate.sh
```

This measures:
- Exact match accuracy
- Item detection F1 score
- BLEU score (text quality)
- ROUGE score (text similarity)

**Expected results:**
```json
{
  "exact_match_accuracy": 0.72,
  "item_f1": 0.85,
  "bleu": 0.68,
  "rouge_l": 0.74
}
```

#### 5.2. Operational Benchmarks

```bash
# Benchmark latency and throughput
python evaluation/eval_operational.py \
  --model outputs/qwen25vl_stcray_lora \
  --test-file data/stcray_vqa_val.jsonl \
  --output results/operational_benchmarks.json
```

**Expected results:**
```json
{
  "latency_p50_ms": 145,
  "latency_p95_ms": 230,
  "throughput_batch1": 6.9,
  "throughput_batch8": 18.3,
  "gpu_memory_gb": 28.4
}
```

### 6. Inference Deployment

#### 6.1. Start API Server

```bash
# Start inference server
bash scripts/deploy_inference.sh
```

Server starts at: `http://localhost:8080`

**Check endpoints:**
```bash
# Health check
curl http://localhost:8080/health

# API docs (open in browser)
open http://localhost:8080/docs
```

#### 6.2. Test API

```bash
# Test with sample image
bash scripts/test_api.sh
```

**Manual test:**
```bash
# Prepare image
IMAGE_BASE64=$(base64 -i data/stcray/test/images/000000.jpg | tr -d '\n')

# Send request
curl -X POST http://localhost:8080/api/v1/inspect \
  -H "Content-Type: application/json" \
  -d "{
    \"scan_id\": \"TEST-001\",
    \"image_base64\": \"$IMAGE_BASE64\",
    \"declared_items\": [\"clothing\", \"electronics\"]
  }" | python -m json.tool
```

**Expected response:**
```json
{
  "scan_id": "TEST-001",
  "detected_items": ["knife", "scissors"],
  "risk_level": "high",
  "action": "manual_inspection",
  "reasoning": "Two prohibited sharp items detected: knife (center-left), scissors (upper-right). Items declared (clothing, electronics) do not match detected items.",
  "match_declaration": false,
  "processing_time_ms": 156
}
```

### 7. Production Deployment

#### Option A: Docker

```bash
# Build image
docker build -f deployment/Dockerfile -t xray-vqa-inference:latest .

# Run with Docker Compose
cd deployment
docker-compose up -d

# View logs
docker-compose logs -f xray-vqa-api
```

#### Option B: Kubernetes

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Upload model to persistent volume
kubectl apply -f deployment/kubernetes/deployment.yaml

# Copy model (one-time)
kubectl cp outputs/qwen25vl_stcray_lora \
  customs-ai/<pod>:/models/qwen25vl_stcray_lora

# Check status
kubectl get pods -n customs-ai
kubectl logs -f -n customs-ai deployment/xray-vqa-api
```

**Access API:**
```bash
# Port forward
kubectl port-forward -n customs-ai service/xray-vqa-api-service 8080:80

# Test
curl http://localhost:8080/health
```

### 8. Monitoring (Optional)

#### Enable Monitoring Stack

```bash
# Start Prometheus and Grafana
cd deployment
docker-compose --profile monitoring up -d
```

**Access dashboards:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

**Key metrics to monitor:**
- API request rate
- Inference latency (p50, p95, p99)
- GPU utilization
- GPU memory usage
- Error rate

## Complete Pipeline Script

For automated execution:

```bash
# Full pipeline in one command
bash scripts/run_full_pipeline.sh
```

This runs:
1. Dataset download
2. Test VQA generation (100 images)
3. Full VQA generation (interactive confirmation)
4. Model training
5. Evaluation

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:**
```yaml
# Reduce batch size in configs/train_stcray.yaml
per_device_train_batch_size: 1  # Instead of 2
gradient_accumulation_steps: 16  # Instead of 8
```

#### 2. LLM API Rate Limit

**Solution:**
```bash
# Increase delay in generate_full_vqa.sh
python data/llm_vqa_generator.py \
  --rate-limit-delay 2.0  # Instead of 1.0
```

#### 3. Dataset Download Fails

**Solution:**
```bash
# Login to HuggingFace (if dataset requires authentication)
huggingface-cli login

# Or download manually
python data/download_stcray.py --max-samples 1000  # Test with subset
```

#### 4. Training Crashes

**Check logs:**
```bash
# View last 100 lines
tail -n 100 outputs/qwen25vl_stcray_lora/train.log

# Check GPU status
nvidia-smi

# Verify dataset
python -c "
import json
with open('data/stcray_vqa_train.jsonl') as f:
    print(f'Lines: {sum(1 for _ in f)}')
"
```

#### 5. API Server Won't Start

**Debug:**
```bash
# Check model exists
ls -lh outputs/qwen25vl_stcray_lora/

# Test model loading
python -c "
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('outputs/qwen25vl_stcray_lora')
print('Model loaded successfully')
"

# Check port availability
lsof -i :8080
```

## Performance Optimization

### Training Speed

1. **Use more GPUs:**
   ```bash
   # Automatically uses all available GPUs
   python training/train_local.py --config configs/train_stcray.yaml
   ```

2. **Enable DeepSpeed:**
   ```yaml
   # configs/train_stcray.yaml
   deepspeed: configs/deepspeed_zero2.json
   ```

3. **Mixed precision:**
   ```yaml
   # Already enabled
   bf16: true
   ```

### Inference Speed

1. **Increase tensor parallelism:**
   ```bash
   TENSOR_PARALLEL_SIZE=4 bash scripts/deploy_inference.sh
   ```

2. **Use faster GPUs:**
   - V100 → A100 (2-3x faster)
   - Enable MIG for A100

3. **Batch requests:**
   Send multiple scans in parallel

## Cost Summary

### Development (Testing)

| Item | Cost |
|------|------|
| Test VQA generation (100 images) | $2-6 |
| Training (2x V100, 20h) | $40-60 |
| **Total** | **~$50** |

### Production (Full)

| Item | Cost |
|------|------|
| STCray download | Free |
| Full VQA generation (46k images) | $300-920 |
| Training (4x V100, 10h) | $80-120 |
| Inference (2x V100, 24/7) | $300-500/mo |
| **Total (one-time)** | **~$400-1,100** |
| **Total (monthly)** | **~$300-500** |

## Next Steps

1. **Improve accuracy:**
   - Generate more diverse VQA pairs (5-7 per image)
   - Fine-tune on domain-specific data
   - Ensemble multiple models

2. **Scale deployment:**
   - Set up auto-scaling (HPA)
   - Add load balancer
   - Implement caching

3. **Add features:**
   - Multi-language support
   - Explainable AI (attention maps)
   - Human-in-the-loop feedback

4. **Monitor and iterate:**
   - Collect real-world feedback
   - Retrain quarterly
   - A/B test improvements

## Resources

- **Documentation:** `README.md`, `ARCHITECTURE.md`, `QUICKSTART.md`
- **Examples:** `examples/vqa_dataset_samples.jsonl`
- **Dataset info:** `docs/DATASET_RECOMMENDATIONS.md`
- **LLM generation:** `docs/AI_AGENT_VQA_GENERATION.md`

## Support

For questions or issues:
1. Check troubleshooting section above
2. Review documentation in `docs/`
3. Examine logs and metrics
4. Consult project maintainer
