# Quick Start Guide - Cloudera AI Workspace

This guide provides step-by-step instructions to get started with X-ray VQA fine-tuning in Cloudera AI Workspace.

## Prerequisites

- Cloudera AI Workspace with GPU access
- Python 3.10+
- At least 2x 24GB GPUs for training
- ~100GB storage for dataset and model

## Choose Your Approach

### YOLO Detection (Recommended for Production)
- **Fast**: 20-100ms inference, 2-4 hours training
- **Lightweight**: 11-47MB models, 2-8GB VRAM
- **Dataset**: STCray (46k images, 21 classes)
- **Guide**: See [docs/YOLO_TRAINING.md](docs/YOLO_TRAINING.md)

### VLM Approach (Advanced, Research)
- **Flexible**: Natural language reasoning
- **Large**: 14GB model, 16GB+ VRAM, days of training
- **Dataset**: OPIXray (8k images, 5 classes)
- **Guide**: Follow this quickstart

---

## Step-by-Step Setup (VLM Approach)

### 1. Clone/Upload Project to Workspace

```bash
# If using git
git clone <your-repo-url>
cd xray_scanning_model_finetuning

# Or upload the project directory to your workspace
```

### 2. Create Virtual Environment

```bash
# Create and activate virtual environment
bash scripts/setup_venv.sh

# Activate
source .venv/bin/activate

# Verify GPU access
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### 3. Download OPIXray Dataset

```bash
# This will provide manual download instructions
python data/download_opixray.py --output-dir data/opixray

# After downloading, verify the dataset
python data/download_opixray.py --verify --output-dir data/opixray
```

**Note:** OPIXray dataset needs to be downloaded manually from:
- GitHub: https://github.com/OPIXray-author/OPIXray
- Google Drive: (check repository README)

Expected structure:
```
data/opixray/
├── images/
│   ├── P00001.jpg
│   ├── P00002.jpg
│   └── ...
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### 4. Prepare VQA Dataset

```bash
# Generate VQA pairs from COCO annotations
python data/create_vqa_pairs.py \
  --opixray-root data/opixray \
  --split all \
  --samples-per-image 2

# This creates:
# - data/opixray_vqa_train.jsonl
# - data/opixray_vqa_val.jsonl
# - data/opixray_vqa_test.jsonl

# Add synthetic customs declarations
for split in train val test; do
  python data/declaration_simulator.py \
    --input data/opixray_vqa_${split}.jsonl \
    --output data/opixray_vqa_${split}_final.jsonl \
    --match-ratio 0.5
done

# Update config to use final files
sed -i 's/opixray_vqa_train.jsonl/opixray_vqa_train_final.jsonl/g' configs/train_local.yaml
sed -i 's/opixray_vqa_val.jsonl/opixray_vqa_val_final.jsonl/g' configs/train_local.yaml
```

### 5. Start Training (Phase 1)

```bash
# Check GPU availability
nvidia-smi

# Start training (will automatically use all available GPUs)
python training/train_local.py --config configs/train_local.yaml

# Monitor in another terminal
tensorboard --logdir outputs/qwen25vl_lora_phase1/logs --host 0.0.0.0 --port 6006
```

**Training Duration:** ~6-8 hours on 4x24GB GPUs

**To resume from checkpoint:**
```bash
python training/train_local.py \
  --config configs/train_local.yaml \
  --resume-from-checkpoint outputs/qwen25vl_lora_phase1/checkpoint-500
```

### 6. Evaluate Model

```bash
# VQA metrics (accuracy, F1, BLEU, ROUGE)
python evaluation/eval_vqa.py \
  --model outputs/qwen25vl_lora_phase1 \
  --test-file data/opixray_vqa_test_final.jsonl \
  --output results/eval_vqa_results.json \
  --max-samples 100  # For quick test

# Operational benchmarks (latency, throughput)
python evaluation/eval_operational.py \
  --model outputs/qwen25vl_lora_phase1 \
  --test-file data/opixray_vqa_test_final.jsonl \
  --batch-sizes 1,2,4,8 \
  --output results/operational_benchmarks.json
```

### 7. Test Inference

```bash
# Test vLLM server (for quick validation)
python inference/vllm_server.py \
  --model outputs/qwen25vl_lora_phase1 \
  --test-mode

# Start API server for production-like testing
python inference/api_server.py \
  --model outputs/qwen25vl_lora_phase1 \
  --host 0.0.0.0 \
  --port 8080
```

Access API documentation: http://<workspace-url>:8080/docs

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Solution:**
```yaml
# Edit configs/train_local.yaml
per_device_train_batch_size: 1  # Reduce from 2
gradient_accumulation_steps: 16  # Increase to maintain effective batch size
```

### Issue 2: Slow Data Loading

**Solution:**
```yaml
# Edit configs/train_local.yaml
dataloader_num_workers: 0  # Disable multi-process loading in Cloudera
```

### Issue 3: vLLM Installation Fails

**Solution:**
```bash
# Install with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### Issue 4: Module Import Errors

**Solution:**
```bash
# Ensure packages are in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Resource Requirements

### Training (Phase 1)
- **GPUs:** 2-4x 24GB (minimum)
- **RAM:** 64GB (minimum)
- **Storage:** 100GB
- **Time:** 6-8 hours

### Inference (vLLM)
- **GPUs:** 1-2x 24GB
- **RAM:** 32GB
- **Latency:** <500ms per image

## Expected Results

### Training Metrics
- Training loss: Should decrease to <1.0
- Validation loss: Should stabilize around 0.8-1.2
- Training time: 6-8 hours (4 GPUs)

### Evaluation Metrics
- VQA Accuracy: >75% (target: >80%)
- F1 Score: >0.80
- BLEU: >0.60
- Inference Latency (P95): <500ms

## Next Steps After Phase 1

1. **Analyze Results:**
   ```bash
   # View evaluation report
   cat results/eval_vqa_results.json
   ```

2. **Tune Hyperparameters (if needed):**
   - Increase epochs: `num_train_epochs: 5`
   - Adjust learning rate: `learning_rate: 1e-4`
   - Larger LoRA rank: `lora.r: 128`

3. **Deploy to Production:**
   ```bash
   # Start production API server
   python inference/api_server.py \
     --model outputs/qwen25vl_lora_phase1 \
     --workers 2 \
     --tensor-parallel-size 2
   ```

4. **Scale to Phase 2 (Ray):**
   ```bash
   # Setup Ray cluster
   ray start --head --port=6379
   
   # Run distributed training
   python training/train_ray.py \
     --config configs/train_ray.yaml \
     --num-workers 8
   ```

## Monitoring and MLOps

### Monitor Model Performance
```bash
# Simulate monitoring
python mlops/monitoring_service.py \
  --simulate \
  --duration 300 \
  --output metrics/monitoring_summary.json
```

### Detect Drift
```bash
# After collecting baseline metrics
python mlops/drift_detector.py \
  --baseline metrics/baseline_data.json \
  --current metrics/current_monitoring.json \
  --output metrics/drift_report.json
```

## File Locations in Cloudera Workspace

```
/home/cdsw/xray_scanning_model_finetuning/
├── data/opixray/              # Dataset (you download)
├── outputs/                   # Training outputs
│   └── qwen25vl_lora_phase1/  # Fine-tuned model
├── results/                   # Evaluation results
├── metrics/                   # Monitoring data
└── .venv/                     # Virtual environment
```

## Useful Commands

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Monitor Training Progress
```bash
tail -f outputs/qwen25vl_lora_phase1/logs/events.out.tfevents.*
```

### Kill Stuck Process
```bash
pkill -9 -f train_local.py
```

### Check Disk Usage
```bash
du -sh data/ outputs/ results/
```

## Support

For issues specific to:
- **Dataset:** Check OPIXray repository issues
- **Training:** Review logs in `outputs/qwen25vl_lora_phase1/logs/`
- **GPU errors:** Check CUDA compatibility with `torch.cuda.is_available()`
- **vLLM:** Check vLLM documentation and GitHub issues

## Checklist

Before starting training, ensure:
- [ ] GPU access verified (`nvidia-smi` works)
- [ ] Virtual environment activated
- [ ] OPIXray dataset downloaded and verified
- [ ] VQA dataset created (~17k samples)
- [ ] Declarations added to VQA samples
- [ ] Config file paths are correct
- [ ] At least 100GB free disk space
- [ ] TensorBoard accessible (optional)

After training, verify:
- [ ] Training completed without errors
- [ ] Model saved in `outputs/qwen25vl_lora_phase1/`
- [ ] Evaluation metrics generated
- [ ] VQA accuracy >75%
- [ ] Inference latency <500ms

Ready to deploy:
- [ ] API server starts successfully
- [ ] Can send test requests
- [ ] Responses are reasonable
- [ ] Monitoring is configured

---

**Last Updated:** 2026-02-05
**Version:** 1.0.0
