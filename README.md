# X-ray Scanning Model Training

AI-powered threat detection for X-ray baggage screening with OpenAI-compatible API for agentic workflows.

## Overview

This project provides **two approaches** for automated X-ray baggage inspection:

### YOLO Object Detection (Recommended)

**Fast, lightweight, production-ready detection**

- **Speed**: 20-100ms per image (real-time)
- **Size**: 11-47MB models (vs 14GB for VLM)
- **VRAM**: 2-8GB (vs 16GB+ for VLM)
- **Training**: 2-4 hours (vs days for VLM)
- **Use case**: Production screening, edge devices, high-throughput scenarios

**Key features:**
- Direct bounding box detection from STCray annotations
- Native Ultralytics or ONNX Runtime backends
- OpenAI-compatible API for agentic workflows
- 24 threat categories with confidence scores
- Occlusion detection for concealed items
- Ultra-fast CAI setup with `uv` (10-100x faster than pip)

### VLM (Vision-Language Model) - Alternative Approach

**Flexible, conversational, multi-task capable**

- Fine-tuned Qwen2.5-VL-7B-Instruct with VQA
- Natural language explanations and reasoning
- Structured JSON output with XGrammar
- Better for research, explanation, and complex queries

---

**Architecture**: Both approaches use a **separation of concerns**:
- **Detection Model**: Item recognition only (YOLO or VLM)
- **Post-Processing**: Declaration comparison, risk assessment, policy logic

**Benefits:**
- Focused model training (better accuracy)
- Flexible policy updates (no retraining)
- Transparent decision-making
- Easier testing and maintenance

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for system design.

## Project Structure

```
xray_scanning_model_finetuning/
├── README.md                       # This file
├── scripts/
│   └── setup_venv.sh              # Environment setup
├── setup/
│   ├── requirements.txt           # Python dependencies
│   └── ray_cluster_config.yaml    # Ray cluster config (Phase 2)
├── configs/
│   ├── train_local.yaml           # Phase 1 config
│   └── train_ray.yaml             # Phase 2 config
├── data/
│   ├── download_opixray.py        # Download dataset
│   ├── create_vqa_pairs.py        # Transform to VQA format
│   ├── declaration_simulator.py   # Generate synthetic declarations
│   └── split_dataset.py           # Train/val/test split
├── training/
│   ├── vqa_dataset.py             # PyTorch Dataset class
│   ├── train_local.py             # Phase 1 training (single-node)
│   └── train_ray.py               # Phase 2 training (distributed)
├── evaluation/
│   ├── eval_vqa.py                # VQA metrics (accuracy, F1, BLEU)
│   └── eval_operational.py        # Operational KPIs (latency, throughput)
├── inference/
│   ├── vllm_server.py             # vLLM inference engine
│   └── api_server.py              # FastAPI REST API
└── mlops/
    ├── monitoring_service.py      # Metrics collection
    └── drift_detector.py          # Drift detection
```

## Quick Start

### 🚀 Deploy to CAI via GitHub Actions (Recommended)

**Fine-tune YOLO on Cloudera AI Workspace:**

#### Web UI Method

1. Go to **Actions** tab → **Deploy X-ray Detection to CAI**
2. Click **Run workflow**
3. Select:
   - **model_type**: `yolo`
   - **dataset**: `luggage_xray` (recommended) or `cargoxray` (quick) or `stcray` (production)
   - **trigger_jobs**: `true`
4. Click **Run workflow**

#### CLI Method

```bash
# Recommended: Luggage X-ray (1 hour, 7k images)
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=luggage_xray \
  --field trigger_jobs=true

# Quick baseline: CargoXray (30 min, 659 images)
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray \
  --field trigger_jobs=true

# Production: STCray (5 hours, 46k images)
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=stcray \
  --field trigger_jobs=true

# Monitor progress
gh run watch
```

📖 **Complete guides:**
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** ← **Complete setup with Git LFS**
- [GitHub Actions Deployment](docs/GITHUB_ACTIONS_DEPLOYMENT.md)
- [CAI Fine-Tuning Manual](docs/CAI_YOLO_FINETUNING.md)
- [Git LFS Setup](docs/GIT_LFS_SETUP.md)

---

### Local Development - YOLO Approach

**1. Install Dependencies**

```bash
pip install ultralytics torch torchvision fastapi uvicorn pillow pyyaml tqdm
```

**2. Download & Process Data**

```bash
# Download STCray dataset (requires HuggingFace access)
# See docs/STCRAY_DOWNLOAD.md for setup instructions
huggingface-cli download Naoufel555/STCray-Dataset --local-dir data/stcray_raw

# Process to annotations format
./scripts/process_stcray_all.sh

# Convert to YOLO format
python3 data/convert_to_yolo_format.py \
    --annotations-dir data/stcray_processed \
    --output-dir data/yolo_dataset \
    --val-split 0.2
```

**3. Train Model**

```bash
# Train YOLOv8n (fastest, 2-3 hours on T4 GPU)
./scripts/train_yolo_local.sh --model yolov8n.pt --epochs 100

# Or train YOLOv8s (balanced, 4-5 hours)
./scripts/train_yolo_local.sh --model yolov8s.pt --epochs 100 --export-onnx
```

**4. Start API Server**

```bash
./scripts/serve_yolo_api.sh \
    --model runs/detect/xray_detection/weights/best.pt \
    --port 8000
```

**5. Test Detection**

```bash
# Test on sample images
python3 scripts/test_yolo_inference.py \
    --model runs/detect/xray_detection/weights/best.pt \
    --images data/stcray_raw/STCray_TestSet/Images/Class\ 11_Knife/*.jpg
```

See [`QUICKSTART.md`](QUICKSTART.md) for detailed guide.

---

### VLM Approach (Alternative)

**1. Environment Setup**

```bash
# Create virtual environment
bash scripts/setup_venv.sh
source .venv/bin/activate
```

**2. Generate VQA Data**

```bash
# Generate VQA pairs using Gemini 2.0 Flash (~$9, 1-2 hours)
export API_KEY="your-api-key"
./scripts/generate_vqa_gemini.sh
```

**3. Train VLM**

See [`docs/VQA_GENERATOR_VERIFICATION.md`](docs/VQA_GENERATOR_VERIFICATION.md) for VLM training.

```bash
# Create VQA pairs from dataset annotations
python data/create_vqa_pairs.py \
  --stcray-root data/stcray_processed \
  --split all \
  --samples-per-image 2
```

### 3. Training (Phase 1: Single-Node)

```bash
# Train on local multi-GPU with STCray dataset (automatic DDP)
python training/train_local.py --config configs/train_stcray.yaml

# Monitor training with TensorBoard
tensorboard --logdir outputs/qwen25vl_lora_stcray/logs
```

**Expected:** 6-8 hours on 4x24GB GPUs for 3 epochs with STCray (~138k VQA pairs)

### 4. Evaluation

```bash
# VQA metrics on STCray validation set
python evaluation/eval_vqa.py \
  --model outputs/qwen25vl_lora_stcray \
  --test-file data/stcray_vqa_val.jsonl \
  --output results/eval_vqa_results.json

# Operational benchmarks
python evaluation/eval_operational.py \
  --model outputs/qwen25vl_lora_stcray \
  --test-file data/stcray_vqa_val.jsonl \
  --batch-sizes 1,2,4,8,16,32 \
  --output results/operational_benchmarks.json
```

### 5. Inference Deployment

```bash
# Start vLLM server
python inference/vllm_server.py \
  --model outputs/qwen25vl_lora_phase1 \
  --tensor-parallel-size 2 \
  --port 8000

# In another terminal, start API server
python inference/api_server.py \
  --model outputs/qwen25vl_lora_phase1 \
  --host 0.0.0.0 \
  --port 8080

# Test API (in another terminal)
curl -X POST http://localhost:8080/api/v1/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "scan_id": "TEST-001",
    "image_base64": "...",
    "declared_items": ["clothing", "electronics"]
  }'
```

Access API docs at: http://localhost:8080/docs

## Phase 2: Distributed Training (Ray)

When you need to scale beyond a single machine:

### 1. Setup Ray Cluster

```bash
# Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# On worker nodes
ray start --address=<head-node-ip>:6379
```

Or use the provided config:

```bash
ray up setup/ray_cluster_config.yaml
```

### 2. Run Distributed Training

```bash
python training/train_ray.py \
  --config configs/train_ray.yaml \
  --ray-address auto \
  --num-workers 4
```

**Benefits over Phase 1:**
- Multi-node scaling (beyond single machine)
- Fault tolerance (automatic recovery)
- Better resource management
- Same model, same data, same results

## MLOps and Monitoring

### Monitoring

```bash
# Simulate monitoring (for testing)
python mlops/monitoring_service.py \
  --simulate \
  --duration 60 \
  --output metrics/monitoring_summary.json
```

**Metrics tracked:**
- Inference latency (P50, P95, P99)
- Throughput (images/second)
- Risk level distribution
- False positive/negative rates
- GPU utilization

**Export to Prometheus:**

```bash
python mlops/monitoring_service.py \
  --simulate \
  --prometheus-output metrics/prometheus_metrics.txt
```

### Drift Detection

```bash
# Detect drift between baseline and current data
python mlops/drift_detector.py \
  --baseline metrics/baseline_data.json \
  --current metrics/current_monitoring.json \
  --output metrics/drift_report.json
```

**Drift types detected:**
- **Input drift**: Image quality degradation, new scanner types
- **Prediction drift**: Risk level distribution shifts
- **Concept drift**: Accuracy drops over time (from feedback)

## Configuration

### Training Hyperparameters (`configs/train_local.yaml`)

```yaml
model_name: Qwen/Qwen2.5-VL-7B-Instruct
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-4
warmup_ratio: 0.03

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### LoRA Configuration

**Why LoRA?**
- Memory efficient (train 7B model on 2x24GB GPUs)
- Fast training (only 0.1-1% parameters updated)
- Easy merging and versioning
- vLLM compatible

**LoRA ranks:**
- `r=64`: Good balance (recommended)
- `r=32`: Faster, lower memory, slightly lower quality
- `r=128`: Higher quality, more memory

## Datasets

**📊 Full Comparison**: See [docs/DATASETS_COMPARISON.md](docs/DATASETS_COMPARISON.md) for detailed analysis

### Luggage X-ray (Recommended - Primary)
- **7,120 X-ray images** of luggage screening
- **12 categories**: 5 threats (Knife, Gun, Lighter, Powerbank, Grenade) + 7 normal items
- **OpenAI JSONL format** with bounding boxes
- **Best for**: YOLO training, balanced dataset, good for both model testing and production
- **Download**: Single curl command - see [docs/LUGGAGE_XRAY_CAI.md](docs/LUGGAGE_XRAY_CAI.md)

**Quick start:**
```bash
# Download dataset
curl -L "https://app.roboflow.com/ds/nMb0ckPbFf?key=EZzAfTucdZ" > roboflow.zip
unzip roboflow.zip && rm roboflow.zip

# Convert to YOLO format (parallel image downloads)
python scripts/convert_luggage_xray_to_yolo.py \
    --input-dir data/luggage_xray \
    --output-dir data/luggage_xray_yolo

# Train YOLO (1 hour on T4 GPU)
python training/train_yolo.py \
    --data data/luggage_xray_yolo/data.yaml \
    --model yolov8n.pt \
    --epochs 100
```

### CargoXray (Alternative - Quick Baseline)
- **659 X-ray images** of cargo containers (trucks, railcars)
- **16 object categories** (textiles, auto parts, tools, shoes, etc.)
- **Clearer images** with larger objects - easier baseline
- **Best for**: Quick testing, pipeline validation
- **Download**: Single curl command - see [docs/CARGOXRAY_QUICKSTART.md](docs/CARGOXRAY_QUICKSTART.md)

### STCray (Advanced - Production)
- **46,642 X-ray images** of baggage screening
- **21 threat categories** (knives, guns, liquids, explosives, etc.)
- **Best for**: Large-scale production deployment
- **Note**: Much larger dataset, longer training time

**Note:** Models are trained ONLY on item recognition. Declaration comparison and risk assessment happen in post-processing (`inference/postprocess.py`).

## Performance Targets

### Training (Phase 1)
- 4x24GB GPUs: ~6-8 hours for 3 epochs
- 8x24GB GPUs: ~3-4 hours for 3 epochs
- Target VQA accuracy: >80%

### Inference (vLLM)
- Latency: <500ms per image (P95)
- Throughput: >10 images/second (batch size 8)
- GPU memory: ~20GB for 7B model

### Operational KPIs
- False positive rate: <10%
- True positive rate: >90%
- Uptime: >99.9%

## API Reference

### POST `/api/v1/inspect`

Inspect X-ray scan and detect threats.

**Request:**
```json
{
  "scan_id": "SCAN-2026-001234",
  "image_base64": "base64_encoded_image",
  "declared_items": ["clothing", "electronics"],
  "mode": "vqa"
}
```

**Response:**
```json
{
  "scan_id": "SCAN-2026-001234",
  "risk_level": "high",
  "detected_items": [
    {
      "item": "folding knife",
      "confidence": 0.89,
      "location": "center-left",
      "occluded": true
    }
  ],
  "declaration_match": false,
  "reasoning": "Detected folding knife not declared. Item appears intentionally concealed.",
  "recommended_action": "PHYSICAL_INSPECTION",
  "processing_time_ms": 347
}
```

## Troubleshooting

### CUDA Out of Memory

**Solution 1:** Reduce batch size
```yaml
per_device_train_batch_size: 1  # Reduce from 2
gradient_accumulation_steps: 16  # Increase to maintain effective batch size
```

**Solution 2:** Use smaller LoRA rank
```yaml
lora:
  r: 32  # Reduce from 64
  alpha: 64  # Reduce proportionally
```

**Solution 3:** Enable gradient checkpointing (add to training script)

### vLLM Installation Issues

```bash
# Use pre-built wheels
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### Slow Data Loading

Increase num_workers:
```yaml
dataloader_num_workers: 8  # Increase from 4
```

## Production Deployment

### Kubernetes Deployment

```bash
# Build Docker image
docker build -t xray-inspection:latest -f inference/docker/Dockerfile .

# Deploy to Kubernetes
kubectl apply -f inference/kubernetes/deployment.yaml
```

### Horizontal Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: xray-inspection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: xray-inspection
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Contributing

### Code Style

```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

### Running Tests

```bash
pytest tests/
```

## License

This project is for customs and border control use. Please ensure compliance with local regulations.


## Support

For issues and questions:
- GitHub Issues: [Project Issues]
- Email: [Contact]
- Documentation: [Docs]

## Roadmap

**Phase 1 (Current):**
- ✅ Single-node training with LoRA
- ✅ vLLM inference deployment
- ✅ Basic MLOps (monitoring, drift detection)

**Phase 2 (Next):**
- ✅ Ray distributed training
- ⏳ Multi-scanner integration
- ⏳ Real-time alerting system

**Phase 3 (Future):**
- Domain adaptation with real JKDM data
- Multi-modal fusion (X-ray + thermal + visual)
- Active learning pipeline
- Federated learning across checkpoints

## Documentation

### Core Documentation
- [`README.md`](README.md) - Project overview and getting started
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - System architecture and design
- [`QUICKSTART.md`](QUICKSTART.md) - Quick start guide for Cloudera AI
- [`CHANGELOG.md`](CHANGELOG.md) - Project change log

### Verification & Quality
- **[`VERIFICATION_SUMMARY.md`](VERIFICATION_SUMMARY.md)** - LLM VQA generator verification summary
- [`docs/VQA_GENERATOR_VERIFICATION.md`](docs/VQA_GENERATOR_VERIFICATION.md) - Detailed verification report with usage examples

### Guides & References
- **[`docs/LOCAL_VQA_SETUP.md`](docs/LOCAL_VQA_SETUP.md)** - 🚀 Quick setup for local VQA generation (minimal dependencies)
- **[`docs/MACOS_LONG_RUNNING.md`](docs/MACOS_LONG_RUNNING.md)** - 🍎 Prevent MacBook sleep during VQA generation
- **[`docs/QWEN_VL_VLLM_GUIDE.md`](docs/QWEN_VL_VLLM_GUIDE.md)** - ⭐ Qwen2.5-VL + vLLM guide (FREE, RECOMMENDED)
- **[`docs/GEMINI_VQA_GENERATION.md`](docs/GEMINI_VQA_GENERATION.md)** - Gemini 2.0 Flash VQA generation (CHEAPEST CLOUD, ~$9)
- [`docs/AI_AGENT_VQA_GENERATION.md`](docs/AI_AGENT_VQA_GENERATION.md) - Vision LLM-based VQA generation (Claude/GPT-4V)
- [`docs/TEXT_LLM_VQA_GENERATION.md`](docs/TEXT_LLM_VQA_GENERATION.md) - Text LLM-based VQA generation (Qwen2.5-3B)
- [`docs/DATASET_RECOMMENDATIONS.md`](docs/DATASET_RECOMMENDATIONS.md) - Dataset analysis and recommendations
- [`docs/COMPLETE_WORKFLOW.md`](docs/COMPLETE_WORKFLOW.md) - End-to-end workflow guide
- [`examples/README_EXAMPLES.md`](examples/README_EXAMPLES.md) - VQA dataset format examples
- [`deployment/README.md`](deployment/README.md) - Deployment guide (Docker/Kubernetes)
