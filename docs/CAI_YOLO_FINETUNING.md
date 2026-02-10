# YOLO Fine-Tuning on Cloudera AI Workspace

Complete guide for fine-tuning YOLO models for X-ray detection on CAI.

## Quick Summary

**Given RolmOCR's poor performance (0% accuracy), we're switching to YOLO detection:**
- ✅ YOLO is purpose-built for object detection
- ✅ Fast training (2-4 hours vs days)
- ✅ Lightweight models (11-47MB vs 14GB)
- ✅ High accuracy on X-ray images
- ✅ OpenAI-compatible API ready

---

## Prerequisites

### On Your Local Machine

1. **Git repository** with all code committed
2. **GitHub Actions secrets** configured (CAI credentials)
3. **Dataset choice**:
   - **CargoXray** (659 images, 30 min training) - Quick baseline
   - **STCray** (46k images, 4 hours training) - Production model

### On CAI Workspace

1. **GPU access** (1x GPU minimum)
2. **Storage**: 50GB for STCray, 5GB for CargoXray
3. **Python 3.10+** runtime

---

## Option 1: Quick Start - CargoXray (Recommended First)

### Why Start with CargoXray?

- ✅ **Fast download** (1 minute vs 30 minutes)
- ✅ **Quick training** (30 minutes vs 4 hours)
- ✅ **Validates pipeline** before committing to large dataset
- ✅ **Simpler images** for baseline testing

### Step 1: Prepare CargoXray Data Locally

```bash
# Download CargoXray (83MB, 1 minute)
cd data/cargoxray
curl -L "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6" > roboflow.zip
unzip roboflow.zip && rm roboflow.zip
cd ../..

# Convert to YOLO format
python scripts/convert_cargoxray_to_yolo.py \
  --input-dir data/cargoxray \
  --output-dir data/cargoxray_yolo
```

**Output:** `data/cargoxray_yolo/` ready for training

### Step 2: Update CAI Job Configuration

Create `cai_integration/jobs_config_cargoxray.yaml`:

```yaml
jobs:
  setup_environment:
    name: "Setup Python Environment"
    script: "cai_integration/setup_environment.py"
    cpu: 4
    memory: 16
    timeout: 3600
    parent_job_key: null

  upload_cargoxray:
    name: "Upload CargoXray Dataset"
    description: "Upload pre-converted CargoXray YOLO dataset"
    script: "cai_integration/upload_cargoxray.py"
    cpu: 2
    memory: 8
    timeout: 1800
    parent_job_key: "setup_environment"

  train_yolo_cargo:
    name: "Train YOLO on CargoXray"
    script: "cai_integration/train_yolo_cargo.py"
    cpu: 8
    memory: 32
    gpu: 1
    timeout: 3600  # 1 hour (CargoXray is small)
    parent_job_key: "upload_cargoxray"
    environment:
      MODEL_NAME: "yolov8n.pt"
      EPOCHS: "100"
      BATCH_SIZE: "16"
      IMG_SIZE: "640"
      DATA_YAML: "data/cargoxray_yolo/data.yaml"
```

### Step 3: Deploy to CAI

**Option A: GitHub Actions (Automated)**

```bash
# Commit changes
git add .
git commit -m "feat: Add CargoXray fine-tuning job"
git push

# Trigger workflow
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray
```

**Option B: Manual Upload**

```bash
# 1. Upload code to CAI workspace
rsync -av --exclude='.git' --exclude='data/stcray_raw' \
  . cai:/home/cdsw/xray_scanning/

# 2. SSH into CAI
ssh cai

# 3. Run setup
cd /home/cdsw/xray_scanning
python cai_integration/setup_environment.py

# 4. Upload dataset
rsync -av data/cargoxray_yolo/ /home/cdsw/data/cargoxray_yolo/

# 5. Run training
python training/train_yolo.py \
  --data /home/cdsw/data/cargoxray_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --name cargoxray_baseline
```

### Step 4: Monitor Training

**In CAI UI:**
1. Navigate to **Jobs** → **train_yolo_cargo**
2. View logs in real-time
3. Check progress: `runs/detect/cargoxray_baseline/`

**Via SSH:**
```bash
# Watch training progress
tail -f runs/detect/cargoxray_baseline/train.log

# Check GPU usage
nvidia-smi -l 1

# View training metrics
tensorboard --logdir runs/detect/cargoxray_baseline
```

### Step 5: Results

**Expected after 30 minutes:**
- ✅ Model trained: `runs/detect/cargoxray_baseline/weights/best.pt`
- ✅ Metrics: `runs/detect/cargoxray_baseline/results.csv`
- ✅ Confusion matrix: `runs/detect/cargoxray_baseline/confusion_matrix.png`

**Expected Performance:**
- **mAP@0.5**: 0.70-0.80 (cargo is easier than baggage)
- **Inference**: 20-30ms per image
- **Model size**: 6MB (YOLOv8n)

---

## Option 2: Production - STCray (Full Dataset)

### Why STCray?

- ✅ **46,642 images** - comprehensive training data
- ✅ **21 threat categories** - real airport screening
- ✅ **Production-ready** for deployment
- ⚠️ **4 hours training** (vs 30 min for CargoXray)

### Step 1: Ensure STCray is Downloaded

```bash
# Check if STCray exists
ls data/stcray_raw/STCray_TestSet/Images/

# If not, download (30 minutes)
huggingface-cli download Naoufel555/STCray-Dataset --local-dir data/stcray_raw

# Process
./scripts/process_stcray_all.sh
```

### Step 2: Use Existing YOLO Job Configuration

The default `cai_integration/jobs_config_yolo.yaml` is already configured for STCray:

```yaml
jobs:
  setup_environment: ...
  
  download_dataset:
    name: "Download STCray Dataset"
    script: "cai_integration/download_dataset.py"
    ...
  
  yolo_training:
    name: "Train YOLO Detection Model"
    script: "cai_integration/yolo_training.py"
    gpu: 1
    timeout: 14400  # 4 hours
    environment:
      MODEL_NAME: "yolov8n.pt"
      EPOCHS: "100"
      BATCH_SIZE: "16"
      IMG_SIZE: "640"
```

### Step 3: Deploy to CAI

**GitHub Actions:**

```bash
# Commit and push
git push

# Trigger workflow for STCray
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=stcray
```

### Step 4: Monitor (4 hours)

Check progress every 30 minutes:

```bash
ssh cai "tail -50 runs/detect/yolo_v1/train.log"
```

### Step 5: Results

**Expected after 4 hours:**
- ✅ Model: `runs/detect/yolo_v1/weights/best.pt` (6-12MB)
- ✅ **mAP@0.5**: 0.60-0.70 (baggage is harder)
- ✅ **Inference**: 20-30ms per image
- ✅ Ready for API deployment

---

## Model Variants

Choose based on your needs:

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| **yolov8n** | 6MB | 25ms | 0.65 | Fast inference, edge devices |
| **yolov8s** | 22MB | 35ms | 0.70 | Balanced performance |
| **yolov8m** | 52MB | 50ms | 0.75 | High accuracy |
| **yolov11n** | 6MB | 20ms | 0.68 | Latest, fastest |

**To change model:**

```yaml
environment:
  MODEL_NAME: "yolov8s.pt"  # or yolov8m.pt, yolov11n.pt
```

---

## Transfer Learning (Best Performance)

Train on CargoXray first, then fine-tune on STCray:

### Step 1: Train on CargoXray (30 min)

```bash
python training/train_yolo.py \
  --data data/cargoxray_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --name stage1_cargo
```

### Step 2: Fine-tune on STCray (2 hours)

```bash
python training/train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model runs/detect/stage1_cargo/weights/best.pt \
  --epochs 50 \
  --name stage2_stcray
```

**Expected improvement:** 10-15% better mAP than training from scratch

---

## CAI Job Structure

### 3-Job Pipeline

```
setup_environment (1 hour)
    ↓
download_dataset (30 min for CargoXray, 30 min for STCray)
    ↓
yolo_training (30 min for CargoXray, 4 hours for STCray)
```

### Resource Requirements

| Job | CPU | Memory | GPU | Timeout |
|-----|-----|--------|-----|---------|
| setup_environment | 4 | 16GB | 0 | 1h |
| download_dataset | 4 | 8GB | 0 | 1h |
| yolo_training (CargoXray) | 8 | 32GB | 1 | 1h |
| yolo_training (STCray) | 8 | 32GB | 1 | 4h |

---

## After Training: Deploy API

### Option 1: Local Testing

```bash
# Download trained model from CAI
scp cai:/home/cdsw/runs/detect/yolo_v1/weights/best.pt models/

# Start API server
python inference/yolo_api_server.py \
  --model models/best.pt \
  --host 0.0.0.0 \
  --port 8000
```

### Option 2: CAI Application

Create persistent API server on CAI:

1. **Create Application** in CAI UI
2. **Command**: `python inference/yolo_api_server.py --model /home/cdsw/runs/detect/yolo_v1/weights/best.pt`
3. **Resources**: 2 CPU, 8GB RAM
4. **Port**: 8000

### Option 3: ONNX Export for Production

```bash
# Export to ONNX (faster inference)
python training/train_yolo.py \
  --export-onnx \
  --model runs/detect/yolo_v1/weights/best.pt

# Served with ONNX Runtime
python inference/yolo_api_server.py \
  --model runs/detect/yolo_v1/weights/best.onnx \
  --backend onnxruntime
```

---

## Testing the Trained Model

### Test Inference

```bash
# Test on sample images
python scripts/test_yolo_inference.py \
  --model runs/detect/yolo_v1/weights/best.pt \
  --images data/cargoxray/test/*.jpg \
  --output test_results/yolo_cargoxray
```

### Test API

```bash
# Start server
python inference/yolo_api_server.py --model runs/detect/yolo_v1/weights/best.pt

# Test OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo-xray",
    "messages": [{"role": "user", "content": "Analyze image"}],
    "image": "<base64_encoded_image>"
  }'
```

---

## Troubleshooting

### Issue 1: Out of Memory

```
RuntimeError: CUDA out of memory
```

**Fix:**
- Reduce batch size: `BATCH_SIZE: "8"`
- Reduce image size: `IMG_SIZE: "416"`
- Use smaller model: `MODEL_NAME: "yolov8n.pt"`

### Issue 2: Job Timeout

```
Job exceeded timeout limit
```

**Fix:**
- Increase timeout: `timeout: 18000` (5 hours)
- Reduce epochs: `EPOCHS: "50"`
- Use smaller dataset first (CargoXray)

### Issue 3: No GPU Available

```
No CUDA GPUs are available
```

**Fix:**
- Request GPU in job config: `gpu: 1`
- Check CAI GPU availability
- Train on CPU (slower): Set `device: "cpu"`

---

## Cost Estimation

### CargoXray Training

| Resource | Duration | Cost Estimate |
|----------|----------|---------------|
| Setup | 30 min | $0.50 |
| Upload | 5 min | $0.10 |
| Training (1x GPU) | 30 min | $2.00 |
| **Total** | **1 hour** | **~$2.60** |

### STCray Training

| Resource | Duration | Cost Estimate |
|----------|----------|---------------|
| Setup | 30 min | $0.50 |
| Download | 30 min | $0.50 |
| Training (1x GPU) | 4 hours | $16.00 |
| **Total** | **5 hours** | **~$17.00** |

---

## Next Steps

### After CargoXray Training (30 min)

1. ✅ **Validate** model works (mAP > 0.70)
2. ✅ **Test API** locally
3. ⬜ **Scale to STCray** (4 hours)

### After STCray Training (4 hours)

1. ✅ **Evaluate** on test set
2. ✅ **Deploy API** to CAI Application
3. ✅ **Integrate** with agentic workflow
4. ✅ **Monitor** performance in production

---

## Summary

**Recommended Path:**

```
Week 1: CargoXray (30 min)
    ↓ Validate pipeline works
Week 2: STCray (4 hours)
    ↓ Production model
Week 3: Deploy & Integrate
```

**Quick Commands:**

```bash
# 1. CargoXray baseline
gh workflow run deploy-to-cai.yml --field dataset=cargoxray

# 2. STCray production
gh workflow run deploy-to-cai.yml --field dataset=stcray

# 3. Deploy API
# (Create CAI Application with trained model)
```

---

## Resources

- **Training Guide**: [docs/YOLO_TRAINING.md](YOLO_TRAINING.md)
- **API Guide**: [docs/YOLO_API.md](YOLO_API.md)
- **Dataset Comparison**: [docs/DATASETS_COMPARISON.md](DATASETS_COMPARISON.md)
- **CargoXray Quickstart**: [docs/CARGOXRAY_QUICKSTART.md](CARGOXRAY_QUICKSTART.md)

---

**Questions?** Check the main [README.md](../README.md) or open an issue!
