# GitHub Actions Deployment to CAI

Complete guide for deploying YOLO fine-tuning to Cloudera AI using GitHub Actions.

## Overview

The GitHub Actions workflow automates:
1. ✅ Code validation
2. ✅ CAI project setup
3. ✅ Job configuration & creation
4. ✅ Optional pipeline trigger

**No manual CAI setup required** - everything is automated!

---

## Prerequisites

### 1. GitHub Secrets Configuration

Add these secrets to your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add **New repository secret** for each:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `CML_HOST` | CAI workspace URL | `https://ml-abc123.gr-docpr.a465-9q4k.cloudera.site` |
| `CML_API_KEY` | CAI API key | `abc123...xyz` |
| `GH_PAT` | GitHub Personal Access Token (optional) | `ghp_...` |

#### How to Get CAI API Key

```bash
# Method 1: From CAI UI
1. Log into CAI Workspace
2. User Settings → API Keys
3. Generate New Key
4. Copy the key

# Method 2: From CLI
cml api-key create --name "github-actions"
```

### 2. Repository Setup

```bash
# Ensure you have the latest code
git pull origin main

# Verify workflow file exists
ls .github/workflows/deploy-to-cai.yml
```

---

## Quick Start

### Option 1: CargoXray (30 min - Recommended First)

**Use Case**: Quick validation, baseline testing

1. Go to **Actions** tab in GitHub
2. Select **Deploy X-ray Detection to CAI**
3. Click **Run workflow**
4. Configure:
   - **Model type**: `yolo`
   - **Dataset**: `cargoxray` ⭐
   - **YOLO model**: `yolov8n.pt`
   - **Epochs**: `100`
   - **Export ONNX**: `false`
   - **Trigger jobs**: `true` ⭐
5. Click **Run workflow**

**Expected Time**: 1 hour total
- Setup: 30 min
- Upload: 5 min
- Training: 30 min

### Option 2: STCray (4 hours - Production)

**Use Case**: Production model with 21 threat categories

1. Go to **Actions** tab in GitHub
2. Select **Deploy X-ray Detection to CAI**
3. Click **Run workflow**
4. Configure:
   - **Model type**: `yolo`
   - **Dataset**: `stcray` ⭐
   - **YOLO model**: `yolov8n.pt`
   - **Epochs**: `100`
   - **Export ONNX**: `false`
   - **Trigger jobs**: `true` ⭐
5. Click **Run workflow**

**Expected Time**: 5 hours total
- Setup: 30 min
- Download: 30 min
- Training: 4 hours

---

## Workflow Parameters

### Core Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| **model_type** | `yolo`, `vlm` | `yolo` | Model architecture |
| **dataset** | `cargoxray`, `stcray` | `stcray` | Dataset to train on |
| **yolo_model** | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov11n.pt` | `yolov8n.pt` | YOLO model variant |
| **yolo_epochs** | Any number | `100` | Training epochs |
| **export_onnx** | `true`, `false` | `false` | Export to ONNX |
| **force_reinstall** | `true`, `false` | `false` | Force env reinstall |
| **trigger_jobs** | `true`, `false` | `false` | Auto-trigger pipeline |

### Model Variants

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n.pt` | 6MB | Fastest (20ms) | Good (65% mAP) | Edge devices, real-time |
| `yolov8s.pt` | 22MB | Fast (35ms) | Better (70% mAP) | Balanced |
| `yolov8m.pt` | 52MB | Medium (50ms) | Best (75% mAP) | High accuracy |
| `yolov11n.pt` | 6MB | Fastest (18ms) | Good (68% mAP) | Latest, fastest |

---

## Using GitHub CLI

### Install GitHub CLI

```bash
# macOS
brew install gh

# Linux
sudo apt install gh

# Authenticate
gh auth login
```

### Deploy CargoXray

```bash
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray \
  --field yolo_model=yolov8n.pt \
  --field yolo_epochs=100 \
  --field export_onnx=false \
  --field trigger_jobs=true
```

### Deploy STCray

```bash
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=stcray \
  --field yolo_model=yolov8n.pt \
  --field yolo_epochs=100 \
  --field export_onnx=false \
  --field trigger_jobs=true
```

### Monitor Progress

```bash
# Watch the latest run
gh run watch

# List recent runs
gh run list --workflow=deploy-to-cai.yml

# View logs
gh run view
```

---

## Workflow Stages

### Stage 1: Validation (2 min)

```
✓ Checkout code
✓ Check model type configuration  
✓ Validate configuration files
✓ Validate YAML syntax
```

### Stage 2: Setup Project (5 min)

```
✓ Setup CAI project
✓ Configure Git integration
✓ Export PROJECT_ID
```

### Stage 3: Create Jobs (2 min)

```
✓ Update jobs config with parameters
✓ Create CAI jobs (3 jobs):
  - setup_environment
  - download_dataset (or upload_cargoxray)
  - yolo_training
```

### Stage 4: Trigger Pipeline (Optional)

```
✓ Trigger setup_environment job
✓ Child jobs auto-trigger via CAI dependencies
```

---

## Monitoring in CAI

### 1. View Jobs

```
CAI UI → Jobs → Job Definitions
```

You'll see 3 jobs:
- `setup_environment` (root)
- `download_dataset` or `upload_cargoxray`
- `yolo_training`

### 2. Monitor Execution

```
CAI UI → Jobs → Job Runs
```

Track progress:
- ✅ **Green**: Success
- 🔵 **Blue**: Running
- 🔴 **Red**: Failed
- ⚪ **Gray**: Pending

### 3. View Logs

Click on any job run → **Logs** tab

### 4. Download Results

After training completes:

```bash
# SSH into CAI
ssh <your-cai-workspace>

# Navigate to results
cd /home/cdsw/runs/detect/

# List trained models
ls -lh */weights/best.pt

# Download locally
scp <cai>:/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt ./models/
```

---

## Troubleshooting

### Issue 1: Workflow Fails at Validation

```
Error: cai_integration/jobs_config_yolo.yaml not found
```

**Fix:**
- Ensure all code is committed and pushed
- Check file exists: `ls cai_integration/jobs_config_yolo.yaml`

### Issue 2: CAI API Authentication Failed

```
Error: 401 Unauthorized
```

**Fix:**
1. Verify `CML_HOST` is correct (include `https://`)
2. Regenerate `CML_API_KEY` in CAI
3. Update GitHub secret

### Issue 3: Job Creation Failed

```
Error: Project not found
```

**Fix:**
- Check `CML_HOST` points to correct workspace
- Ensure API key has project creation permissions
- Try manually in CAI first to verify permissions

### Issue 4: Training Job Timeout

```
Job exceeded timeout limit
```

**Fix:**
- For STCray, 4 hours might not be enough on slower GPUs
- Edit `jobs_config_yolo.yaml`:
  ```yaml
  yolo_training:
    timeout: 18000  # 5 hours
  ```
- Commit and re-run workflow

### Issue 5: Out of Memory During Training

```
RuntimeError: CUDA out of memory
```

**Fix:**
- Use smaller batch size:
  ```bash
  --field yolo_model=yolov8n.pt  # smallest model
  ```
- Edit job after creation in CAI:
  - Set `BATCH_SIZE: "8"` (instead of 16)
  - Set `IMG_SIZE: "416"` (instead of 640)

---

## Advanced Usage

### Custom Training Parameters

Edit workflow inputs:

```yaml
# Longer training for better accuracy
yolo_epochs: '200'

# Larger model for production
yolo_model: 'yolov8m.pt'

# Export ONNX for faster inference
export_onnx: true
```

### Transfer Learning

**Step 1**: Train on CargoXray

```bash
gh workflow run deploy-to-cai.yml \
  --field dataset=cargoxray \
  --field trigger_jobs=true
```

**Step 2**: After completion, update `jobs_config_yolo.yaml`:

```yaml
yolo_training:
  environment:
    MODEL_NAME: "/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt"
    DATASET: "stcray"
    EPOCHS: "50"  # Fine-tuning requires fewer epochs
```

**Step 3**: Re-run workflow with STCray

### Multiple Model Variants

Train multiple models in parallel:

```bash
# Terminal 1: Fast model
gh workflow run deploy-to-cai.yml \
  --field yolo_model=yolov8n.pt \
  --field trigger_jobs=true

# Terminal 2: Accurate model  
gh workflow run deploy-to-cai.yml \
  --field yolo_model=yolov8m.pt \
  --field trigger_jobs=true
```

---

## Cost Estimation

### CargoXray

| Stage | Duration | GPU | Cost |
|-------|----------|-----|------|
| Setup | 30 min | 0 | $0.50 |
| Upload | 5 min | 0 | $0.10 |
| Training | 30 min | 1 | $2.00 |
| **Total** | **1 hour** | **0.5h GPU** | **~$2.60** |

### STCray

| Stage | Duration | GPU | Cost |
|-------|----------|-----|------|
| Setup | 30 min | 0 | $0.50 |
| Download | 30 min | 0 | $0.50 |
| Training | 4 hours | 1 | $16.00 |
| **Total** | **5 hours** | **4h GPU** | **~$17.00** |

---

## CI/CD Integration

### Automatic Deployment on Push

Add to `.github/workflows/deploy-to-cai.yml`:

```yaml
on:
  push:
    branches:
      - main
    paths:
      - 'training/**'
      - 'cai_integration/**'
  workflow_dispatch:
    # ... existing inputs
```

This auto-deploys when training code changes.

### Nightly Training

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:
    # ... existing inputs
```

### Pull Request Validation

```yaml
on:
  pull_request:
    paths:
      - 'training/**'
      - 'cai_integration/**'
```

Validates configuration before merging.

---

## Next Steps

### After Successful Deployment

1. ✅ **Verify jobs created** in CAI UI
2. ✅ **Monitor training** progress
3. ✅ **Download trained model** after completion
4. ✅ **Test locally**:
   ```bash
   python inference/yolo_api_server.py --model models/best.pt
   ```
5. ✅ **Deploy to production** (CAI Application)

### Testing the Model

```bash
# Download model
scp cai:/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt ./models/

# Test inference
python scripts/test_yolo_inference.py \
  --model models/best.pt \
  --images data/cargoxray/test/*.jpg

# Start API server
python inference/yolo_api_server.py \
  --model models/best.pt \
  --port 8000
```

---

## Summary

**Quick Deployment:**

```bash
# 1. CargoXray baseline (1 hour)
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray \
  --field trigger_jobs=true

# 2. Monitor
gh run watch

# 3. Download model
scp cai:/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt .

# 4. Deploy API
python inference/yolo_api_server.py --model best.pt
```

**Expected Results:**
- ✅ Model trained in 30 min
- ✅ mAP@0.5: 0.70-0.80
- ✅ OpenAI-compatible API ready
- ✅ Inference: 20-30ms per image

---

## Resources

- **CAI Fine-Tuning Guide**: [docs/CAI_YOLO_FINETUNING.md](CAI_YOLO_FINETUNING.md)
- **YOLO Training Guide**: [docs/YOLO_TRAINING.md](YOLO_TRAINING.md)
- **API Documentation**: [docs/YOLO_API.md](YOLO_API.md)
- **GitHub Actions Docs**: https://docs.github.com/en/actions

---

**Questions?** Open an issue or check the main [README.md](../README.md)!
