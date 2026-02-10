# Complete Deployment Guide: Git LFS + GitHub Actions

**One-time setup** for automated YOLO fine-tuning on CAI.

---

## Overview

This guide sets up:
1. ✅ **Git LFS** - Store CargoXray dataset in repo (~300MB)
2. ✅ **GitHub Actions** - Automated deployment to CAI
3. ✅ **Manual trigger** - Create jobs without auto-running

**Result:** One command to deploy fine-tuning jobs to CAI!

---

## Quick Start (3 Steps)

### Step 1: Setup Git LFS & Push Dataset (10 min)

```bash
# Run automated setup script
./scripts/setup_git_lfs.sh

# This will:
# - Install Git LFS
# - Download CargoXray (83MB)
# - Convert to YOLO format
# - Commit & push to GitHub (~300MB via LFS)
```

### Step 2: Configure GitHub Secrets (2 min)

Add secrets in **GitHub → Settings → Secrets → Actions**:

| Secret | Value | Where to Get |
|--------|-------|--------------|
| `CML_HOST` | `https://ml-xyz.cloudera.site` | Your CAI workspace URL |
| `CML_API_KEY` | `abc123...` | CAI → User Settings → API Keys |

### Step 3: Deploy via GitHub Actions (1 min)

**Option A: Web UI (Easiest)**

```
1. Go to GitHub → Actions tab
2. Select "Deploy X-ray Detection to CAI"
3. Click "Run workflow"
4. Configure:
   - model_type: yolo
   - dataset: cargoxray
   - trigger_jobs: false ← Manual trigger (recommended)
5. Click "Run workflow"
```

**Option B: CLI**

```bash
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray \
  --field trigger_jobs=false
```

**Result:** Jobs created in CAI (not started automatically)

---

## What Gets Created

### GitHub Actions (5 min)

```
✓ Validate code
✓ Setup CAI project
✓ Create 3 jobs in CAI:
  1. setup_environment
  2. upload_cargoxray (uses Git LFS data)
  3. yolo_training
✓ Jobs ready to run manually
```

### CAI Workspace

```
Project: xray-scanning-model-finetuning
├── setup_environment (pending) ← Uses uv for 10-100x faster package installation
├── upload_cargoxray (pending)
└── yolo_training (pending)
```

**Performance Note:** Environment setup now uses `uv` (ultra-fast Python package installer) instead of `pip`, reducing setup time from ~15 minutes to ~2-3 minutes.

---

## Running the Training (Manual Trigger)

### Method 1: CAI Web UI

```
1. Go to CAI → Jobs
2. Find "setup_environment"
3. Click "Run"
4. Child jobs auto-trigger after parent completes
```

### Method 2: CAI API

```bash
# Get job ID from CAI UI or API
cml jobs run --job-id <job_id>
```

### Method 3: Auto-trigger via GitHub Actions

```bash
# Re-run workflow with trigger_jobs=true
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray \
  --field trigger_jobs=true  # ← Auto-start
```

---

## Workflow Parameters

### Essential

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `model_type` | yolo, vlm | yolo | Model architecture |
| `dataset` | cargoxray, stcray | stcray | Dataset to use |
| `trigger_jobs` | true, false | **false** | Auto-start training |

### YOLO-Specific

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `yolo_model` | yolov8n.pt, yolov8s.pt, yolov8m.pt | yolov8n.pt | Model variant |
| `yolo_epochs` | Any number | 100 | Training epochs |
| `export_onnx` | true, false | false | Export to ONNX |

---

## Git LFS Benefits

### Before (Without LFS)

```
GitHub Actions
    ↓
Setup CAI project
    ↓
Download dataset from Roboflow (slow)
    ↓
Upload to CAI (slow)
    ↓
Train
```

### After (With LFS)

```
GitHub Actions
    ↓
Checkout code + LFS files (fast)
    ↓
Setup CAI project
    ↓
Dataset already in repo ✓
    ↓
Train
```

**Time saved:** ~5-10 minutes per deployment

---

## Cost Breakdown

### GitHub LFS (Free Tier)

```
Storage: 1GB (CargoXray uses ~300MB)
Bandwidth: 1GB/month
Cost: FREE

Paid upgrade: $5/month for 50GB
```

### CAI Training

```
CargoXray Training:
- Setup: 30 min (CPU) = $0.50
- Training: 30 min (GPU) = $2.00
- Total: ~$2.50

STCray Training:
- Setup: 30 min (CPU) = $0.50
- Download: 30 min (CPU) = $0.50
- Training: 4 hours (GPU) = $16.00
- Total: ~$17.00
```

---

## File Structure

```
.github/workflows/
└── deploy-to-cai.yml         # ← GitHub Actions workflow

scripts/
├── setup_git_lfs.sh          # ← Run this first
└── deploy_to_cai.sh          # ← Alternative CLI tool

data/
├── cargoxray/                # ← Git LFS tracked
└── cargoxray_yolo/           # ← Git LFS tracked

.gitattributes                # ← LFS configuration
.gitignore                    # ← Exclude STCray (too large)

cai_integration/
├── jobs_config_yolo.yaml     # ← CAI job definitions
├── create_jobs.py            # ← Creates jobs in CAI
└── trigger_jobs.py           # ← Triggers jobs (if needed)

docs/
├── GIT_LFS_SETUP.md          # ← Detailed LFS guide
├── GITHUB_ACTIONS_DEPLOYMENT.md  # ← Detailed Actions guide
└── CAI_YOLO_FINETUNING.md    # ← Manual CAI setup
```

---

## Troubleshooting

### Issue 1: Git LFS Not Working

```bash
# Reinstall Git LFS
git lfs install --force

# Pull LFS files
git lfs pull

# Verify tracking
git lfs ls-files
```

### Issue 2: GitHub Actions Fails at Checkout

```
Error: Unable to download LFS files
```

**Fix:** Ensure `lfs: true` in checkout action (already configured)

### Issue 3: CAI API Authentication Failed

```
Error: 401 Unauthorized
```

**Fix:**
```bash
# Regenerate API key
CAI UI → User Settings → API Keys → Create New

# Update GitHub secret
GitHub → Settings → Secrets → CML_API_KEY
```

### Issue 4: Dataset Not Found in CAI

```
Error: data.yaml not found
```

**Fix:** Ensure Git LFS files were checked out:
```bash
# In GitHub Actions logs, check for:
# "Downloading LFS objects: 100% (N/N)"
```

### Issue 5: Exceeded LFS Bandwidth

```
Error: LFS bandwidth limit exceeded
```

**Fix:**
- Wait for monthly reset
- Upgrade to paid LFS plan ($5/month)
- Use external storage for dataset

---

## Best Practices

### 1. Manual Trigger by Default

```yaml
# Always start with trigger_jobs=false
# This lets you review jobs in CAI before running
--field trigger_jobs=false
```

### 2. Test with CargoXray First

```bash
# Quick validation (1 hour)
--field dataset=cargoxray

# Then scale to production
--field dataset=stcray
```

### 3. Use Smaller Model for Testing

```bash
# Fast testing
--field yolo_model=yolov8n.pt

# Production
--field yolo_model=yolov8m.pt
```

### 4. Version Your Datasets

```bash
# Tag dataset versions
git tag -a v1.0-cargoxray -m "CargoXray dataset v1.0"
git push --tags
```

---

## Advanced Usage

### A. Deploy Multiple Configurations

```bash
# Terminal 1: Fast baseline
gh workflow run deploy-to-cai.yml \
  --field yolo_model=yolov8n.pt \
  --field yolo_epochs=50

# Terminal 2: Production model
gh workflow run deploy-to-cai.yml \
  --field yolo_model=yolov8m.pt \
  --field yolo_epochs=200 \
  --field export_onnx=true
```

### B. CI/CD Integration

Add to workflow for auto-deployment on push:

```yaml
on:
  push:
    branches: [main]
    paths: ['training/**', 'cai_integration/**']
  workflow_dispatch:
    # ... existing inputs
```

### C. Dataset Versioning Strategy

```bash
# Tag major dataset updates
git tag -a v1.0-data -m "Initial CargoXray dataset"
git tag -a v1.1-data -m "Updated with 100 more images"
git push --tags

# Reference in workflow
--field dataset_version=v1.1-data
```

---

## Monitoring & Logs

### GitHub Actions

```bash
# Watch workflow
gh run watch

# View logs
gh run view --log

# List runs
gh run list --workflow=deploy-to-cai.yml
```

### CAI Jobs

```
CAI UI → Jobs → Job Runs → View Logs
```

### Download Trained Model

```bash
# After training completes
scp cai:/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt ./models/

# Test locally
python inference/yolo_api_server.py --model models/best.pt
```

---

## Complete Command Reference

```bash
# 1. Setup Git LFS (one-time)
./scripts/setup_git_lfs.sh

# 2. Deploy to CAI (creates jobs, no auto-run)
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray \
  --field trigger_jobs=false

# 3. Monitor deployment
gh run watch

# 4. Go to CAI and manually trigger setup_environment job
# (via CAI Web UI or API)

# 5. After training, download model
scp cai:/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt ./models/

# 6. Test locally
python inference/yolo_api_server.py --model models/best.pt --port 8000
```

---

## Summary

**Setup (One-time):**
1. Run `./scripts/setup_git_lfs.sh` (10 min)
2. Add GitHub secrets (2 min)

**Deploy (Every time):**
1. Run workflow via GitHub Actions (1 min)
2. Jobs created in CAI
3. Manually trigger in CAI when ready
4. Training completes (30 min - 4 hours)
5. Download & deploy model

**Benefits:**
- ✅ Dataset version controlled
- ✅ Automated deployment
- ✅ Manual trigger control
- ✅ Reproducible training
- ✅ No external storage needed

---

## Resources

- **Git LFS Guide**: [docs/GIT_LFS_SETUP.md](docs/GIT_LFS_SETUP.md)
- **GitHub Actions**: [docs/GITHUB_ACTIONS_DEPLOYMENT.md](docs/GITHUB_ACTIONS_DEPLOYMENT.md)
- **CAI Manual Setup**: [docs/CAI_YOLO_FINETUNING.md](docs/CAI_YOLO_FINETUNING.md)
- **Quick Start**: [QUICKSTART_YOLO.md](QUICKSTART_YOLO.md)

---

**Ready to deploy?** Run:

```bash
./scripts/setup_git_lfs.sh
```

Then trigger the workflow via GitHub Actions! 🚀
