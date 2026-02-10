# YOLO Fine-Tuning Quick Start

**Goal**: Train YOLO for X-ray detection on CAI (given RolmOCR's 0% accuracy)

## 5-Minute Setup

### Prerequisites

```bash
# 1. GitHub secrets configured
#    Settings → Secrets → Actions → Add:
#    - CML_HOST: https://your-cai-workspace.cloudera.site
#    - CML_API_KEY: your-cai-api-key

# 2. GitHub CLI installed (optional)
brew install gh  # or: sudo apt install gh
gh auth login
```

### Deploy (2 ways)

#### Option A: Web UI (No CLI needed)

```
1. Go to GitHub repo → Actions tab
2. Click "Deploy X-ray Detection to CAI"
3. Click "Run workflow"
4. Select:
   - model_type: yolo
   - dataset: cargoxray  ← Start here (30 min)
   - trigger_jobs: true
5. Click "Run workflow"
```

#### Option B: Command Line

```bash
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray \
  --field trigger_jobs=true
```

### Monitor

```bash
# Watch progress
gh run watch

# Or check in GitHub: Actions → Latest run
```

## What Happens

```
GitHub Actions (5 min)
  ↓ Validates code
  ↓ Creates CAI project
  ↓ Creates 3 jobs
  ↓ Triggers pipeline

CAI Workspace (1 hour for CargoXray)
  ↓ setup_environment (30 min)
  ↓ upload_cargoxray (5 min)
  ↓ yolo_training (30 min)

Result: Trained model at
/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt
```

## After Training

### Download Model

```bash
# SSH or use CAI file browser
scp cai:/home/cdsw/runs/detect/cargoxray_v1/weights/best.pt ./models/
```

### Test Locally

```bash
# Start API server
python inference/yolo_api_server.py --model models/best.pt --port 8000

# Test
curl -X POST http://localhost:8000/v1/detect \
  -F "file=@data/cargoxray/test/sample.jpg"
```

## Next Steps

### If Results Good → Scale to STCray

```bash
# Production model (4 hours)
gh workflow run deploy-to-cai.yml \
  --field dataset=stcray \
  --field trigger_jobs=true
```

### If You Need Help

- **GitHub Actions**: [docs/GITHUB_ACTIONS_DEPLOYMENT.md](docs/GITHUB_ACTIONS_DEPLOYMENT.md)
- **CAI Manual Setup**: [docs/CAI_YOLO_FINETUNING.md](docs/CAI_YOLO_FINETUNING.md)
- **YOLO Training**: [docs/YOLO_TRAINING.md](docs/YOLO_TRAINING.md)

## Troubleshooting

### Workflow fails

```bash
# Check logs
gh run view --log-failed
```

### CAI API error

```bash
# Regenerate API key
CAI UI → User Settings → API Keys → Create New
# Update GitHub secret
Settings → Secrets → CML_API_KEY
```

### Training timeout

```bash
# Edit timeout in jobs_config_yolo.yaml:
yolo_training:
  timeout: 18000  # 5 hours
```

## Cost

| Dataset | Time | GPU | Cost |
|---------|------|-----|------|
| CargoXray | 1h | 0.5h | ~$3 |
| STCray | 5h | 4h | ~$17 |

---

**That's it!** Your YOLO model will be trained automatically. 🚀
