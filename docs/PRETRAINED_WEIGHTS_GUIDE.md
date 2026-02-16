# Pre-trained Model Weights - Download Guide

How pre-trained YOLO model weights are downloaded and used in CAI Applications.

---

## Overview

When you deploy with a **pre-trained model name** (e.g., `yolov8n.pt`), the weights are automatically downloaded by Ultralytics.

---

## Download Process

### **1. Model Name Detection**

The launcher script checks if MODEL_PATH is a pre-trained model name:

```bash
PRETRAINED_MODELS="yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt yolov11n.pt yolov11s.pt yolov11m.pt"

if echo "$PRETRAINED_MODELS" | grep -q "$MODEL_PATH"; then
    echo "✓ Using pre-trained model: $MODEL_PATH"
    echo "  (Ultralytics will download automatically if needed)"
fi
```

### **2. Automatic Download**

When starting the application:

```bash
# Pre-download model during startup
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**What happens:**
1. ✅ Ultralytics checks local cache: `~/.cache/ultralytics/`
2. ✅ If not found, downloads from GitHub: `https://github.com/ultralytics/assets/releases/`
3. ✅ Saves to cache directory
4. ✅ Model ready for inference

### **3. Cache Location**

**In CAI Application:**
- Cache directory: `/home/cdsw/.cache/ultralytics/`
- Persists across application restarts (if using persistent storage)

**Model files downloaded:**
```
~/.cache/ultralytics/
├── yolov8n.pt          # 6 MB
├── yolov8s.pt          # 22 MB
├── yolov8m.pt          # 52 MB
├── yolov8l.pt          # 87 MB
└── yolov8x.pt          # 136 MB
```

---

## Available Pre-trained Models

### YOLOv8 Series

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡ Fast | **Testing, benchmarking** |
| `yolov8s.pt` | 22 MB | ⚡⚡ Medium | Balanced |
| `yolov8m.pt` | 52 MB | ⚡ Slow | High accuracy |
| `yolov8l.pt` | 87 MB | 🐢 Slower | Very high accuracy |
| `yolov8x.pt` | 136 MB | 🐌 Slowest | Maximum accuracy |

### YOLOv11 Series (Latest)

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `yolov11n.pt` | ~6 MB | ⚡⚡⚡ Fast | Latest nano |
| `yolov11s.pt` | ~22 MB | ⚡⚡ Medium | Latest small |
| `yolov11m.pt` | ~52 MB | ⚡ Slow | Latest medium |

---

## Deploy with Pre-trained Model

### Method 1: Deploy via Script

```bash
# Set credentials
export CAI_API_KEY="your-api-key"
export CAI_DOMAIN="https://ml-xxx.cloudera.site"

# Deploy with pre-trained model
python cai_integration/deploy_yolo_application.py \
  --model "yolov8n.pt" \
  --subdomain "xray-benchmark"
```

**What happens:**
1. ✅ Script creates CAI Application
2. ✅ Sets `MODEL_PATH=yolov8n.pt` in environment
3. ✅ Application starts
4. ✅ Launcher script detects it's a pre-trained model name
5. ✅ Runs: `YOLO('yolov8n.pt')` which auto-downloads
6. ✅ Model cached at `~/.cache/ultralytics/yolov8n.pt`
7. ✅ FastAPI server starts with loaded model

### Method 2: Deploy via GitHub Actions

```yaml
# In workflow inputs:
model_type: yolo
dataset: cargoxray
trigger_jobs: true
deploy_api: true

# Then after training, redeploy with pre-trained:
python cai_integration/deploy_yolo_application.py --model yolov8n.pt
```

---

## Download Timing

### **First Startup (Cold Start)**

```
Application starts
  ↓
Launcher script executes
  ↓
Detect pre-trained model name
  ↓
Download weights from GitHub (~6MB for yolov8n)
  ↓ (~30-60 seconds)
Cache at ~/.cache/ultralytics/
  ↓
Load model into memory
  ↓ (~10-20 seconds)
Start FastAPI server
  ↓
Ready! (~1-2 minutes total)
```

### **Subsequent Restarts (Warm Start)**

```
Application starts
  ↓
Launcher script executes
  ↓
Load from cache (~/.cache/ultralytics/yolov8n.pt)
  ↓ (~10-20 seconds)
Start FastAPI server
  ↓
Ready! (~30 seconds total)
```

---

## Important Notes

### ⚠️ Pre-trained Models Limitation

**Pre-trained YOLOv8 models are trained on COCO dataset (80 everyday objects)**:
- ✅ Good for: Infrastructure testing, deployment verification
- ❌ Poor for: X-ray threat detection (not trained on X-rays)
- ❌ Won't detect: blades, knives, scissors, X-ray specific items

**Example COCO classes:**
```python
{
  0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
  39: 'bottle', 76: 'scissors', 43: 'knife', ...
}
```

Some classes overlap (scissors, knife, bottle) but detection will be **poor on X-ray images**.

### ✅ For Production

**You must train on X-ray datasets:**
```bash
# Quick training (30 min)
dataset: cargoxray

# Recommended (2 hours)
dataset: luggage_xray

# Production (4-8 hours)
dataset: stcray
```

---

## Verification

### Check if Model Downloaded

After application starts, check logs in CAI UI:

```
Starting YOLO X-ray Detection API
Configuration:
  Model Path:       yolov8n.pt
  
✓ Using pre-trained model: yolov8n.pt
  (Ultralytics will download automatically if needed)

Downloading pre-trained model (if not cached)...
Downloading https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt to ~/.cache/ultralytics/yolov8n.pt...
100%|██████████| 6.23M/6.23M [00:02<00:00, 3.04MB/s]
✓ Model ready
  Cached at: ~/.cache/ultralytics/

Starting API Server
Initializing detection engine...
  Model: yolov8n.pt
  Backend: ultralytics
Loaded Ultralytics YOLO model from yolov8n.pt
Classes: 80
✓ Detection engine ready
```

### Test Download Works

Test locally first:

```bash
# Test automatic download
python3 << 'EOF'
from ultralytics import YOLO

print("Testing automatic download...")
model = YOLO('yolov8n.pt')  # Will download if not cached
print(f"✓ Model loaded")
print(f"  Classes: {len(model.names)}")
print(f"  Names: {list(model.names.values())[:5]}")
EOF
```

---

## Network Requirements

### CAI Application Needs:

**Outbound HTTPS Access** to:
- `github.com` - For downloading weights
- `raw.githubusercontent.com` - For model files

**Firewall Rules:**
- Allow HTTPS (443) outbound
- Allow access to GitHub releases

**Check network access:**
```bash
# Test from CAI terminal
curl -I https://github.com/ultralytics/assets/releases
```

---

## File Sizes & Download Times

| Model | Size | Download Time (10 Mbps) | Download Time (100 Mbps) |
|-------|------|-------------------------|--------------------------|
| yolov8n.pt | 6 MB | ~5 seconds | ~1 second |
| yolov8s.pt | 22 MB | ~18 seconds | ~2 seconds |
| yolov8m.pt | 52 MB | ~42 seconds | ~4 seconds |
| yolov8l.pt | 87 MB | ~70 seconds | ~7 seconds |
| yolov8x.pt | 136 MB | ~110 seconds | ~11 seconds |

---

## Troubleshooting

### Download Fails

**Error**: `Failed to download yolov8n.pt`

**Causes:**
1. No internet access from CAI
2. GitHub blocked by firewall
3. Network timeout

**Solutions:**
```bash
# Option 1: Pre-download in setup job
# Add to jobs_config_yolo.yaml -> setup_environment:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Option 2: Upload weights to project
# Download locally, then upload to CAI:
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
# Upload to CAI: models/pretrained/yolov8n.pt
# Deploy with: --model models/pretrained/yolov8n.pt

# Option 3: Use trained model instead
# Train on X-ray dataset, then deploy trained weights
```

### Model Not Found After Download

**Check cache:**
```bash
# In CAI terminal
ls -lh ~/.cache/ultralytics/

# Should show:
# -rw-r--r-- 1 user user 6.2M ... yolov8n.pt
```

### Slow Download

**Optimization:**
1. Use smaller model: `yolov8n.pt` instead of `yolov8x.pt`
2. Pre-download in setup job
3. Upload weights manually to project

---

## Recommended Approach

### For Benchmarking

**Best practice**: Quick train on actual X-ray data

```bash
# GitHub Actions:
model_type: yolo
dataset: cargoxray        # 30 min training
yolo_epochs: 50
deploy_api: true
```

**Why**: Gets you a real X-ray detection model faster than explaining pre-trained limitations!

### For Infrastructure Testing Only

**If you just need to test deployment:**

```bash
# Deploy with pre-trained
python cai_integration/deploy_yolo_application.py \
  --model "yolov8n.pt" \
  --subdomain "xray-infra-test"

# Test endpoints work (detection results will be poor on X-rays)
curl https://xray-infra-test.[domain]/health
```

---

## Summary

### How Pre-trained Weights Are Downloaded:

1. **Model name** (e.g., `yolov8n.pt`) is specified
2. **Launcher script** detects it's a pre-trained model
3. **Ultralytics library** auto-downloads from GitHub
4. **Cached** at `~/.cache/ultralytics/`
5. **Loaded** into memory
6. **Server starts** with model ready

### Updated Launcher Features:

✅ **Supports pre-trained model names** (yolov8n.pt, etc.)  
✅ **Auto-downloads if not cached**  
✅ **Shows clear status messages**  
✅ **Lists available models if error**  
✅ **Pre-downloads during startup** (faster first request)

### File Locations:

| Type | Location |
|------|----------|
| Pre-trained weights | `~/.cache/ultralytics/yolov8n.pt` |
| Trained weights | `runs/detect/*/weights/best.pt` |
| Launcher script | `cai_integration/launch_yolo_application.sh` |

---

**Now you can deploy with either pre-trained OR trained models!** 🎉
