# Manual CAI Training from Command Line

Complete guide for running YOLO training on X-Ray Baggage dataset from CAI session terminal (CPU-only).

---

## Method 1: Use CAI Jobs API (Recommended)

### Step 1: Use the committed CPU Baggage config

The repo includes **`cai_integration/jobs_config_yolo_cpu_baggage.yaml`**: three jobs with **`gpu: 0`**, **`YOLO_DEVICE=cpu`**, and **distinct CAI job names** so they do not collide with the default GPU pipeline.

### Step 2: Set CAI credentials

```bash
export CML_HOST="your-cai-workspace.cloudera.site"
export CML_API_KEY="your-api-key"
export PROJECT_ID="your-project-id"
```

**How to get these values:**
- `CML_HOST`: Your CAI workspace URL (without https://)
- `CML_API_KEY`: Settings → API Keys → Create new key
- `PROJECT_ID`: Check URL when in your project (e.g., `/projects/abc123`)

### Step 3: Create jobs

```bash
cd /home/cdsw

python cai_integration/create_jobs.py \
  --project-id "$PROJECT_ID" \
  --config cai_integration/jobs_config_yolo_cpu_baggage.yaml
```

### Step 4: Trigger jobs

Child jobs run when parents succeed. Manually start the pipeline from the CAI **Jobs** UI:

1. Open **Jobs**
2. Run **“Setup Python Environment (YOLO CPU — Baggage)”** (root job)

CAI job chain:

1. **Setup Python Environment (YOLO CPU — Baggage)** →  
2. **Process X-Ray Baggage (CPU pipeline)** →  
3. **Train YOLO on CPU — X-Ray Baggage**

---

## Method 2: Direct Script Execution (Faster for Testing)

Run the scripts directly in your CAI session without creating jobs:

### Step 1: Setup Environment

```bash
cd /home/cdsw

# Run setup script
bash cai_integration/setup_environment.sh
```

### Step 2: Activate Virtual Environment

```bash
source /home/cdsw/.venv/bin/activate

# Verify installation
python -c "import torch; from ultralytics import YOLO; print(f'PyTorch: {torch.__version__}, YOLO: OK')"
```

### Step 3: Download X-Ray Baggage Dataset

```bash
# Pull Git LFS file
git lfs pull

# Process the dataset
python cai_integration/download_xray_baggage.py
```

Expected output:
```
✓ xray_baggage_yolo ready: 1,412 train / 157 val
✓ data.yaml created
```

### Step 4: Train YOLO on CPU

```bash
# Set environment variables for CPU training
export DATASET="xray_baggage"
export MODEL_NAME="yolov8n.pt"
export EPOCHS="100"
export BATCH_SIZE="16"
export IMG_SIZE="640"
export YOLO_DEVICE="cpu"
export LEARNING_RATE="0.01"
export OPTIMIZER="SGD"
export PATIENCE="10"

# Run training
python cai_integration/yolo_training.py
```

**Training will take ~2-3 hours on CPU.**

### Monitor Training

```bash
# In another terminal, watch progress
tail -f /home/cdsw/runs/detect/train*/results.csv

# Or check GPU/CPU usage
top
```

---

## Method 3: Use Python Directly (Most Flexible)

Create a training script in your CAI session:

```bash
cd /home/cdsw
source .venv/bin/activate

# Create training script
cat > train_xray_baggage_cpu.py << 'PYEOF'
#!/usr/bin/env python3
"""Quick training script for X-Ray Baggage on CPU."""

from ultralytics import YOLO
from pathlib import Path

# Configuration
DATA_YAML = "data/xray_baggage_yolo/data.yaml"
MODEL = "yolov8n.pt"
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "cpu"

# Verify data exists
if not Path(DATA_YAML).exists():
    print(f"❌ Error: {DATA_YAML} not found")
    print("Run: python cai_integration/download_xray_baggage.py")
    exit(1)

print("=" * 60)
print("Training YOLO on X-Ray Baggage (CPU)")
print("=" * 60)
print(f"Data: {DATA_YAML}")
print(f"Model: {MODEL}")
print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print("=" * 60)

# Load model
model = YOLO(MODEL)

# Train
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=8,
    project="runs/detect",
    name="xray_baggage_cpu",
    patience=10,
    save=True,
    plots=True,
    verbose=True,
)

print("\n" + "=" * 60)
print("✅ Training complete!")
print(f"Best model: runs/detect/xray_baggage_cpu/weights/best.pt")
print("=" * 60)
PYEOF

chmod +x train_xray_baggage_cpu.py
```

### Run the Training Script

```bash
python train_xray_baggage_cpu.py
```

---

## Quick Command Summary

For the fastest manual training:

```bash
# 1. Setup (one-time)
cd /home/cdsw
bash cai_integration/setup_environment.sh
source .venv/bin/activate

# 2. Prepare dataset
git lfs pull
python cai_integration/download_xray_baggage.py

# 3. Train (single command)
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(
    data='data/xray_baggage_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu',
    workers=8,
    project='runs/detect',
    name='xray_baggage_cpu',
    patience=10
)
print('✅ Training complete!')
"
```

---

## CPU vs GPU Training Time

| Dataset | CPU Time | GPU Time | Recommendation |
|---------|----------|----------|----------------|
| xray_baggage (1.5K) | 2-3 hours | 30-45 min | ✅ CPU OK |
| luggage_xray (7K) | 10-12 hours | 2 hours | ⚠️ GPU preferred |
| combined (55K) | 80-100 hours | 4 hours | ❌ GPU required |

For xray_baggage, **CPU training is totally viable** for quick testing!

---

## Troubleshooting

### Issue: Git LFS files not available

```bash
# Pull LFS files
git lfs pull

# Verify zip exists
ls -lh data/X-Ray\ Baggage.coco.zip
```

### Issue: Virtual environment not found

```bash
# Create and activate
python3.10 -m venv /home/cdsw/.venv
source /home/cdsw/.venv/bin/activate

# Install dependencies
pip install -r setup/requirements.txt
```

### Issue: CUDA detected on CPU-only job

Force CPU usage:
```bash
export CUDA_VISIBLE_DEVICES=""
export DEVICE="cpu"
```

### Check Results

```bash
# View training results
ls -lh runs/detect/*/weights/best.pt

# Check metrics
cat runs/detect/*/results.csv | column -t -s,
```

---

## Next Steps After Training

1. **Validate model**:
   ```bash
   python -c "
   from ultralytics import YOLO
   model = YOLO('runs/detect/xray_baggage_cpu/weights/best.pt')
   metrics = model.val()
   print(f'mAP50: {metrics.box.map50:.3f}')
   print(f'mAP50-95: {metrics.box.map:.3f}')
   "
   ```

2. **Test inference**:
   ```bash
   python -c "
   from ultralytics import YOLO
   model = YOLO('runs/detect/xray_baggage_cpu/weights/best.pt')
   results = model.predict('data/xray_baggage_yolo/images/valid/valid_000001.jpg')
   results[0].show()
   "
   ```

3. **Export to ONNX** (for production):
   ```bash
   python -c "
   from ultralytics import YOLO
   model = YOLO('runs/detect/xray_baggage_cpu/weights/best.pt')
   model.export(format='onnx')
   "
   ```

---

**Happy training!** 🚀
