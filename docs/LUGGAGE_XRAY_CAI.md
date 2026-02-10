# Luggage X-ray Dataset for CAI

Guide for downloading and using the Luggage X-ray dataset in Cloudera AI Workspace (CAI).

---

## Overview

The Luggage X-ray dataset (yolov5xray v1) is a **recommended dataset** for YOLO training with:
- **7,120 images** (6,164 train, 956 validation)
- **12 categories** (5 threats + 7 normal items)
- **Medium complexity** - perfect for learning and production
- **~1 hour training time** on single GPU

---

## Quick Start in CAI

### Option 1: Manual Run (Recommended for First Time)

```bash
# SSH or open terminal in CAI
cd /home/cdsw

# Run the download script
python cai_integration/download_luggage_xray.py
```

**Expected time**: 15-30 minutes (depends on network speed)

**Output**:
```
data/luggage_xray_yolo/
├── data.yaml
├── images/
│   ├── train/    # 6,164 images
│   └── valid/    # 956 images
└── labels/
    ├── train/    # 6,164 .txt files
    └── valid/    # 956 .txt files
```

### Option 2: As CAI Job (Automated)

Add to your `jobs_config_yolo.yaml`:

```yaml
jobs:
  # New job for Luggage X-ray download
  download_luggage_xray:
    name: "Download Luggage X-ray Dataset"
    description: "Download and prepare Luggage X-ray dataset (7,120 images, 12 categories)"
    script: "cai_integration/download_luggage_xray.py"
    kernel: "python3"
    cpu: 4
    memory: 8
    timeout: 3600  # 1 hour
    parent_job_key: "setup_environment"
    runtime_identifier: "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.10-standard:2025.09.1-b5"
    environment:
      MAX_WORKERS: "8"  # Parallel download threads

  # Updated YOLO training to use Luggage dataset
  yolo_training:
    name: "Train YOLO on Luggage X-ray"
    description: "Train YOLOv8 on Luggage X-ray dataset (fast, 1 hour)"
    script: "cai_integration/yolo_training.py"
    kernel: "python3"
    cpu: 8
    memory: 32
    gpu: 1
    timeout: 3600  # 1 hour for Luggage dataset
    parent_job_key: "download_luggage_xray"  # Wait for download
    runtime_identifier: "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.10-cuda:2025.09.1-b5"
    environment:
      MODEL_NAME: "yolov8n.pt"
      EPOCHS: "100"
      BATCH_SIZE: "16"
      IMG_SIZE: "640"
      DATASET: "luggage_xray"
```

---

## Dataset Details

### Categories (12 Classes)

**Threat Items (5)**:
- `blade` - Razor blades, utility blades
- `dagger` - Fixed-blade daggers
- `knife` - Combat knives, kitchen knives
- `scissors` - All types of scissors
- `SwissArmyKnife` - Multi-tool knives

**Normal Items (7)**:
- `Cans` - Metal cans, containers
- `CartonDrinks` - Juice boxes, milk cartons
- `GlassBottle` - Glass bottles
- `PlasticBottle` - Plastic bottles, water bottles
- `SprayCans` - Aerosol cans
- `Tin` - Tin containers
- `VacuumCup` - Thermos, insulated cups

### Statistics

| Metric | Value |
|--------|-------|
| Total images | 7,120 |
| Train images | 6,164 |
| Validation images | 956 |
| Categories | 12 |
| Threat categories | 5 |
| Format | YOLO (txt labels) |
| Avg objects/image | 1-3 |
| Image size | Varies (640-1024px) |

---

## Training in CAI

### Using the Downloaded Dataset

Once downloaded, train with:

```bash
# From CAI terminal
cd /home/cdsw

# Activate venv (if using)
source .venv/bin/activate

# Train YOLOv8n (recommended)
python training/train_yolo.py \
  --data data/luggage_xray_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

### Expected Results (YOLOv8n, 100 epochs)

| Metric | Value |
|--------|-------|
| **Training time** | ~1 hour (1x GPU) |
| **mAP@0.5** | 0.80-0.85 |
| **mAP@0.5:0.95** | 0.55-0.60 |
| **Inference speed** | 20ms/image |
| **Model size** | 6MB |

### Training Variants

```bash
# YOLOv8s (better accuracy, 2 hours)
python training/train_yolo.py \
  --data data/luggage_xray_yolo/data.yaml \
  --model yolov8s.pt \
  --epochs 100

# YOLOv8m (best accuracy, 3 hours)
python training/train_yolo.py \
  --data data/luggage_xray_yolo/data.yaml \
  --model yolov8m.pt \
  --epochs 100
```

---

## Script Details

### What the Script Does

The `download_luggage_xray.py` script:

1. **Downloads** dataset from Roboflow (~350MB)
2. **Extracts** JSONL annotations
3. **Downloads images** from URLs in parallel (8 workers)
4. **Converts** from OpenAI JSONL format to YOLO format
5. **Creates** `data.yaml` for training
6. **Verifies** all files downloaded correctly

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WORKERS` | 8 | Parallel download threads |

### Requirements

- **Network access** to Roboflow
- **Disk space**: ~1GB (images + labels)
- **Memory**: 8GB recommended
- **CPU**: 4 cores (for parallel downloads)
- **Time**: 15-30 minutes

---

## Comparison with Other Datasets

| Feature | Luggage X-ray | STCray | CargoXray |
|---------|---------------|--------|-----------|
| **Images** | 7,120 | 46,642 | 659 |
| **Categories** | 12 | 21 | 16 |
| **Threats** | ✅ 5 | ✅ 21 | ❌ None |
| **Training time** | 1 hour | 4-8 hours | 20 mins |
| **Download time** | 20 mins | 2 hours | 2 mins |
| **Complexity** | Medium | Very High | Low |
| **Best for** | **YOLO learning** | Production | Testing |

---

## Troubleshooting

### Download Fails

```bash
# Check network access
curl -I https://app.roboflow.com

# Increase workers if timeout
export MAX_WORKERS=16
python cai_integration/download_luggage_xray.py
```

### Out of Disk Space

```bash
# Check available space
df -h /home/cdsw

# Clean up old datasets
rm -rf data/cargoxray data/cargoxray_yolo

# Retry download
python cai_integration/download_luggage_xray.py
```

### Images Not Downloading

```bash
# Reduce parallel workers
export MAX_WORKERS=4
python cai_integration/download_luggage_xray.py
```

### Dataset Already Exists

The script will prompt:
```
⚠️  Dataset already exists at: /home/cdsw/data/luggage_xray_yolo
Delete and re-download? (yes/no):
```

Type `yes` to re-download or `no` to keep existing.

---

## Next Steps

### After Download

1. **Verify dataset**:
   ```bash
   ls -lh data/luggage_xray_yolo/
   cat data/luggage_xray_yolo/data.yaml
   ```

2. **Train model** (see Training section above)

3. **Deploy API**:
   ```bash
   python inference/yolo_api_server.py \
     --model runs/detect/xray_detection_luggage_xray/weights/best.pt \
     --data data/luggage_xray_yolo/data.yaml
   ```

4. **Test inference**:
   ```bash
   python inference/test_yolo_api.py \
     --image data/luggage_xray_yolo/images/valid/valid_000001.jpg
   ```

---

## References

- **Dataset source**: [Roboflow Universe - yolov5xray v1](https://app.roboflow.com/)
- **Script**: `cai_integration/download_luggage_xray.py`
- **Documentation**: [DATASETS_COMPARISON.md](DATASETS_COMPARISON.md)
- **Training guide**: [YOLO_TRAINING.md](YOLO_TRAINING.md)

---

**Questions?** See [README.md](../README.md) or check other dataset guides!
