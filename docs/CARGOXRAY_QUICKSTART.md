# CargoXray Quick Start Guide

This guide shows you how to download and train on the CargoXray dataset - a simpler alternative to STCray for initial testing and baseline models.

## Overview

**CargoXray Dataset:**
- **659 X-ray images** of cargo containers (trucks, railcars)
- **16 object categories** (textiles, auto parts, tools, shoes, etc.)
- **Clearer images** with larger objects compared to baggage X-rays
- **Ready-to-use** splits (70% train, 20% valid, 10% test)
- **Source**: Roboflow Universe

## Step 1: Download Dataset

```bash
cd data
mkdir -p cargoxray
cd cargoxray

# Download from Roboflow (no DVC required!)
curl -L "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6" > roboflow.zip
unzip roboflow.zip
rm roboflow.zip

cd ../..
```

**Downloaded structure:**
```
data/cargoxray/
├── README.roboflow.txt
├── train/ (462 images)
│   ├── *.jpg
│   └── _annotations.coco.json
├── valid/ (132 images)
│   ├── *.jpg
│   └── _annotations.coco.json
└── test/ (65 images)
    ├── *.jpg
    └── _annotations.coco.json
```

## Step 2: Convert to YOLO Format

```bash
# Convert COCO annotations to YOLO format
python scripts/convert_cargoxray_to_yolo.py \
  --input-dir data/cargoxray \
  --output-dir data/cargoxray_yolo
```

**Output:**
- ✅ 659 images converted
- ✅ 16 unique categories (duplicates/typos normalized)
- ✅ YOLO format: `class_id x_center y_center width height`
- ✅ `data.yaml` created for training

**Categories:**
1. auto_parts
2. bags
3. bicycle
4. car_wheels
5. clothes
6. fabrics
7. lamps
8. office_supplies
9. shoes
10. spare_parts
11. tableware
12. textiles
13. tools
14. toys
15. unknown
16. xray_objects

## Step 3: Train YOLO Model

### Quick Training (Local)

```bash
# Train YOLOv8n on CargoXray
python training/train_yolo.py \
  --data data/cargoxray_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --name cargoxray_v1
```

**Expected training time:**
- 1x GPU: ~30 minutes
- 2x GPU: ~15 minutes

### Production Training (Larger Model)

```bash
# Train YOLOv8m for better accuracy
python training/train_yolo.py \
  --data data/cargoxray_yolo/data.yaml \
  --model yolov8m.pt \
  --epochs 200 \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --name cargoxray_production
```

## Step 4: Test Inference

```bash
# Test on sample images
python scripts/test_yolo_inference.py \
  --model runs/detect/cargoxray_v1/weights/best.pt \
  --images data/cargoxray_yolo/test/images \
  --output test_results/cargoxray
```

## Step 5: Start API Server

```bash
# Start OpenAI-compatible API server
python inference/yolo_api_server.py \
  --model runs/detect/cargoxray_v1/weights/best.pt \
  --host 0.0.0.0 \
  --port 8000
```

**Test the API:**
```bash
# Using curl
curl -X POST http://localhost:8000/v1/detect \
  -F "file=@data/cargoxray_yolo/test/images/sample.jpg"

# Using Python client
python test_yolo_api.py \
  --base-url http://localhost:8000 \
  --image data/cargoxray_yolo/test/images/sample.jpg
```

## Use Cases

### 1. Baseline for STCray Training

Train on CargoXray first (simpler), then fine-tune on STCray (harder):

```bash
# Stage 1: Pre-train on CargoXray (30 min)
python training/train_yolo.py \
  --data data/cargoxray_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --name stage1_cargo

# Stage 2: Fine-tune on STCray (2-3 hours)
python training/train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model runs/detect/stage1_cargo/weights/best.pt \
  --epochs 50 \
  --name stage2_stcray
```

### 2. Testing RolmOCR

CargoXray has clearer images - better for testing RolmOCR baseline:

```bash
# Convert a few samples for RolmOCR testing
python -c "
import json
import shutil
from pathlib import Path

# Select 10 random cargo images
src_dir = Path('data/cargoxray/test')
dest_dir = Path('data/test_xrays/cargo_samples')
dest_dir.mkdir(parents=True, exist_ok=True)

# Copy images
for img in list(src_dir.glob('*.jpg'))[:10]:
    shutil.copy(img, dest_dir / img.name)
print(f'Copied 10 cargo X-ray images to {dest_dir}')
"

# Test RolmOCR
python test_rolmocr.py \
  --image-dir data/test_xrays/cargo_samples
```

### 3. Multi-Domain Model

Train a universal X-ray detector (both cargo and baggage):

```bash
# Would need to merge datasets first
# Then train on combined dataset
```

## Expected Performance

### CargoXray Metrics (Baseline)

| Model | mAP@0.5 | mAP@0.5:0.95 | Inference Time |
|-------|---------|--------------|----------------|
| YOLOv8n | ~0.75 | ~0.45 | 20-30ms |
| YOLOv8s | ~0.80 | ~0.50 | 30-40ms |
| YOLOv8m | ~0.85 | ~0.55 | 50-70ms |

*Note: Cargo X-rays typically have better detection performance than baggage due to larger, clearer objects*

### STCray Comparison

For reference, STCray (baggage) is harder:

| Model | Dataset | mAP@0.5 | Objects |
|-------|---------|---------|---------|
| YOLOv8n | CargoXray | ~0.75 | Large containers |
| YOLOv8n | STCray | ~0.65 | Small weapons/items |

## Troubleshooting

### Download Issues

If the Roboflow download fails:
```bash
# Try with wget
wget -O roboflow.zip "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6"
```

### Conversion Issues

If conversion fails:
```bash
# Check COCO annotations exist
ls -la data/cargoxray/train/_annotations.coco.json

# Re-run with verbose output
python scripts/convert_cargoxray_to_yolo.py --input-dir data/cargoxray --output-dir data/cargoxray_yolo
```

### Training Issues

If training is slow:
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 416`
- Use smaller model: `yolov8n.pt` instead of `yolov8m.pt`

## Next Steps

1. ✅ Downloaded CargoXray (659 images)
2. ✅ Converted to YOLO format
3. ⬜ Train baseline model (~30 min)
4. ⬜ Evaluate on test set
5. ⬜ Compare with STCray results
6. ⬜ Deploy API server
7. ⬜ Test in agentic workflow

## Resources

- **Roboflow Dataset**: https://app.roboflow.com/ds/BbQux1Jbmr
- **Conversion Script**: `scripts/convert_cargoxray_to_yolo.py`
- **Training Guide**: [docs/YOLO_TRAINING.md](YOLO_TRAINING.md)
- **API Guide**: [docs/YOLO_API.md](YOLO_API.md)

---

**Questions?** Open an issue or check the main README.md for more details!
