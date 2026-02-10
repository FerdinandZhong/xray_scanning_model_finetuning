# Quick Start Guide

Get started with X-ray threat detection in minutes using the Luggage X-ray dataset.

## Prerequisites

- Python 3.10+
- GPU with at least 8GB VRAM (for training)
- ~10GB storage for dataset

## Choose Your Approach

### YOLO Detection (Recommended)
- **Fast**: 20-100ms inference, 1 hour training
- **Lightweight**: 11-47MB models, 2-8GB VRAM
- **Dataset**: Luggage X-ray (7k images, 12 classes including 5 threats)
- **Best for**: Production deployment

### VLM Approach (Advanced)
- **Flexible**: Natural language reasoning
- **Large**: Multi-GB models, 16GB+ VRAM
- **Dataset**: STCray with VQA format
- **Best for**: Research, complex queries

---

## Quick Start - YOLO on Luggage X-ray

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd xray_scanning_model_finetuning

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install ultralytics torch torchvision pillow pyyaml tqdm requests openai
```

### 2. Download Dataset

```bash
# Download Luggage X-ray dataset (530MB)
curl -L "https://app.roboflow.com/ds/nMb0ckPbFf?key=EZzAfTucdZ" > roboflow.zip
unzip roboflow.zip
rm roboflow.zip
```

### 3. Convert to YOLO Format

```bash
# Convert OpenAI JSONL to YOLO format (parallel downloads)
python scripts/convert_luggage_xray_to_yolo.py \
    --input-dir data/luggage_xray \
    --output-dir data/luggage_xray_yolo \
    --max-workers 8

# Expected output:
# ✓ data/luggage_xray_yolo/
#   ├── images/train/ (6,164 images)
#   ├── images/valid/ (956 images)
#   ├── labels/train/ (6,164 .txt files)
#   ├── labels/valid/ (956 .txt files)
#   └── data.yaml
```

### 4. Train YOLO Model

```bash
# Train YOLOv8n (fastest, ~1 hour on T4 GPU)
python training/train_yolo.py \
    --data data/luggage_xray_yolo/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640

# Or use YOLOv8s for better accuracy (~2 hours)
python training/train_yolo.py \
    --data data/luggage_xray_yolo/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 8 \
    --imgsz 640
```

### 5. Test Model

```bash
# Run inference on validation set
python scripts/test_yolo_inference.py \
    --model runs/detect/train/weights/best.pt \
    --images data/luggage_xray_yolo/images/valid/*.jpg \
    --conf 0.25

# Start API server (OpenAI-compatible)
python inference/yolo_api_server.py \
    --model runs/detect/train/weights/best.pt \
    --port 8000
```

### 6. Test API

```bash
# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What objects do you see?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  }'
```

---

## CAI Deployment (Recommended for Production)

### Option 1: GitHub Actions (Automated)

1. Go to **Actions** tab → **Deploy X-ray Detection to CAI**
2. Click **Run workflow**
3. Select:
   - **model_type**: `yolo`
   - **dataset**: `luggage_xray` (recommended)
   - **trigger_jobs**: `true`
4. Monitor progress in CAI Workspace

### Option 2: Manual CAI Setup

```bash
# In CAI Workspace terminal
git clone <your-repo-url>
cd xray_scanning_model_finetuning

# Download and prepare dataset
python cai_integration/download_luggage_xray.py

# Train model
python cai_integration/yolo_training.py \
    --dataset luggage_xray \
    --epochs 100 \
    --batch 16
```

See [docs/LUGGAGE_XRAY_CAI.md](docs/LUGGAGE_XRAY_CAI.md) for detailed CAI guide.

---

## Testing Models Before Training

Test pre-trained models on the dataset before fine-tuning:

```bash
# Test GPT-4.1 (requires OpenAI API key)
export OPENAI_API_KEY="your-api-key"
python scripts/test_gpt4_luggage.py --num-samples 10

# Test RolmOCR (requires endpoint URL)
python scripts/test_rolmocr_luggage.py \
    --base-url "https://your-endpoint.cloudera.site/openai/v1" \
    --model "reducto/RolmOCR" \
    --num-samples 10

# Test PaddleOCR (for text detection comparison)
python scripts/test_paddleocr_luggage.py \
    --base-url "http://your-paddleocr-endpoint:8080" \
    --num-samples 10
```

Results are saved to `test_results/` with confusion matrices and detailed reports.

---

## Alternative Datasets

### CargoXray (Quick Baseline)
- 659 images, 16 classes
- 30 minutes training
- Best for pipeline validation

```bash
# Download
curl -L "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6" > roboflow.zip
unzip roboflow.zip && rm roboflow.zip

# Convert
python scripts/convert_cargoxray_to_yolo.py \
    --input-dir data/cargoxray \
    --output-dir data/cargoxray_yolo

# Train
python training/train_yolo.py \
    --data data/cargoxray_yolo/data.yaml \
    --model yolov8n.pt \
    --epochs 100
```

See [docs/CARGOXRAY_QUICKSTART.md](docs/CARGOXRAY_QUICKSTART.md)

### STCray (Production Scale)
- 46,642 images, 21 classes
- 4 hours training
- Best for large-scale deployment

See [docs/DATASETS_COMPARISON.md](docs/DATASETS_COMPARISON.md)

---

## VLM Approach (Advanced)

For VLM fine-tuning with VQA format:

1. Generate VQA pairs from dataset
2. Fine-tune Qwen2.5-VL or similar model
3. Deploy with vLLM

See [docs/VQA_GENERATOR_VERIFICATION.md](docs/VQA_GENERATOR_VERIFICATION.md) for details.

---

## Troubleshooting

### GPU Out of Memory
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `yolov8n.pt` instead of `yolov8s.pt`
- Reduce image size: `--imgsz 512`

### Dataset Download Issues
- Check internet connection
- Verify Roboflow API key is valid
- Try downloading manually and extracting to `data/luggage_xray/`

### Training Not Converging
- Increase epochs: `--epochs 150` or `--epochs 200`
- Adjust learning rate: `--lr0 0.001`
- Check dataset labels with: `yolo val data=data/luggage_xray_yolo/data.yaml`

---

## Next Steps

- 📖 Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment
- 📊 Compare datasets in [docs/DATASETS_COMPARISON.md](docs/DATASETS_COMPARISON.md)
- 🚀 Use GitHub Actions for automated CAI deployment
- 📈 Monitor training with TensorBoard: `tensorboard --logdir runs/detect/`

## Support

For issues and questions:
- Documentation: [docs/](docs/)
- GitHub Issues: Create an issue in the repository
