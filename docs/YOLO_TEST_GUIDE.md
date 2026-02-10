# YOLO Implementation Testing Guide

This guide provides step-by-step instructions to test the complete YOLO detection pipeline.

## Prerequisites

- STCray dataset downloaded and processed
- Python environment with dependencies installed
- GPU with 8GB+ VRAM (recommended)

## Test Plan

### Phase 1: Data Conversion (5-10 minutes)

**Objective**: Verify STCray annotations convert correctly to YOLO format

```bash
# Test with small sample first
python3 data/convert_to_yolo_format.py \
    --annotations-dir data/stcray_processed \
    --output-dir data/yolo_dataset_test \
    --val-split 0.2 \
    --limit 100

# Verify output structure
ls -lh data/yolo_dataset_test/
cat data/yolo_dataset_test/data.yaml

# Check a sample label file
head -5 data/yolo_dataset_test/labels/train/*.txt

# Expected: class_id x_center y_center width height (normalized values)
```

**Success criteria**:
- ✓ `data.yaml` created with 24 classes
- ✓ Label files have correct format (5 values per line)
- ✓ All values in [0, 1] range
- ✓ Images copied to `images/train/` and `images/val/`

### Phase 2: Model Training (2-4 hours)

**Objective**: Train YOLOv8n on small dataset to verify training pipeline

```bash
# Test training on small dataset
python3 training/train_yolo.py \
    --data data/yolo_dataset_test/data.yaml \
    --model yolov8n.pt \
    --epochs 10 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --project runs/detect \
    --name xray_test

# Monitor training
tensorboard --logdir runs/detect/xray_test
```

**Success criteria**:
- ✓ Training starts without errors
- ✓ Loss decreases over epochs
- ✓ mAP increases over epochs
- ✓ Checkpoints saved to `runs/detect/xray_test/weights/`
- ✓ Training completes in ~10-30 minutes

**Expected metrics** (10 epochs, 100 samples):
- mAP50: 0.3-0.5 (low due to small dataset)
- Training loss: Should decrease from ~5.0 to ~2.0

### Phase 3: Inference Testing (5 minutes)

**Objective**: Verify model can perform inference and generate correct output format

```bash
# Test inference on sample images
python3 scripts/test_yolo_inference.py \
    --model runs/detect/xray_test/weights/best.pt \
    --images data/yolo_dataset_test/images/val/*.jpg \
    --conf-threshold 0.25 \
    --save-dir test_results \
    --no-show

# Check outputs
ls -lh test_results/
cat test_results/summary.json | python3 -m json.tool
```

**Success criteria**:
- ✓ Inference runs without errors
- ✓ JSON results have correct schema (items, total_count, has_concealed_items)
- ✓ Bounding boxes are normalized [0, 1]
- ✓ Confidence scores in [0, 1]
- ✓ Location strings are valid
- ✓ Visualization images saved

**Validate JSON schema**:
```bash
python3 << 'EOF'
import json
import jsonschema

# Load schema
with open('inference/output_schema.json') as f:
    schema = json.load(f)

# Load test result
with open('test_results/result_1.json') as f:
    result = json.load(f)
    detections = result['detections']

# Validate each detection
for det in detections:
    detection_obj = {
        "items": [det],
        "total_count": 1,
        "has_concealed_items": False
    }
    jsonschema.validate(detection_obj, schema)
    print(f"✓ {det['class_name']} validation passed")

print("\n✓ All detections match schema!")
EOF
```

### Phase 4: API Server Testing (10 minutes)

**Objective**: Verify API server starts and responds correctly

```bash
# Start API server in background
./scripts/serve_yolo_api.sh \
    --model runs/detect/xray_test/weights/best.pt \
    --port 8000 &

# Wait for startup
sleep 5

# Test health endpoint
curl http://localhost:8000/health | python3 -m json.tool

# Test direct detection endpoint
curl -X POST http://localhost:8000/v1/detect \
    -F "file=@data/yolo_dataset_test/images/val/$(ls data/yolo_dataset_test/images/val | head -1)" \
    | python3 -m json.tool

# Test OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
    -F "file=@data/yolo_dataset_test/images/val/$(ls data/yolo_dataset_test/images/val | head -1)" \
    | python3 -m json.tool

# Stop server
pkill -f yolo_api_server
```

**Success criteria**:
- ✓ Server starts without errors
- ✓ `/health` returns `{"status": "healthy"}`
- ✓ `/v1/detect` returns correct JSON schema
- ✓ `/v1/chat/completions` returns OpenAI format
- ✓ Response includes `items`, `total_count`, `has_concealed_items`
- ✓ Bounding boxes included in response

### Phase 5: Full Dataset Training (Production) (2-4 hours)

**Objective**: Train on complete STCray dataset for production use

```bash
# Convert full dataset
python3 data/convert_to_yolo_format.py \
    --annotations-dir data/stcray_processed \
    --output-dir data/yolo_dataset \
    --val-split 0.2

# Train production model
./scripts/train_yolo_local.sh \
    --model yolov8s.pt \
    --epochs 100 \
    --export-onnx

# Expected training time:
# - YOLOv8n: 2-3 hours
# - YOLOv8s: 4-5 hours
# - YOLOv8m: 8-10 hours
```

**Success criteria**:
- ✓ Training completes successfully
- ✓ mAP50 > 0.75 on validation set
- ✓ mAP50-95 > 0.45 on validation set
- ✓ Precision > 0.80
- ✓ Recall > 0.70
- ✓ ONNX model exported (if --export-onnx used)

### Phase 6: Integration Testing (Agentic Workflow)

**Objective**: Test OpenAI-compatible API with agent frameworks

```python
# test_agent_integration.py
from openai import OpenAI
import json

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Test with X-ray image
test_image = "data/yolo_dataset/images/val/test_image.jpg"

with open(test_image, "rb") as f:
    response = client.chat.completions.create(
        model="yolo-detection",
        messages=[],  # Not used but required
        files={"file": f}
    )

# Parse and validate
result_json = response.choices[0].message.content
detections = json.loads(result_json)

print("OpenAI-Compatible API Test Results:")
print(f"  Total detections: {detections['total_count']}")
print(f"  Concealed items: {detections['has_concealed_items']}")

for item in detections['items']:
    print(f"    - {item['name']}: {item['confidence']:.2f} @ {item['location']}")
    assert 0 <= item['confidence'] <= 1, "Confidence out of range"
    assert len(item['bbox']) == 4, "Invalid bbox format"

print("\n✓ Integration test passed!")
```

## Performance Benchmarks

Expected performance on NVIDIA T4 GPU (16GB):

| Metric | YOLOv8n | YOLOv8s | YOLOv8m |
|--------|---------|---------|---------|
| Training time (100 epochs) | 2-3h | 4-5h | 8-10h |
| Inference (GPU) | 20ms | 30ms | 50ms |
| Model size | 6MB | 22MB | 52MB |
| mAP50 (STCray) | 0.76 | 0.81 | 0.85 |
| VRAM usage | 4GB | 6GB | 8GB |

## Troubleshooting

### Data Conversion Fails

```bash
# Check annotations format
python3 -c "
import json
with open('data/stcray_processed/train/annotations.json') as f:
    data = json.load(f)
    print(f'Train samples: {len(data)}')
    print(f'First sample keys: {data[0].keys()}')
    print(f'Categories: {data[0][\"categories\"]}')
    print(f'Bboxes: {data[0][\"bboxes\"]}')
"
```

### Training OOM

- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 512`
- Use smaller model: `yolov8n.pt`

### API Server Fails

```bash
# Check model loads correctly
python3 -c "
from ultralytics import YOLO
model = YOLO('runs/detect/xray_test/weights/best.pt')
print(f'Model loaded: {model.names}')
"

# Check port availability
lsof -i :8000
```

### Low mAP

- Train longer: `--epochs 200`
- Use larger model: `yolov8s` or `yolov8m`
- Check data quality: verify annotations
- Review training curves: look for overfitting/underfitting

## Next Steps

After successful testing:

1. **Deploy to CAI**: Use `cai_integration/jobs_config_yolo.yaml`
2. **Set up monitoring**: Track inference latency and accuracy
3. **Integrate with agents**: Use OpenAI-compatible endpoint
4. **Production optimization**: Export to ONNX for 10-20% speedup

## Validation Checklist

- [ ] Data conversion produces valid YOLO format
- [ ] Training completes without errors
- [ ] Model achieves mAP50 > 0.75
- [ ] Inference generates correct JSON schema
- [ ] API server starts and responds correctly
- [ ] OpenAI-compatible endpoint works
- [ ] Bounding boxes are correctly normalized
- [ ] Location strings are accurate
- [ ] Occlusion detection works
- [ ] ONNX export successful (if enabled)

## Support

- [YOLO Training Guide](YOLO_TRAINING.md)
- [YOLO API Documentation](YOLO_API.md)
- [STCray Dataset Guide](STCRAY_DOWNLOAD.md)
- [GitHub Issues](https://github.com/FerdinandZhong/xray_scanning_model_finetuning/issues)
