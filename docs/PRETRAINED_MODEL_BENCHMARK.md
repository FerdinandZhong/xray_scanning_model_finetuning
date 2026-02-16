# Pre-trained Model Deployment for Benchmarking

Guide to deploy a model for benchmarking the deployment infrastructure and API performance.

---

## Recommended: Option 2 - Quick Train on CargoXray

**Best for**: Realistic X-ray detection benchmarking

This trains a model on actual X-ray images in ~30 minutes, giving you:
- ✅ Real X-ray detection performance
- ✅ Actual threat detection results
- ✅ Representative API latency
- ✅ Good for benchmarking deployment system

### Steps:

**1. Trigger Quick Training via GitHub Actions**

Go to Actions → "Deploy X-ray Detection to CAI" → Run workflow:

```yaml
model_type: yolo
dataset: cargoxray        # Small dataset (659 images)
yolo_model: yolov8n.pt    # Fast model
yolo_epochs: 50           # Quick training
trigger_jobs: true
deploy_api: true          # Auto-deploy after training
api_subdomain: xray-benchmark
```

**Expected time**: ~30-40 minutes total
- Training: 20-30 min
- Deployment: 2-5 min

**2. Test the Deployed API**

After completion:

```bash
# Health check
curl https://xray-benchmark.[your-domain]/health

# Test with X-ray image
curl -X POST https://xray-benchmark.[your-domain]/v1/detect \
  -F "file=@data/cargoxray_yolo/images/test/test_000001.jpg" \
  | jq .

# View API docs
open https://xray-benchmark.[your-domain]/docs
```

**3. Run Benchmark Tests**

```bash
# Test multiple images
for i in {1..20}; do
  time curl -X POST https://xray-benchmark.[your-domain]/v1/detect \
    -F "file=@data/cargoxray_yolo/images/test/test_$(printf '%06d' $i).jpg" \
    -o /dev/null -s 2>&1 | grep real
done
```

---

## Alternative: Option 1 - Pre-trained YOLOv8 (COCO)

**Best for**: Testing deployment infrastructure only

⚠️ **Warning**: This model is trained on COCO dataset (everyday objects), NOT X-ray images. It won't detect X-ray threats accurately.

**Use this only to**:
- Test deployment system
- Verify API endpoints work
- Check infrastructure setup

### Steps:

**1. Download Pre-trained Model**

```bash
# Activate environment
source .venv/bin/activate

# Download YOLOv8 nano (fastest)
python3 << 'EOF'
from ultralytics import YOLO
import shutil
from pathlib import Path

# Download model
print("Downloading YOLOv8n (COCO pre-trained)...")
model = YOLO('yolov8n.pt')

# Create models directory
Path('models/pretrained').mkdir(parents=True, exist_ok=True)

# Model is cached by ultralytics, copy to our directory
print("✅ Model ready: yolov8n.pt")
print("⚠️  This model is trained on COCO (everyday objects), not X-rays")
EOF
```

**2. Deploy Manually**

Since the model isn't trained on X-rays, deploy it for infrastructure testing only:

```bash
# Set environment variables
export CAI_API_KEY="your-api-key"
export CAI_DOMAIN="https://ml-xxx.cloudera.site"
export CAI_PROJECT_NAME="your-project"
export MODEL_PATH="yolov8n.pt"
export APP_SUBDOMAIN="xray-benchmark-test"

# Deploy
python cai_integration/deploy_yolo_application.py
```

**3. Test Infrastructure Only**

```bash
# Test health endpoint
curl https://xray-benchmark-test.[your-domain]/health

# Test API structure (results won't be meaningful for X-rays)
curl -X POST https://xray-benchmark-test.[your-domain]/v1/detect \
  -F "file=@any_image.jpg" \
  | jq .
```

---

## Comparison

| Aspect | Option 1: Pre-trained COCO | Option 2: Quick Train CargoXray |
|--------|---------------------------|--------------------------------|
| **Setup Time** | 2 min | 30-40 min |
| **X-ray Detection** | ❌ Poor | ✅ Good |
| **Threat Detection** | ❌ No | ✅ Yes |
| **Benchmark Value** | Infrastructure only | Full system |
| **Recommendation** | Testing only | ✅ **Recommended** |

---

## Benchmark Metrics to Collect

### 1. API Performance

```bash
# Latency test
for i in {1..100}; do
  curl -X POST https://xray-benchmark.[domain]/v1/detect \
    -F "file=@test_image.jpg" \
    -w "\nTime: %{time_total}s\n" \
    -o /dev/null -s
done | grep "Time:" | awk '{sum+=$2; count++} END {print "Avg: " sum/count "s"}'
```

**Expected Results** (with GPU):
- Average latency: 20-50ms
- Throughput: 20-50 images/second
- Memory: 2-4GB GPU RAM

### 2. Detection Accuracy (Option 2 only)

Test on validation set:

```bash
# Create test script
cat > test_benchmark.py << 'EOF'
import requests
from pathlib import Path
import json
import time

API_URL = "https://xray-benchmark.[domain]/v1/detect"
TEST_DIR = Path("data/cargoxray_yolo/images/test")

results = []
for img_path in list(TEST_DIR.glob("*.jpg"))[:50]:
    start = time.time()
    
    with open(img_path, 'rb') as f:
        response = requests.post(
            API_URL,
            files={'file': f}
        )
    
    latency = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        results.append({
            'image': img_path.name,
            'detections': len(data['items']),
            'latency': latency
        })
    
    print(f"✓ {img_path.name}: {len(data['items'])} items in {latency:.3f}s")

# Summary
print(f"\nSummary:")
print(f"  Images tested: {len(results)}")
print(f"  Avg detections: {sum(r['detections'] for r in results) / len(results):.1f}")
print(f"  Avg latency: {sum(r['latency'] for r in results) / len(results):.3f}s")
EOF

python test_benchmark.py
```

### 3. Resource Usage

Monitor in CAI UI:
1. Applications → xray-benchmark → Metrics
2. Check:
   - CPU usage
   - GPU utilization
   - Memory consumption
   - Request rate

---

## After Benchmarking

### Compare with Production Training

After benchmarking with CargoXray, train on larger dataset:

```bash
# Train on Luggage X-ray (recommended for production)
# Via GitHub Actions:
model_type: yolo
dataset: luggage_xray
yolo_model: yolov8s.pt    # Better accuracy
yolo_epochs: 100          # Full training
deploy_api: true
api_subdomain: xray-yolo-v1
```

**Compare metrics**:
- Accuracy improvement
- Inference speed
- Model size

### Save Benchmark Results

```bash
# Create benchmark report
cat > benchmark_results.md << EOF
# YOLO Deployment Benchmark Results

## Configuration
- Model: yolov8n (CargoXray)
- Dataset: CargoXray (659 images)
- Training time: 30 minutes
- Deployment time: 3 minutes

## Performance Metrics
- Average latency: [X]ms
- Throughput: [Y] images/sec
- GPU memory: [Z]GB
- Detection accuracy: [A]%

## Infrastructure
- CAI Application: Running
- Endpoints: All working
- Health check: ✅
- API docs: ✅

## Next Steps
- Train on larger dataset (luggage_xray)
- Optimize thresholds
- Test with real-world images
EOF
```

---

## Quick Start Commands

### Recommended (Option 2):

```bash
# 1. Trigger via GitHub Actions
# Actions → Deploy X-ray Detection to CAI → Run workflow
# Set: dataset=cargoxray, yolo_epochs=50, deploy_api=true

# 2. Wait ~30-40 minutes

# 3. Test
curl https://xray-benchmark.[domain]/health
curl -X POST https://xray-benchmark.[domain]/v1/detect \
  -F "file=@test_image.jpg" | jq .
```

### Infrastructure Test Only (Option 1):

```bash
# 1. Download model
source .venv/bin/activate
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 2. Deploy
export CAI_API_KEY="your-key"
export CAI_DOMAIN="https://ml-xxx.cloudera.site"
python cai_integration/deploy_yolo_application.py --model yolov8n.pt

# 3. Test infrastructure
curl https://xray-benchmark-test.[domain]/health
```

---

## Troubleshooting

### "Model not found"

For Option 1 (pre-trained):
```bash
# Verify model downloaded
python -c "from ultralytics import YOLO; print(YOLO('yolov8n.pt'))"

# Check cache location
ls -la ~/.cache/ultralytics/
```

For Option 2 (trained):
```bash
# Check training completed
# CAI UI → Jobs → yolo_training → Check status

# Verify model exists
# CAI UI → Files → runs/detect/*/weights/best.pt
```

### Slow inference

```bash
# Check GPU is being used
curl https://xray-benchmark.[domain]/health
# Should show: "backend": "ultralytics", GPU should be active

# Check application logs
# CAI UI → Applications → Logs
# Look for: "CUDA available: True"
```

---

## Summary

**Recommended Approach**:
1. ✅ Use Option 2 (Quick train on CargoXray - 30 min)
2. ✅ Deploy automatically via GitHub Actions
3. ✅ Benchmark with real X-ray images
4. ✅ Collect performance metrics
5. ✅ Compare with production training

**Option 1** is only useful for testing infrastructure, not for X-ray detection benchmarking.

---

*Start your benchmark now!* 🚀
