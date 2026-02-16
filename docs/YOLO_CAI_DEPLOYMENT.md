# YOLO API Deployment on CAI

Complete guide for deploying the trained YOLO model as a persistent CAI Application with exposed endpoints.

---

## Overview

This deployment creates a **CAI Application** that:
- ✅ Runs persistently (survives restarts)
- ✅ Exposes public HTTPS endpoints
- ✅ Auto-scales based on traffic
- ✅ Provides OpenAI-compatible API
- ✅ Includes interactive API documentation
- ✅ Supports GPU acceleration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CAI Application                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ FastAPI Server (uvicorn)                               │ │
│  │   ├─ /health (Health check)                            │ │
│  │   ├─ /docs (Swagger UI)                                │ │
│  │   ├─ /v1/chat/completions (OpenAI-compatible)          │ │
│  │   └─ /v1/detect (Direct detection)                     │ │
│  │                                                          │ │
│  │ YOLO Detection Engine                                   │ │
│  │   ├─ Ultralytics YOLO backend                          │ │
│  │   ├─ GPU inference (CUDA)                              │ │
│  │   └─ Trained model (best.pt)                           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↓
              HTTPS://xray-yolo-api.[domain]
```

---

## Prerequisites

### 1. Trained YOLO Model

You need a trained YOLO model in `runs/detect/*/weights/best.pt`. Train one using:

```bash
# Option A: Using GitHub Actions (recommended)
# Go to Actions tab → Deploy X-ray Detection to CAI → Run workflow

# Option B: Using CAI Jobs
python cai_integration/trigger_jobs.py \
  --config cai_integration/jobs_config_yolo.yaml

# Option C: Local training then upload
python training/train_yolo.py --data data/luggage_xray_yolo/data.yaml
```

### 2. CAI Credentials

Required environment variables:
- `CAI_API_KEY` - Your CAI API key
- `CAI_DOMAIN` - Your CAI workspace URL (e.g., https://ml-xxx.cloudera.site)
- `CAI_PROJECT_NAME` - Project name (optional, auto-detects if not provided)

**How to get CAI API Key:**
1. Log into CAI workspace
2. Go to User Settings → API Keys
3. Create new API key
4. Copy and save securely

---

## Deployment Methods

### Method 1: Automated via GitHub Actions (Recommended)

The GitHub Actions workflow automatically deploys after training completes.

**Setup:**

1. Add secrets to your GitHub repository:
   - Go to Settings → Secrets and variables → Actions
   - Add: `CAI_API_KEY`, `CAI_DOMAIN`, `CAI_PROJECT_NAME`

2. Trigger deployment:
   ```bash
   # Go to Actions tab → Deploy X-ray Detection to CAI
   # Set trigger_jobs: true
   # Deployment happens automatically after training
   ```

### Method 2: Manual via Python Script

Deploy an already-trained model:

```bash
# Activate environment
source .venv/bin/activate  # or .venv_yolo/bin/activate

# Deploy with auto-detected model
python cai_integration/deploy_yolo_application.py \
  --api-key "YOUR_CAI_API_KEY" \
  --domain "https://ml-xxx.cloudera.site" \
  --project "xray-scanning"

# Or specify model explicitly
python cai_integration/deploy_yolo_application.py \
  --api-key "YOUR_CAI_API_KEY" \
  --domain "https://ml-xxx.cloudera.site" \
  --model runs/detect/xray_detection_luggage_xray/weights/best.pt \
  --subdomain "xray-yolo-v2"
```

**Using environment variables:**

```bash
export CAI_API_KEY="your-api-key"
export CAI_DOMAIN="https://ml-xxx.cloudera.site"
export CAI_PROJECT_NAME="xray-scanning"
export MODEL_PATH="runs/detect/xray_detection_luggage_xray/weights/best.pt"
export APP_SUBDOMAIN="xray-yolo-api"

python cai_integration/deploy_yolo_application.py
```

### Method 3: Via CAI Jobs

Add to your job pipeline in `jobs_config_yolo.yaml`:

```yaml
jobs:
  # ... other jobs (setup, download, train) ...
  
  deploy_yolo_application:
    name: "Deploy YOLO API Application"
    script: "cai_integration/deploy_yolo_application.py"
    parent_job_key: "yolo_training"
    # ... configuration ...
```

Then trigger:

```bash
python cai_integration/trigger_jobs.py \
  --config cai_integration/jobs_config_yolo.yaml
```

---

## Application Configuration

### Environment Variables

Configure the application behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | Auto-detect | Path to trained model (.pt file) |
| `BACKEND` | `ultralytics` | Inference backend (`ultralytics` or `onnx`) |
| `CONF_THRESHOLD` | `0.25` | Confidence threshold (0.0-1.0) |
| `IOU_THRESHOLD` | `0.45` | IOU threshold for NMS (0.0-1.0) |
| `DEVICE` | `0` | GPU device (`0`, `cpu`, etc.) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8080` | Server port (CAI uses 8080) |

### Resource Allocation

**Default Resources:**
- CPU: 4 cores
- Memory: 16 GB
- GPU: 1x NVIDIA GPU

**To modify** (edit `launch_yolo_application.sh` or create custom application):
```python
app_config = {
    "cpu": 8,      # More CPU cores
    "memory": 32,  # More RAM (GB)
    "gpu": 1,      # Number of GPUs
}
```

---

## Using the Deployed API

### Health Check

```bash
curl https://xray-yolo-api.[your-domain]/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "ultralytics"
}
```

### Direct Detection Endpoint

```bash
curl -X POST https://xray-yolo-api.[your-domain]/v1/detect \
  -F "file=@data/luggage_xray_yolo/images/valid/valid_000001.jpg"
```

**Expected Response:**
```json
{
  "items": [
    {
      "name": "scissors",
      "confidence": 0.95,
      "location": "center",
      "bbox": [0.5, 0.5, 0.2, 0.3]
    }
  ],
  "total_count": 1,
  "has_concealed_items": false
}
```

### OpenAI-Compatible Endpoint

```bash
curl -X POST https://xray-yolo-api.[your-domain]/v1/chat/completions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

**Expected Response:**
```json
{
  "id": "chatcmpl-yolo-1707799200",
  "object": "chat.completion",
  "created": 1707799200,
  "model": "runs/detect/xray_detection/weights/best.pt",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"items\":[...],\"total_count\":1,\"has_concealed_items\":false}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Interactive API Documentation

Visit: `https://xray-yolo-api.[your-domain]/docs`

Features:
- 📖 Complete API reference
- 🧪 Interactive testing (try API in browser)
- 📋 Request/response schemas
- 🔍 Example requests

---

## Testing the Deployment

### 1. Quick Health Check

```bash
# Check if application is running
curl https://xray-yolo-api.[your-domain]/health

# Should return: {"status":"healthy","model_loaded":true}
```

### 2. Test Detection

```bash
# Test with a validation image
curl -X POST https://xray-yolo-api.[your-domain]/v1/detect \
  -F "file=@data/luggage_xray_yolo/images/valid/valid_000001.jpg" \
  | jq .
```

### 3. Performance Test

```bash
# Test multiple images
for i in {1..10}; do
  time curl -X POST https://xray-yolo-api.[your-domain]/v1/detect \
    -F "file=@data/luggage_xray_yolo/images/valid/valid_00000$i.jpg" \
    -o /dev/null -s
done
```

---

## Monitoring and Maintenance

### View Application Logs

**In CAI UI:**
1. Go to Applications
2. Select "xray-yolo-detection-api"
3. Click "Logs" tab

**Via CLI:**
```bash
# Coming soon: CLI tool for fetching logs
```

### Application Status

Check application status in CAI UI:
- **Running** ✅ - Application is healthy
- **Starting** 🔄 - Wait 1-2 minutes
- **Stopped** 🛑 - Check logs for errors
- **Failed** ❌ - Review logs and fix issues

### Restart Application

**Via UI:**
1. Go to Applications
2. Select application
3. Click "Restart"

**Via Script:**
```bash
# Redeploy (updates + restarts)
python cai_integration/deploy_yolo_application.py
```

### Update Model

To deploy a newly trained model:

```bash
# Train new model
python training/train_yolo.py --data data/luggage_xray_yolo/data.yaml

# Redeploy application (will use latest model)
python cai_integration/deploy_yolo_application.py \
  --model runs/detect/xray_detection2/weights/best.pt
```

---

## Performance Tuning

### 1. Adjust Confidence Threshold

Lower threshold = more detections (higher recall, more false positives)
Higher threshold = fewer detections (higher precision, missed items)

```bash
# Deploy with custom threshold
python cai_integration/deploy_yolo_application.py
# Then update environment variables in CAI UI
# Set CONF_THRESHOLD=0.30
```

### 2. GPU Selection

```bash
# Use specific GPU
# Set DEVICE=0 (or 1, 2, etc.) in environment variables
```

### 3. Model Optimization

**Export to ONNX** for faster inference:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/xray_detection/weights/best.pt')
model.export(format='onnx')

# Deploy ONNX model
export BACKEND=onnx
export MODEL_PATH=runs/detect/xray_detection/weights/best.onnx
python cai_integration/deploy_yolo_application.py
```

---

## Troubleshooting

### Application Won't Start

**Check logs for errors:**
1. Go to CAI UI → Applications → xray-yolo-detection-api → Logs
2. Look for Python errors or missing dependencies

**Common issues:**
```
ERROR: Model not found
→ Solution: Verify MODEL_PATH is correct

ERROR: CUDA out of memory
→ Solution: Reduce batch size or use smaller model (yolov8n)

ERROR: Import ultralytics failed
→ Solution: Check that .venv includes ultralytics package
```

### Slow Inference

**Optimize performance:**
1. Use GPU instead of CPU (`DEVICE=0`)
2. Export to ONNX format
3. Use smaller model (yolov8n vs yolov8m)
4. Increase GPU resources in application config

### Application Keeps Restarting

**Check resource limits:**
1. Monitor GPU memory usage
2. Increase memory allocation if needed
3. Review error logs for OOM (out of memory) errors

---

## Security Considerations

### 1. API Authentication

**Current**: No authentication (add via middleware)

**Recommended**: Add authentication layer:
```python
# Add to yolo_api_server.py
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/v1/detect")
async def detect(
    file: UploadFile,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    # Validate token
    if not validate_token(credentials.credentials):
        raise HTTPException(401, "Invalid token")
    # ... detection logic ...
```

### 2. Rate Limiting

**Add rate limiting** to prevent abuse:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/detect")
@limiter.limit("100/minute")
async def detect(request: Request, file: UploadFile):
    # ... detection logic ...
```

### 3. Input Validation

**Validate image uploads:**
- Maximum file size: 10 MB
- Allowed formats: JPEG, PNG
- Image dimensions: 640-4096 px

---

## Cost Optimization

### 1. Auto-scaling

Configure auto-scaling based on traffic (CAI feature).

### 2. Use Appropriate Model Size

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolov8n | ⚡⚡⚡ | ⭐⭐ | High-throughput screening |
| yolov8s | ⚡⚡ | ⭐⭐⭐ | Balanced (recommended) |
| yolov8m | ⚡ | ⭐⭐⭐⭐ | High-accuracy needs |

### 3. Batch Processing

For bulk processing, use batch inference:
```python
# Process multiple images in one request
results = model.predict([img1, img2, img3], batch=True)
```

---

## Next Steps

After successful deployment:

1. **Integrate with Frontend**:
   - Use `/v1/detect` endpoint from web app
   - Display bounding boxes and confidence scores

2. **Monitor Performance**:
   - Track inference latency
   - Monitor GPU utilization
   - Log detection results

3. **Continuous Improvement**:
   - Collect edge cases
   - Retrain model with new data
   - A/B test different models

---

## Support

**Issues?** Check:
1. Application logs in CAI UI
2. [Troubleshooting](#troubleshooting) section above
3. [GitHub Issues](https://github.com/your-repo/issues)

**Questions?** Contact:
- Project lead: [email]
- Documentation: [README.md](../README.md)

---

*Last updated: February 2026*
