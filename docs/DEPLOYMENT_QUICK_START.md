# YOLO CAI Deployment - Quick Start

Fast guide to deploy YOLO model as CAI Application.

---

## What Was Created

✅ **CAI Application Deployment Script** (`cai_integration/deploy_yolo_application.py`)
- Automatically creates/updates CAI Application
- Configures resources (CPU, Memory, GPU)
- Exposes HTTPS endpoints
- Finds latest trained model automatically

✅ **Application Launcher** (`cai_integration/launch_yolo_application.sh`)
- Starts FastAPI server in CAI
- Loads trained YOLO model
- Configures endpoints
- Runs on port 8100 (CAI standard)

✅ **Updated Jobs Configuration** (`cai_integration/jobs_config_yolo.yaml`)
- Added `deploy_yolo_application` job
- Automatically deploys after training
- Configurable via environment variables

✅ **Comprehensive Documentation** (`docs/YOLO_CAI_DEPLOYMENT.md`)
- Full deployment guide
- API usage examples
- Troubleshooting tips
- Security & optimization

---

## Prerequisites

1. **Trained YOLO Model**
   ```bash
   # Train via GitHub Actions or:
   python training/train_yolo.py --data data/luggage_xray_yolo/data.yaml
   ```

2. **CAI Credentials**
   ```bash
   export CAI_API_KEY="your-api-key"
   export CAI_DOMAIN="https://ml-xxx.cloudera.site"
   export CAI_PROJECT_NAME="your-project"
   ```

---

## Deploy in 3 Steps

### Step 1: Activate Environment

```bash
source .venv/bin/activate  # or .venv_yolo/bin/activate
```

### Step 2: Run Deployment Script

```bash
python cai_integration/deploy_yolo_application.py
```

**Or with explicit parameters:**

```bash
python cai_integration/deploy_yolo_application.py \
  --api-key "YOUR_API_KEY" \
  --domain "https://ml-xxx.cloudera.site" \
  --project "xray-scanning" \
  --model "runs/detect/xray_detection_luggage_xray/weights/best.pt" \
  --subdomain "xray-yolo-api"
```

### Step 3: Wait & Test

**Wait 1-2 minutes** for application to start, then test:

```bash
# Health check
curl https://xray-yolo-api.[your-domain]/health

# Test detection
curl -X POST https://xray-yolo-api.[your-domain]/v1/detect \
  -F "file=@data/luggage_xray_yolo/images/valid/valid_000001.jpg"

# View API docs
open https://xray-yolo-api.[your-domain]/docs
```

---

## Available Endpoints

Once deployed, your application exposes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs |
| `/v1/detect` | POST | Direct detection (recommended) |
| `/v1/chat/completions` | POST | OpenAI-compatible format |

---

## Configuration

### Change Detection Thresholds

Edit environment variables in CAI UI or redeploy:

```bash
export CONF_THRESHOLD=0.30  # Lower = more detections
export IOU_THRESHOLD=0.45   # NMS threshold
python cai_integration/deploy_yolo_application.py
```

### Use Different Model

```bash
python cai_integration/deploy_yolo_application.py \
  --model "runs/detect/xray_detection2/weights/best.pt"
```

### Custom Subdomain

```bash
python cai_integration/deploy_yolo_application.py \
  --subdomain "xray-detection-v2"
```

---

## Integration with GitHub Actions

The deployment is **automatically triggered** after training completes when using GitHub Actions.

**Setup:**
1. Add GitHub Secrets: `CAI_API_KEY`, `CAI_DOMAIN`, `CAI_PROJECT_NAME`
2. Run workflow: Actions → Deploy X-ray Detection to CAI
3. Set `trigger_jobs: true`
4. Application deploys automatically after training

---

## Verification Checklist

After deployment, verify:

- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ API docs accessible at `/docs`
- ✅ Test detection works on sample image
- ✅ Application status shows "Running" in CAI UI
- ✅ Response time < 100ms (with GPU)

---

## Common Issues & Quick Fixes

### "Model not found"
```bash
# Check available models
find runs/detect -name "best.pt"

# Specify model explicitly
python cai_integration/deploy_yolo_application.py \
  --model "runs/detect/xray_detection_luggage_xray/weights/best.pt"
```

### "CAI_API_KEY not set"
```bash
# Set environment variable
export CAI_API_KEY="your-api-key"

# Or pass as argument
python cai_integration/deploy_yolo_application.py --api-key "your-api-key"
```

### Application won't start
1. Check logs in CAI UI (Applications → xray-yolo-detection-api → Logs)
2. Verify model file exists
3. Ensure sufficient GPU memory
4. Restart application in CAI UI

---

## Next Steps

1. **Test API**: `curl https://xray-yolo-api.[domain]/health`
2. **Read Full Guide**: [YOLO_CAI_DEPLOYMENT.md](YOLO_CAI_DEPLOYMENT.md)
3. **Integrate**: Use `/v1/detect` endpoint in your application
4. **Monitor**: Check logs and metrics in CAI UI

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `cai_integration/deploy_yolo_application.py` | Deployment script |
| `cai_integration/launch_yolo_application.sh` | Application launcher |
| `cai_integration/jobs_config_yolo.yaml` | Jobs configuration |
| `inference/yolo_api_server.py` | FastAPI server |
| `docs/YOLO_CAI_DEPLOYMENT.md` | Full documentation |

---

**Need Help?** See [YOLO_CAI_DEPLOYMENT.md](YOLO_CAI_DEPLOYMENT.md) for comprehensive guide.
