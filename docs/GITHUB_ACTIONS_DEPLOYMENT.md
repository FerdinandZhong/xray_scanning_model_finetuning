# GitHub Actions - Automated YOLO Deployment

Complete guide for using GitHub Actions to automatically train and deploy YOLO models to CAI.

---

## Overview

The GitHub Actions workflow now includes **automatic API deployment** after training completes. This means:

✅ **Train** YOLO model  
✅ **Deploy** as CAI Application  
✅ **Expose** HTTPS endpoints  
✅ **All automated** in one workflow run

---

## Prerequisites

### 1. GitHub Repository Secrets

Add these secrets to your GitHub repository:

| Secret | Value | Description |
|--------|-------|-------------|
| `CML_API_KEY` | Your CAI API key | For CAI authentication |
| `CML_HOST` | `https://ml-xxx.cloudera.site` | Your CAI workspace URL |
| `GH_PAT` | GitHub Personal Access Token | For Git operations (optional) |

**How to add secrets:**
1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add each secret

**How to get CAI API Key:**
1. Log into CAI workspace
2. User Settings → API Keys
3. Create new API key
4. Copy and add to GitHub secrets

---

## Quick Start - Using GitHub Actions UI

### Step 1: Go to Actions Tab

1. Navigate to your GitHub repository
2. Click **Actions** tab at the top

### Step 2: Select Workflow

1. Find "Deploy X-ray Detection to CAI" workflow
2. Click on it

### Step 3: Run Workflow

1. Click **Run workflow** button (top right)
2. Configure parameters:

#### Required Parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **model_type** | `yolo` | Use YOLO detection model |
| **dataset** | `luggage_xray` | Recommended dataset (7k images) |
| **trigger_jobs** | `true` | Auto-trigger training pipeline |
| **deploy_api** | `true` ✅ | Deploy API after training |

#### Optional Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **yolo_model** | `yolov8n.pt` | Model variant (n=fast, s=balanced, m=accurate) |
| **yolo_epochs** | `100` | Training epochs |
| **export_onnx** | `false` | Export to ONNX format |
| **api_subdomain** | `xray-yolo-api` | Custom subdomain for API |
| **force_reinstall** | `false` | Force environment reinstall |

### Step 4: Monitor Execution

1. Click on the running workflow
2. Watch each job complete:
   - ✅ validate
   - ✅ setup-project
   - ✅ create-jobs
   - ✅ **deploy-yolo-api** ← Configures deployment
   - ✅ trigger-pipeline

### Step 5: Monitor Training in CAI

The workflow triggers training in CAI. Monitor progress:

1. Go to your CAI workspace
2. Navigate to **Jobs** → **Job Runs**
3. Watch jobs execute in order:
   - `setup_environment` (10-20 min)
   - `download_luggage_xray` (15-30 min)
   - `yolo_training` (1-2 hours)
   - `deploy_yolo_application` ← **API deployment** (2-5 min)

### Step 6: Access Your Deployed API

After all jobs complete (~2-3 hours):

```bash
# Check health
curl https://xray-yolo-api.[your-domain]/health

# View docs
open https://xray-yolo-api.[your-domain]/docs

# Test detection
curl -X POST https://xray-yolo-api.[your-domain]/v1/detect \
  -F "file=@image.jpg"
```

---

## Configuration Examples

### Example 1: Fast Training (30 minutes)

**Use Case**: Quick testing, baseline model

```yaml
model_type: yolo
dataset: cargoxray        # Small dataset (659 images)
yolo_model: yolov8n.pt    # Fast model
yolo_epochs: 50           # Fewer epochs
trigger_jobs: true
deploy_api: true
api_subdomain: xray-yolo-quick
```

### Example 2: Production Training (2 hours)

**Use Case**: Production-ready model

```yaml
model_type: yolo
dataset: luggage_xray     # Medium dataset (7k images)
yolo_model: yolov8s.pt    # Balanced model
yolo_epochs: 100          # Full training
export_onnx: true         # Export for production
trigger_jobs: true
deploy_api: true
api_subdomain: xray-yolo-v1
```

### Example 3: High Accuracy (4-8 hours)

**Use Case**: Maximum accuracy needed

```yaml
model_type: yolo
dataset: stcray           # Large dataset (46k images)
yolo_model: yolov8m.pt    # Accurate model
yolo_epochs: 100
export_onnx: true
trigger_jobs: true
deploy_api: true
api_subdomain: xray-yolo-production
```

### Example 4: Deployment Only

**Use Case**: Deploy already-trained model

```yaml
model_type: yolo
dataset: luggage_xray
trigger_jobs: false       # Don't train
deploy_api: true          # Only deploy
api_subdomain: xray-yolo-api
```

Then manually deploy:
```bash
python cai_integration/deploy_yolo_application.py \
  --model "runs/detect/xray_detection/weights/best.pt"
```

---

## Workflow Architecture

```
GitHub Actions Workflow
├─ validate (5 min)
│   └─ Check configs, validate YAML
├─ setup-project (10 min)
│   └─ Create/update CAI project
├─ create-jobs (5 min)
│   └─ Create CAI jobs with parameters
├─ deploy-yolo-api (2 min)  ← NEW!
│   └─ Configure deployment job in CAI
└─ trigger-pipeline (1 min)
    └─ Start training pipeline

CAI Job Pipeline (runs after GitHub Actions)
├─ setup_environment (10-20 min)
│   └─ Install dependencies
├─ download_luggage_xray (15-30 min)
│   └─ Download dataset
├─ yolo_training (1-4 hours)
│   └─ Train YOLO model
└─ deploy_yolo_application (2-5 min)  ← NEW!
    └─ Deploy as CAI Application
        ├─ Create application
        ├─ Configure resources
        ├─ Start FastAPI server
        └─ Expose HTTPS endpoints
```

---

## Deployment Job Details

### What Happens During Deployment

1. **Find Latest Model**
   - Auto-detects `runs/detect/*/weights/best.pt`
   - Uses latest trained model

2. **Create CAI Application**
   - Name: `xray-yolo-detection-api`
   - Resources: 4 CPU, 16GB RAM, 1 GPU
   - Runtime: Python 3.10 with CUDA

3. **Configure Environment**
   - Sets MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD
   - Configures port 8080 (CAI standard)
   - Sets up GPU device

4. **Start FastAPI Server**
   - Loads YOLO model
   - Initializes detection engine
   - Exposes endpoints

5. **Make Available**
   - Public HTTPS endpoint
   - Accessible at: `https://[subdomain].[domain]`

### Resource Allocation

| Resource | Value | Purpose |
|----------|-------|---------|
| CPU | 4 cores | Request handling |
| Memory | 16 GB | Model loading |
| GPU | 1x NVIDIA | Fast inference |
| Disk | 10 GB | Model storage |

---

## Monitoring and Verification

### 1. Check GitHub Actions Status

In GitHub Actions:
- All jobs should show ✅ green checkmarks
- Review "Summary" for deployment details
- Check logs if any issues

### 2. Monitor CAI Jobs

In CAI UI:
1. **Jobs** → **Job Runs**
2. Check status of each job
3. View logs for details

**Expected Timeline:**
- setup_environment: 10-20 min
- download_luggage_xray: 15-30 min
- yolo_training: 1-2 hours (luggage_xray)
- deploy_yolo_application: 2-5 min

### 3. Verify Application Deployment

In CAI UI:
1. **Applications** tab
2. Find "xray-yolo-detection-api"
3. Status should show: **Running** ✅

### 4. Test Endpoints

```bash
# 1. Health check (should return immediately)
curl https://xray-yolo-api.[your-domain]/health

Expected:
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "ultralytics"
}

# 2. Test detection (with sample image)
curl -X POST https://xray-yolo-api.[your-domain]/v1/detect \
  -F "file=@data/luggage_xray_yolo/images/valid/valid_000001.jpg" \
  | jq .

Expected:
{
  "items": [...],
  "total_count": 1,
  "has_concealed_items": false
}

# 3. View interactive docs
open https://xray-yolo-api.[your-domain]/docs
```

---

## Troubleshooting

### GitHub Actions Fails

**Check:**
1. Secrets are configured correctly (`CML_API_KEY`, `CML_HOST`)
2. YAML configuration is valid
3. GitHub Actions logs for error details

**Common Issues:**
```
Error: Authentication failed
→ Solution: Verify CML_API_KEY is correct

Error: Project not found
→ Solution: Check CML_HOST URL format

Error: Job creation failed
→ Solution: Check CAI workspace has sufficient resources
```

### Training Job Fails

**Check CAI Job Logs:**
1. Jobs → Job Runs → [failed job] → Logs

**Common Issues:**
```
ERROR: CUDA out of memory
→ Solution: Use smaller model (yolov8n) or reduce batch size

ERROR: Dataset not found
→ Solution: Check download job completed successfully

ERROR: Package not found
→ Solution: Enable force_reinstall in workflow
```

### Deployment Job Fails

**Check:**
1. CAI Applications → xray-yolo-detection-api → Logs
2. Model file exists: `runs/detect/*/weights/best.pt`
3. Sufficient GPU resources available

**Common Issues:**
```
ERROR: Model not found
→ Solution: Verify training completed successfully

ERROR: Application already exists
→ Solution: Script will update existing application

ERROR: Port 8080 already in use
→ Solution: Stop conflicting application in CAI
```

### API Not Responding

**Check:**
1. Application status is "Running" in CAI UI
2. Application logs for startup errors
3. Correct subdomain in URL

**Solutions:**
```bash
# Restart application
1. CAI UI → Applications → xray-yolo-detection-api
2. Click "Restart"
3. Wait 1-2 minutes for startup

# Check logs
1. CAI UI → Applications → xray-yolo-detection-api
2. Click "Logs" tab
3. Look for errors during startup
```

---

## Advanced Configuration

### Custom Deployment Configuration

Edit `cai_integration/jobs_config_yolo.yaml`:

```yaml
deploy_yolo_application:
  environment:
    MODEL_PATH: ""  # Auto-detect or specify path
    BACKEND: "ultralytics"  # or "onnx"
    CONF_THRESHOLD: "0.25"  # Lower = more detections
    IOU_THRESHOLD: "0.45"   # NMS threshold
    DEVICE: "0"            # GPU device ID
    APP_SUBDOMAIN: "xray-yolo-api"  # Custom subdomain
```

### Multiple Deployments

Deploy multiple versions:

```bash
# Deploy version 1
Run workflow with api_subdomain: xray-yolo-v1

# Deploy version 2 (later)
Run workflow with api_subdomain: xray-yolo-v2

# Both APIs will be available:
https://xray-yolo-v1.[domain]/v1/detect
https://xray-yolo-v2.[domain]/v1/detect
```

### A/B Testing

Deploy two models and compare:

```bash
# Train with yolov8n
api_subdomain: xray-yolo-fast

# Train with yolov8m
api_subdomain: xray-yolo-accurate

# Test both and compare results
```

---

## Best Practices

### 1. Start Small
- Use `cargoxray` dataset first (30 min)
- Verify workflow works end-to-end
- Then scale to larger datasets

### 2. Monitor Resources
- Check CAI resource usage
- Ensure sufficient GPU availability
- Scale down unused applications

### 3. Version Control
- Use descriptive subdomains: `xray-yolo-v1`, `xray-yolo-prod`
- Tag models in Git after successful deployment
- Keep deployment logs for debugging

### 4. Testing
- Always test health endpoint first
- Verify API docs are accessible
- Test with sample images before production use

### 5. Security
- Keep API keys secure in GitHub Secrets
- Don't commit secrets to repository
- Monitor API usage for abuse
- Consider adding authentication layer

---

## Cost Optimization

### Reduce Training Time
```yaml
dataset: cargoxray     # Smaller dataset
yolo_model: yolov8n.pt # Faster model
yolo_epochs: 50        # Fewer epochs
```

### Reduce Deployment Resources
Edit application configuration:
```yaml
cpu: 2      # Fewer cores
memory: 8   # Less RAM
```

### Auto-scaling
- Enable in CAI UI
- Application scales down when idle
- Reduces costs during low traffic

---

## Next Steps

After successful deployment:

1. **Test API thoroughly**
   - Try different images
   - Verify detection accuracy
   - Check response times

2. **Integrate with application**
   - Use `/v1/detect` endpoint
   - Handle responses in your code
   - Display bounding boxes

3. **Monitor performance**
   - Track inference latency
   - Monitor GPU utilization
   - Collect detection metrics

4. **Iterate and improve**
   - Collect edge cases
   - Retrain with new data
   - Deploy updated models

---

## Support

**Issues?**
1. Check GitHub Actions logs
2. Review CAI job logs
3. Read [YOLO_CAI_DEPLOYMENT.md](YOLO_CAI_DEPLOYMENT.md)
4. Check [DEPLOYMENT_QUICK_START.md](DEPLOYMENT_QUICK_START.md)

**Questions?**
- GitHub Issues: [Create issue](https://github.com/your-repo/issues)
- Documentation: [README.md](../README.md)

---

**Automated deployment is now live!** 🚀

Train and deploy YOLO models with a single workflow run.
