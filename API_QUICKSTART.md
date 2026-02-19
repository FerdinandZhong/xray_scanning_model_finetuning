# YOLO X-ray API - Quick Start

Your API is deployed and ready! 🎉

**Base URL**: https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site

---

## Quick Test (30 seconds)

### 1. Check Health

```bash
curl https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/health
```

Expected: `{"status":"healthy","model_loaded":true,"backend":"ultralytics"}`

### 2. Test Detection

```bash
# Download a sample X-ray image
curl -o test_xray.jpg https://raw.githubusercontent.com/ultralytics/assets/main/yolo/v8/person.jpg

# Test detection endpoint
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/detect \
  -F "file=@test_xray.jpg"
```

### 3. Run Full Test Suite

```bash
# If you have test images in data/
python scripts/test_deployed_yolo_api.py data/luggage_xray_yolo/test/images/xray_001.jpg

# Or let it auto-find a test image
python scripts/test_deployed_yolo_api.py
```

---

## API Endpoints

### 1. `/v1/detect` - Direct Detection

**Request:**
```bash
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/detect \
  -F "file=@your_xray.jpg"
```

**Response:**
```json
{
  "detections": [
    {"class_id": 3, "label": "Knife", "confidence": 0.89, "bbox": {...}}
  ],
  "num_detections": 1,
  "threats_detected": ["Knife"],
  "is_threat": true
}
```

### 2. `/v1/chat/completions` - OpenAI Format

**Request:**
```bash
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/chat/completions \
  -F "file=@your_xray.jpg"
```

**Response:**
```json
{
  "id": "chatcmpl-yolo-1739291234",
  "model": "yolov8x.pt",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "{\"detections\":[...],\"is_threat\":true}"
    }
  }]
}
```

---

## Python Integration

```python
import requests
import json

url = "https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/detect"

with open("xray_scan.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()
if result['is_threat']:
    print(f"⚠️ THREAT: {', '.join(result['threats_detected'])}")
else:
    print("✓ No threats detected")
```

---

## Interactive Documentation

Open in browser: **https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/docs**

Try the API interactively with the "Try it out" button!

---

## Performance Notes

- **First request**: ~30 seconds (model initialization)
- **Subsequent requests**: <100ms with GPU
- **Model**: yolov8x.pt (pre-trained on COCO, not X-ray specific)

⚠️ **For production X-ray detection**, train on the luggage dataset and redeploy!

---

## Full Documentation

- **API Testing Guide**: `docs/API_TESTING_GUIDE.md`
- **Deployment Guide**: `docs/YOLO_CAI_DEPLOYMENT.md`
- **Quick Start**: `docs/DEPLOYMENT_QUICK_START.md`
