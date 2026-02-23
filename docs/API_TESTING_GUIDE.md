# YOLO X-ray Detection API - Testing Guide

Your API is now live at:
**Base URL**: https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site

## Quick Health Check

```bash
# Test if the API is running
curl https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "ultralytics"
}
```

---

## 1. Direct Detection Endpoint: `/v1/detect`

This endpoint returns structured detection results directly.

### cURL Request

```bash
# With a local image file
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/detect \
  -F "file=@path/to/xray_image.jpg" \
  -H "Accept: application/json"

# Example with a luggage X-ray image
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/detect \
  -F "file=@data/luggage_xray_yolo/test/images/xray_001.jpg" \
  -H "Accept: application/json"
```

### Python Request

```python
import requests

# API endpoint
url = "https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/detect"

# Open and send image
with open("path/to/xray_image.jpg", "rb") as f:
    files = {"file": ("xray.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)

# Parse response
if response.status_code == 200:
    result = response.json()
    print(f"Found {len(result['detections'])} objects")
    
    for detection in result['detections']:
        print(f"  - {detection['label']}: {detection['confidence']:.2%}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Expected Response

```json
{
  "detections": [
    {
      "class_id": 3,
      "label": "Knife",
      "confidence": 0.89,
      "bbox": {
        "x_min": 245.3,
        "y_min": 178.2,
        "x_max": 312.8,
        "y_max": 289.6
      }
    },
    {
      "class_id": 7,
      "label": "Laptop",
      "confidence": 0.92,
      "bbox": {
        "x_min": 100.5,
        "y_min": 200.1,
        "x_max": 350.8,
        "y_max": 450.3
      }
    }
  ],
  "num_detections": 2,
  "threats_detected": ["Knife"],
  "is_threat": true
}
```

---

## 2. OpenAI-Compatible Endpoint: `/v1/chat/completions`

This endpoint wraps detection results in OpenAI's chat completion format for agentic workflows.

### cURL Request

```bash
# Basic request
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/chat/completions \
  -F "file=@path/to/xray_image.jpg"

# With explicit content type
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/chat/completions \
  -F "file=@data/luggage_xray_yolo/test/images/xray_001.jpg" \
  -H "Accept: application/json"
```

### Python Request (OpenAI Client Compatible)

```python
import requests

# API endpoint
url = "https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/chat/completions"

# Send image
with open("path/to/xray_image.jpg", "rb") as f:
    files = {"file": ("xray.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)

# Parse OpenAI-format response
if response.status_code == 200:
    completion = response.json()
    
    # Extract detection result from message content
    import json
    detection_result = json.loads(completion['choices'][0]['message']['content'])
    
    print(f"Model: {completion['model']}")
    print(f"Detections: {detection_result['num_detections']}")
    print(f"Threat Detected: {detection_result['is_threat']}")
    
    if detection_result['threats_detected']:
        print(f"Threats: {', '.join(detection_result['threats_detected'])}")
else:
    print(f"Error: {response.status_code}")
```

### Alternative: Using in Agentic Workflow

```python
# For AI agents using OpenAI SDK pattern
import requests
import json

def analyze_xray_with_yolo(image_path: str) -> dict:
    """
    Analyze X-ray image using YOLO detection API.
    Compatible with OpenAI-like workflows.
    """
    url = "https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/chat/completions"
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        response.raise_for_status()
    
    completion = response.json()
    detection_data = json.loads(completion['choices'][0]['message']['content'])
    
    return detection_data

# Use in your workflow
result = analyze_xray_with_yolo("scanned_bag.jpg")

if result['is_threat']:
    print(f"⚠️ THREAT DETECTED: {', '.join(result['threats_detected'])}")
    # Trigger alert, manual inspection, etc.
else:
    print("✓ No threats detected")
```

### Expected Response (OpenAI Format)

```json
{
  "id": "chatcmpl-yolo-1739291234",
  "object": "chat.completion",
  "created": 1739291234,
  "model": "yolov8x.pt",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"detections\":[{\"class_id\":3,\"label\":\"Knife\",\"confidence\":0.89,\"bbox\":{\"x_min\":245.3,\"y_min\":178.2,\"x_max\":312.8,\"y_max\":289.6}}],\"num_detections\":1,\"threats_detected\":[\"Knife\"],\"is_threat\":true}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

## 3. Complete Test Script

Save this as `test_yolo_api.py`:

```python
#!/usr/bin/env python3
"""Test script for deployed YOLO X-ray Detection API."""

import requests
import json
import sys
from pathlib import Path

BASE_URL = "https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site"

def test_health():
    """Test health endpoint."""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    return response.status_code == 200

def test_detect(image_path: str):
    """Test direct detection endpoint."""
    print("=" * 60)
    print("Testing /v1/detect Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/detect"
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    with open(image_path, "rb") as f:
        files = {"file": ("xray.jpg", f, "image/jpeg")}
        response = requests.post(url, files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Success!")
        print(f"  Detections: {result['num_detections']}")
        print(f"  Threat: {result['is_threat']}")
        
        if result['threats_detected']:
            print(f"  Threats: {', '.join(result['threats_detected'])}")
        
        print(f"\n  Objects found:")
        for det in result['detections']:
            print(f"    - {det['label']}: {det['confidence']:.2%}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_chat_completions(image_path: str):
    """Test OpenAI-compatible endpoint."""
    print("=" * 60)
    print("Testing /v1/chat/completions Endpoint")
    print("=" * 60)
    
    url = f"{BASE_URL}/v1/chat/completions"
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    with open(image_path, "rb") as f:
        files = {"file": ("xray.jpg", f, "image/jpeg")}
        response = requests.post(url, files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        completion = response.json()
        
        print(f"\n✓ Success!")
        print(f"  Model: {completion['model']}")
        print(f"  ID: {completion['id']}")
        
        # Parse detection result from message content
        detection_result = json.loads(completion['choices'][0]['message']['content'])
        
        print(f"\n  Detection Results:")
        print(f"    Detections: {detection_result['num_detections']}")
        print(f"    Threat: {detection_result['is_threat']}")
        
        if detection_result['threats_detected']:
            print(f"    Threats: {', '.join(detection_result['threats_detected'])}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False

if __name__ == "__main__":
    print("\n🚀 Testing YOLO X-ray Detection API\n")
    
    # Test health
    if not test_health():
        print("❌ Health check failed!")
        sys.exit(1)
    
    # Get test image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_paths = [
            "data/luggage_xray_yolo/test/images/xray_001.jpg",
            "data/cargoxray/test/images/P00001.png",
            "test_images/sample_xray.jpg"
        ]
        image_path = None
        for path in test_paths:
            if Path(path).exists():
                image_path = path
                break
        
        if not image_path:
            print("No test image provided. Usage:")
            print(f"  python test_yolo_api.py path/to/xray_image.jpg")
            sys.exit(1)
    
    print(f"Using test image: {image_path}\n")
    
    # Test both endpoints
    detect_ok = test_detect(image_path)
    print()
    chat_ok = test_chat_completions(image_path)
    
    print()
    print("=" * 60)
    if detect_ok and chat_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
```

---

## 4. Quick Test Commands

### Test Health (No image needed)

```bash
curl https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/health
```

### Test Detection with Sample Image

If you have test images:

```bash
# Direct detection
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/detect \
  -F "file=@data/luggage_xray_yolo/test/images/xray_001.jpg" | jq '.'

# OpenAI format
curl -X POST https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/v1/chat/completions \
  -F "file=@data/luggage_xray_yolo/test/images/xray_001.jpg" | jq '.'
```

---

## 5. API Documentation

View interactive API docs (try it out in browser):
**https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/docs**

---

## Endpoint Comparison

| Feature | `/v1/detect` | `/v1/chat/completions` |
|---------|-------------|------------------------|
| **Format** | Direct JSON | OpenAI-compatible |
| **Use Case** | Simple integration | Agentic workflows |
| **Input** | Image file | Image file |
| **Output** | DetectionResult | ChatCompletion |
| **Response Structure** | Flat, easy to parse | Nested in `choices[0].message.content` |
| **Integration** | Direct HTTP clients | OpenAI SDK pattern |

---

## Response Fields

### DetectionResult (from `/v1/detect`)

```typescript
{
  detections: Array<{
    class_id: number,
    label: string,
    confidence: number,  // 0.0 to 1.0
    bbox: {
      x_min: number,
      y_min: number,
      x_max: number,
      y_max: number
    }
  }>,
  num_detections: number,
  threats_detected: string[],  // List of threat items found
  is_threat: boolean           // True if any threats detected
}
```

### Key Classes

**Common items:**
- Laptop, Mobile Phone, Laptop Charger, Umbrella, Watch
- Coin, Belt, Sunglasses, Boots, Ring

**Threats (is_threat=true):**
- Knife, Pliers, Scissors, Hammer, Wrench

---

## Troubleshooting

### "Connection refused"
- Wait 1-2 minutes after deployment
- Check application status in CAI UI

### "Model not loaded" (503)
- Check application logs for model download errors
- Verify MODEL_PATH is correct

### "404 Not Found"
- Verify subdomain: `xray-yolo-api`
- Check full URL is correct

### Slow first request
- First request triggers model initialization (~30 seconds for yolov8x.pt)
- Subsequent requests are fast (<100ms with GPU)

---

## Next Steps

1. **Test the API** - Use the test script or curl commands above
2. **View API Docs** - Open `/docs` endpoint in browser
3. **Integrate** - Use in your application or agentic workflow
4. **Monitor** - Check application logs in CAI UI
5. **Train** - Deploy a trained model for better X-ray detection

**Note:** The current deployment uses `yolov8x.pt` (pre-trained on COCO dataset).
For real X-ray threat detection, train on the luggage dataset and redeploy!
