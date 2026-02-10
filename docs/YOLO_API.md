# YOLO API Server Documentation

OpenAI-compatible API server for X-ray baggage threat detection using YOLO.

## Quick Start

### Start Server

```bash
# Using script (recommended)
./scripts/serve_yolo_api.sh \
    --model runs/detect/xray_detection/weights/best.pt \
    --port 8000

# Direct Python
python3 inference/yolo_api_server.py \
    --model runs/detect/xray_detection/weights/best.pt \
    --backend ultralytics \
    --conf-threshold 0.25 \
    --iou-threshold 0.45 \
    --device 0 \
    --host 0.0.0.0 \
    --port 8000
```

**Server will be available at**: `http://localhost:8000`

## API Endpoints

### 1. Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "ultralytics"
}
```

### 2. OpenAI-Compatible Detection (for Agentic Workflows)

```http
POST /v1/chat/completions
Content-Type: multipart/form-data
```

**Request**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -F "file=@/path/to/xray-image.jpg"
```

**Response** (OpenAI chat completion format):
```json
{
  "id": "chatcmpl-yolo-1707504000",
  "object": "chat.completion",
  "created": 1707504000,
  "model": "runs/detect/xray_detection/weights/best.pt",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"items\":[{\"name\":\"Knife\",\"confidence\":0.92,\"location\":\"center\",\"bbox\":[0.45,0.52,0.12,0.08]},{\"name\":\"Scissors\",\"confidence\":0.87,\"location\":\"upper-right\",\"bbox\":[0.78,0.23,0.09,0.11]}],\"total_count\":2,\"has_concealed_items\":false}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### 3. Direct Detection Endpoint

```http
POST /v1/detect
Content-Type: multipart/form-data
```

**Request**:
```bash
curl -X POST http://localhost:8000/v1/detect \
  -F "file=@/path/to/xray-image.jpg"
```

**Response** (structured JSON):
```json
{
  "items": [
    {
      "name": "Knife",
      "confidence": 0.92,
      "location": "center",
      "bbox": [0.45, 0.52, 0.12, 0.08]
    },
    {
      "name": "Scissors",
      "confidence": 0.87,
      "location": "upper-right",
      "bbox": [0.78, 0.23, 0.09, 0.11]
    }
  ],
  "total_count": 2,
  "has_concealed_items": false
}
```

## Response Schema

### Item Detection Object

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Item category (e.g., "Knife", "Gun", "Explosive") |
| `confidence` | float | Detection confidence [0.0, 1.0] |
| `location` | string | Spatial location (e.g., "center", "upper-left") |
| `bbox` | array[4] | Bounding box `[x_center, y_center, width, height]` normalized to [0, 1] |

### Location Strings

- `center` - Center of image
- `upper`, `lower`, `left`, `right` - Edge regions
- `upper-left`, `upper-right`, `lower-left`, `lower-right` - Corners
- `center-left`, `center-right`, `upper-center`, `lower-center` - Mid-points

### Complete Response

| Field | Type | Description |
|-------|------|-------------|
| `items` | array | List of detected items |
| `total_count` | integer | Total number of detections |
| `has_concealed_items` | boolean | Whether items are overlapping/concealed (IOU > 0.3) |

## Configuration

### Backend Options

**Ultralytics (Native PyTorch)** - Recommended for development
```bash
python3 inference/yolo_api_server.py \
    --model best.pt \
    --backend ultralytics
```

**ONNX Runtime** - Recommended for production (10-20% faster)
```bash
python3 inference/yolo_api_server.py \
    --model best.onnx \
    --backend onnx
```

### Threshold Tuning

**Confidence Threshold** (`--conf-threshold`):
- Higher (0.5-0.7): Fewer false positives, more missed threats
- Lower (0.15-0.25): More detections, more false positives
- **Default**: 0.25 (balanced)

**IOU Threshold** (`--iou-threshold`):
- Controls Non-Maximum Suppression (NMS)
- Higher (0.6-0.8): More boxes kept (for dense scenes)
- Lower (0.3-0.4): Fewer boxes (removes overlaps)
- **Default**: 0.45 (standard)

**Example for high-security screening** (minimize missed threats):
```bash
./scripts/serve_yolo_api.sh \
    --conf-threshold 0.15 \
    --iou-threshold 0.3
```

### Device Selection

```bash
# GPU 0
--device 0

# CPU only
--device cpu

# Multi-GPU (load balancing not yet supported)
--device 0
```

## Agentic Workflow Integration

### OpenAI Client (Python)

```python
from openai import OpenAI
import json

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No auth required
)

# Send X-ray image
with open("xray_scan.jpg", "rb") as f:
    response = client.chat.completions.create(
        model="yolo-detection",
        messages=[],  # Not used, but required by OpenAI format
        files={"file": f}
    )

# Parse results
result_json = response.choices[0].message.content
detections = json.loads(result_json)

print(f"Found {detections['total_count']} threats:")
for item in detections['items']:
    print(f"  - {item['name']} (conf: {item['confidence']:.2f}, loc: {item['location']})")
```

### LangChain Integration

```python
from langchain.tools import Tool
import requests
import json

def detect_xray_threats(image_path: str) -> str:
    """Detect threats in X-ray baggage scan."""
    with open(image_path, 'rb') as f:
        response = requests.post(
            "http://localhost:8000/v1/detect",
            files={"file": f}
        )
    result = response.json()
    
    # Format for agent
    if result['total_count'] == 0:
        return "No threats detected in scan."
    
    threats = [f"{item['name']} (confidence: {item['confidence']:.0%}, location: {item['location']})" 
               for item in result['items']]
    
    warning = " WARNING: Items may be concealed." if result['has_concealed_items'] else ""
    
    return f"Detected {result['total_count']} threat(s): {', '.join(threats)}.{warning}"

# Create LangChain tool
xray_detection_tool = Tool(
    name="XrayThreatDetection",
    func=detect_xray_threats,
    description="Detect prohibited items in X-ray baggage scans. Input: path to X-ray image file. Returns: list of detected threats with confidence scores and locations."
)
```

### AutoGen Integration

```python
from autogen import ConversableAgent
import requests
import json

def xray_detection_function(image_path: str) -> dict:
    """Detect threats in X-ray scan for AutoGen agent."""
    with open(image_path, 'rb') as f:
        response = requests.post(
            "http://localhost:8000/v1/detect",
            files={"file": f}
        )
    return response.json()

# Register with AutoGen agent
agent = ConversableAgent(
    name="SecurityAgent",
    system_message="You are a security screening assistant.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": "..."}]},
    function_map={"detect_xray_threats": xray_detection_function}
)
```

## Performance Optimization

### Batch Processing

For multiple images, use concurrent requests:

```python
import asyncio
import aiohttp

async def detect_batch(image_paths: list[str]):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for path in image_paths:
            with open(path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=path)
                tasks.append(session.post("http://localhost:8000/v1/detect", data=data))
        
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]

# Process 100 images
results = asyncio.run(detect_batch(image_list))
```

### Production Deployment

**1. Use ONNX Backend**:
```bash
# Export model
python3 -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx')"

# Serve ONNX
./scripts/serve_yolo_api.sh --model best.onnx --backend onnx
```

**2. Use Docker**:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY inference/ inference/
COPY runs/detect/xray_detection/weights/best.pt model.pt
CMD ["python3", "inference/yolo_api_server.py", "--model", "model.pt", "--host", "0.0.0.0"]
```

**3. Use Production WSGI Server**:
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 inference.yolo_api_server:app
```

**4. Add Authentication**:
```python
from fastapi import Depends, HTTPException, Header

async def verify_token(authorization: str = Header(None)):
    if not authorization or authorization != "Bearer YOUR_SECRET_TOKEN":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions(...):
    ...
```

## Monitoring

### Logs

```bash
# Run with detailed logs
python3 inference/yolo_api_server.py --model best.pt 2>&1 | tee api.log
```

### Metrics

Add Prometheus metrics:
```python
from prometheus_client import Counter, Histogram, start_http_server

detection_counter = Counter('xray_detections_total', 'Total detections')
inference_latency = Histogram('xray_inference_seconds', 'Inference latency')

@app.post("/v1/detect")
async def detect(file: UploadFile):
    with inference_latency.time():
        result = detection_engine.predict(image)
    detection_counter.inc(result.total_count)
    return result

# Start metrics server
start_http_server(9090)
```

## Troubleshooting

### Model Not Loading

```bash
# Check model file exists
ls -lh runs/detect/xray_detection/weights/best.pt

# Verify PyTorch can load it
python3 -c "from ultralytics import YOLO; YOLO('best.pt')"
```

### Slow Inference

- Use ONNX backend: `--backend onnx`
- Use GPU: `--device 0`
- Reduce image size during training: `--imgsz 512`
- Use smaller model: `yolov8n` instead of `yolov8m`

### Poor Detection Quality

- Lower confidence threshold: `--conf-threshold 0.15`
- Check image quality (resolution, contrast)
- Retrain with more data or better augmentations
- Use larger model: `yolov8s` or `yolov8m`

## API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Next Steps

- [YOLO Training Guide](YOLO_TRAINING.md) - Train your own model
- [Complete Workflow](COMPLETE_WORKFLOW.md) - End-to-end pipeline
- [Deployment Guide](../cai_integration/README.md) - Deploy to CAI

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics YOLO Inference](https://docs.ultralytics.com/modes/predict/)
- [ONNX Runtime](https://onnxruntime.ai/)
