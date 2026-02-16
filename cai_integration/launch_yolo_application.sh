#!/bin/bash
# CAI Application launcher for YOLO Detection API
# This script is executed by CAI when starting the application

set -e

echo "========================================"
echo "Starting YOLO X-ray Detection API"
echo "========================================"
echo ""

# Get configuration from environment variables (set by CAI Application)
MODEL_PATH="${MODEL_PATH:-runs/detect/xray_detection/weights/best.pt}"
BACKEND="${BACKEND:-ultralytics}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.45}"
DEVICE="${DEVICE:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"  # CAI Applications use port 8080

echo "Configuration:"
echo "  Model Path:       $MODEL_PATH"
echo "  Backend:          $BACKEND"
echo "  Conf Threshold:   $CONF_THRESHOLD"
echo "  IOU Threshold:    $IOU_THRESHOLD"
echo "  Device:           $DEVICE"
echo "  Host:             $HOST"
echo "  Port:             $PORT"
echo ""

# Check if model exists (or is a pre-trained model name)
PRETRAINED_MODELS="yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt yolov11n.pt yolov11s.pt yolov11m.pt"

if [ -f "$MODEL_PATH" ]; then
    echo "✓ Using local model: $MODEL_PATH"
elif echo "$PRETRAINED_MODELS" | grep -q "$MODEL_PATH"; then
    echo "✓ Using pre-trained model: $MODEL_PATH"
    echo "  (Ultralytics will download automatically if needed)"
else
    echo "ERROR: Model not found at $MODEL_PATH"
    echo ""
    echo "Available local models:"
    find runs/detect -name "best.pt" -type f 2>/dev/null || echo "  No trained models found in runs/detect"
    echo ""
    echo "Available pre-trained models:"
    echo "  yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt"
    echo "  yolov11n.pt, yolov11s.pt, yolov11m.pt"
    exit 1
fi

# Check if venv exists and activate
if [ -d ".venv_yolo" ]; then
    echo "Activating Python virtual environment..."
    source .venv_yolo/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating Python virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found, using system Python"
fi

# Verify Python and dependencies
echo "Checking dependencies..."
python3 -c "import fastapi; import uvicorn; from ultralytics import YOLO; print('✓ All dependencies available')" || {
    echo "ERROR: Missing dependencies"
    echo "Installing required packages..."
    pip install --quiet fastapi uvicorn pillow ultralytics
}

# Pre-download model if it's a pre-trained model name
if echo "$PRETRAINED_MODELS" | grep -q "$MODEL_PATH"; then
    echo ""
    echo "Downloading pre-trained model (if not cached)..."
    python3 -c "from ultralytics import YOLO; YOLO('$MODEL_PATH'); print('✓ Model ready')"
    echo "  Cached at: ~/.cache/ultralytics/"
fi

echo ""
echo "========================================"
echo "Starting API Server"
echo "========================================"
echo ""
echo "Endpoints will be available at:"
echo "  Health Check:     http://$HOST:$PORT/health"
echo "  API Docs:         http://$HOST:$PORT/docs"
echo "  OpenAI API:       http://$HOST:$PORT/v1/chat/completions"
echo "  Direct Detection: http://$HOST:$PORT/v1/detect"
echo ""

# Start the server
exec python3 inference/yolo_api_server.py \
    --model "$MODEL_PATH" \
    --backend "$BACKEND" \
    --conf-threshold "$CONF_THRESHOLD" \
    --iou-threshold "$IOU_THRESHOLD" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT"
