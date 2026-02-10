#!/bin/bash
# Start YOLO API server for X-ray baggage detection

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}YOLO X-ray Detection API Server${NC}"
echo -e "${GREEN}=================================${NC}"

# Default parameters
MODEL="${MODEL:-runs/detect/xray_detection/weights/best.pt}"
BACKEND="${BACKEND:-ultralytics}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.45}"
DEVICE="${DEVICE:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --conf-threshold)
            CONF_THRESHOLD="$2"
            shift 2
            ;;
        --iou-threshold)
            IOU_THRESHOLD="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH         Path to model (.pt or .onnx)"
            echo "  --backend NAME       Backend (ultralytics or onnx, default: ultralytics)"
            echo "  --conf-threshold N   Confidence threshold (default: 0.25)"
            echo "  --iou-threshold N    IOU threshold for NMS (default: 0.45)"
            echo "  --device DEVICE      Device (0, cpu, etc.)"
            echo "  --host HOST          Host to bind (default: 0.0.0.0)"
            echo "  --port PORT          Port to bind (default: 8000)"
            echo "  --help               Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  Same as options above (e.g., MODEL=path/to/best.pt)"
            echo ""
            echo "Endpoints:"
            echo "  GET  /health                 Health check"
            echo "  POST /v1/chat/completions    OpenAI-compatible detection"
            echo "  POST /v1/detect              Direct detection"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo -e "${RED}Error: Model not found: $MODEL${NC}"
    echo -e "${YELLOW}Train a model first:${NC}"
    echo "  ./scripts/train_yolo_local.sh"
    exit 1
fi

# Validate backend
if [[ "$BACKEND" != "ultralytics" && "$BACKEND" != "onnx" ]]; then
    echo -e "${RED}Error: Invalid backend: $BACKEND${NC}"
    echo "Use 'ultralytics' or 'onnx'"
    exit 1
fi

# Check model extension matches backend
MODEL_EXT="${MODEL##*.}"
if [[ "$BACKEND" == "ultralytics" && "$MODEL_EXT" != "pt" ]]; then
    echo -e "${YELLOW}Warning: Using ultralytics backend with .$MODEL_EXT model${NC}"
    echo "Expected .pt model for ultralytics backend"
fi
if [[ "$BACKEND" == "onnx" && "$MODEL_EXT" != "onnx" ]]; then
    echo -e "${YELLOW}Warning: Using onnx backend with .$MODEL_EXT model${NC}"
    echo "Expected .onnx model for onnx backend"
fi

# Print configuration
echo ""
echo -e "${YELLOW}Server Configuration:${NC}"
echo "  Model:           $MODEL"
echo "  Backend:         $BACKEND"
echo "  Conf Threshold:  $CONF_THRESHOLD"
echo "  IOU Threshold:   $IOU_THRESHOLD"
echo "  Device:          $DEVICE"
echo "  Host:            $HOST"
echo "  Port:            $PORT"
echo ""
echo -e "${YELLOW}API Endpoints:${NC}"
echo "  Health Check:     http://$HOST:$PORT/health"
echo "  OpenAI API:       http://$HOST:$PORT/v1/chat/completions"
echo "  Direct Detection: http://$HOST:$PORT/v1/detect"
echo "  API Docs:         http://$HOST:$PORT/docs"
echo ""

# Start server
echo -e "${GREEN}Starting API server...${NC}"
echo ""

python3 inference/yolo_api_server.py \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --conf-threshold "$CONF_THRESHOLD" \
    --iou-threshold "$IOU_THRESHOLD" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT"
