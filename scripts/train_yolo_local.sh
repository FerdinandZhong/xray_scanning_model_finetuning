#!/bin/bash
# Train YOLO model locally for X-ray baggage detection

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}YOLO X-ray Detection Training${NC}"
echo -e "${GREEN}=================================${NC}"

# Default parameters
DATA_YAML="${DATA_YAML:-data/yolo_dataset/data.yaml}"
MODEL="${MODEL:-yolov8n.pt}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
PROJECT="${PROJECT:-runs/detect}"
NAME="${NAME:-xray_detection}"
EXPORT_ONNX="${EXPORT_ONNX:-false}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_YAML="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --export-onnx)
            EXPORT_ONNX="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data PATH          Path to data.yaml (default: data/yolo_dataset/data.yaml)"
            echo "  --model NAME         Model name (yolov8n, yolov8s, yolov8m, yolov11n, etc.)"
            echo "  --epochs N           Number of epochs (default: 100)"
            echo "  --batch N            Batch size (default: 16)"
            echo "  --imgsz N            Image size (default: 640)"
            echo "  --device DEVICE      Device (0, cpu, or 0,1,2,3 for multi-GPU)"
            echo "  --project PATH       Project directory (default: runs/detect)"
            echo "  --name NAME          Run name (default: xray_detection)"
            echo "  --export-onnx        Export best model to ONNX after training"
            echo "  --help               Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  Same as options above (e.g., MODEL=yolov8s)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if data.yaml exists
if [ ! -f "$DATA_YAML" ]; then
    echo -e "${RED}Error: Data file not found: $DATA_YAML${NC}"
    echo -e "${YELLOW}Run the data conversion first:${NC}"
    echo "  python3 data/convert_to_yolo_format.py --help"
    exit 1
fi

# Print configuration
echo ""
echo -e "${YELLOW}Training Configuration:${NC}"
echo "  Data:    $DATA_YAML"
echo "  Model:   $MODEL"
echo "  Epochs:  $EPOCHS"
echo "  Batch:   $BATCH"
echo "  ImgSize: $IMGSZ"
echo "  Device:  $DEVICE"
echo "  Project: $PROJECT"
echo "  Name:    $NAME"
echo "  Export:  $EXPORT_ONNX"
echo ""

# Confirm before starting
read -p "Start training? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Build command
CMD="python3 training/train_yolo.py \
    --data $DATA_YAML \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch $BATCH \
    --imgsz $IMGSZ \
    --device $DEVICE \
    --project $PROJECT \
    --name $NAME"

if [ "$EXPORT_ONNX" = "true" ]; then
    CMD="$CMD --export-onnx"
fi

# Run training
echo -e "${GREEN}Starting training...${NC}"
echo ""
eval $CMD

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Test the model:"
    echo "     python3 scripts/test_yolo_inference.py --model $PROJECT/$NAME/weights/best.pt"
    echo ""
    echo "  2. Start API server:"
    echo "     ./scripts/serve_yolo_api.sh --model $PROJECT/$NAME/weights/best.pt"
    echo ""
    echo "  3. View training results:"
    echo "     tensorboard --logdir $PROJECT/$NAME"
else
    echo ""
    echo -e "${RED}✗ Training failed!${NC}"
    exit 1
fi
