#!/bin/bash
# Quick script to process both train and test sets from extracted STCray data

set -e

echo "============================================================"
echo "STCray Dataset Processing"
echo "============================================================"
echo ""

# Check if Python is available
if [ -x "venv_vqa/bin/python" ]; then
    PYTHON_CMD="venv_vqa/bin/python"
    echo "✓ Using venv_vqa Python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✓ Using system Python3"
else
    PYTHON_CMD="python"
    echo "✓ Using system Python"
fi

echo ""

# Process train set
echo "Processing training set..."
$PYTHON_CMD data/process_stcray_extracted.py \
  --input-dir data/stcray_raw/STCray_TrainSet \
  --output-dir data/stcray_processed/train

echo ""
echo "Processing test set..."
$PYTHON_CMD data/process_stcray_extracted.py \
  --input-dir data/stcray_raw/STCray_TestSet \
  --output-dir data/stcray_processed/test

echo ""
echo "============================================================"
echo "✓ Processing Complete!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  - data/stcray_processed/train/annotations.json"
echo "  - data/stcray_processed/test/annotations.json"
echo ""
echo "Next step: Generate VQA dataset"
echo "  export API_KEY='your-api-key'"
echo "  ./scripts/generate_vqa_gemini.sh"
echo ""
