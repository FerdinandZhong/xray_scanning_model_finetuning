#!/bin/bash
# Download STCray dataset RAR files from HuggingFace
# Files will be tracked with Git LFS

set -e

echo "============================================================"
echo "STCray Dataset RAR Download"
echo "============================================================"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ Error: huggingface-cli not found"
    echo "   Install with: pip install huggingface_hub[cli]"
    exit 1
fi

# Check if user is logged in
echo "Checking HuggingFace authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Error: Not logged in to HuggingFace"
    echo "   Please run: huggingface-cli login"
    exit 1
fi

echo "✓ Authenticated as: $(huggingface-cli whoami | head -1)"
echo ""

# Create output directory
OUTPUT_DIR="data/stcray_raw"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Download options
DOWNLOAD_TRAIN=true
DOWNLOAD_TEST=true
DOWNLOAD_AUGMENTED=false  # 21.5 GB, set to true if needed

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train)
            DOWNLOAD_TRAIN=false
            shift
            ;;
        --skip-test)
            DOWNLOAD_TEST=false
            shift
            ;;
        --with-augmented)
            DOWNLOAD_AUGMENTED=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-train] [--skip-test] [--with-augmented] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

# Download function
download_file() {
    local filename=$1
    local size=$2
    
    echo "============================================================"
    echo "Downloading: $filename ($size)"
    echo "============================================================"
    
    if [ -f "$OUTPUT_DIR/$filename" ]; then
        echo "⚠ File already exists: $OUTPUT_DIR/$filename"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipped: $filename"
            echo ""
            return
        fi
    fi
    
    huggingface-cli download Naoufel555/STCray-Dataset \
        "$filename" \
        --repo-type dataset \
        --local-dir "$OUTPUT_DIR" \
        --local-dir-use-symlinks False
    
    echo "✓ Downloaded: $filename"
    echo ""
}

# Download files
if [ "$DOWNLOAD_TRAIN" = true ]; then
    download_file "STCray_TrainSet.rar" "1.09 GB"
fi

if [ "$DOWNLOAD_TEST" = true ]; then
    download_file "STCray_TestSet.rar" "988 MB"
fi

if [ "$DOWNLOAD_AUGMENTED" = true ]; then
    download_file "STCray_Augmented.rar" "21.5 GB"
fi

echo "============================================================"
echo "✓ Download Complete"
echo "============================================================"
echo ""
echo "Files downloaded to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Install unrar (if not already installed):"
echo "   macOS:   brew install unrar"
echo "   Ubuntu:  sudo apt-get install unrar"
echo ""
echo "2. Extract the files:"
echo "   cd $OUTPUT_DIR"
if [ "$DOWNLOAD_TRAIN" = true ]; then
    echo "   unrar x STCray_TrainSet.rar"
fi
if [ "$DOWNLOAD_TEST" = true ]; then
    echo "   unrar x STCray_TestSet.rar"
fi
if [ "$DOWNLOAD_AUGMENTED" = true ]; then
    echo "   unrar x STCray_Augmented.rar"
fi
echo ""
echo "3. (Optional) Process into annotations format:"
echo "   python data/process_stcray_extracted.py --input-dir $OUTPUT_DIR"
echo ""
