#!/bin/bash
# Set up Git LFS for STCray dataset files

set -e

echo "============================================================"
echo "Git LFS Setup for STCray Dataset"
echo "============================================================"
echo ""

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Error: git-lfs not found"
    echo ""
    echo "Install Git LFS:"
    echo "  macOS:   brew install git-lfs"
    echo "  Ubuntu:  sudo apt-get install git-lfs"
    echo "  Windows: Download from https://git-lfs.github.com/"
    echo ""
    exit 1
fi

echo "✓ Git LFS found: $(git-lfs version)"
echo ""

# Initialize Git LFS in the repository
echo "Initializing Git LFS..."
git lfs install

echo "✓ Git LFS initialized"
echo ""

# Track RAR files with LFS
echo "Setting up LFS tracking for RAR files..."
git lfs track "*.rar"
git lfs track "data/stcray_raw/*.rar"

echo "✓ LFS tracking configured"
echo ""

# Show what's being tracked
echo "Currently tracked patterns:"
git lfs track
echo ""

# Add .gitattributes to git
if [ -f .gitattributes ]; then
    echo "Adding .gitattributes to git..."
    git add .gitattributes
    echo "✓ .gitattributes staged"
    echo ""
fi

echo "============================================================"
echo "✓ Git LFS Setup Complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Download STCray RAR files:"
echo "   ./scripts/download_stcray_rar.sh"
echo ""
echo "2. Add RAR files to git:"
echo "   git add data/stcray_raw/*.rar"
echo ""
echo "3. Commit:"
echo "   git commit -m \"Add STCray dataset RAR files\""
echo ""
echo "4. Push to remote:"
echo "   git push"
echo ""
echo "Note: RAR files will be stored in Git LFS, not in the main repository"
echo ""
