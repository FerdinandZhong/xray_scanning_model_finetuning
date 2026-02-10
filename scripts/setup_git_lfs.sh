#!/bin/bash
#
# Setup Git LFS and push CargoXray dataset to repository
#
# This script:
# 1. Installs/verifies Git LFS
# 2. Downloads CargoXray dataset
# 3. Converts to YOLO format
# 4. Commits and pushes to GitHub with LFS
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================"
echo "Git LFS Setup for CargoXray Dataset"
echo -e "======================================================================${NC}"
echo ""

# Step 1: Check/Install Git LFS
echo -e "${BLUE}Step 1: Checking Git LFS...${NC}"
if ! command -v git-lfs &> /dev/null; then
    echo -e "${YELLOW}Git LFS not found. Installing...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Installing via Homebrew..."
        brew install git-lfs
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Installing via apt-get..."
        sudo apt-get update
        sudo apt-get install -y git-lfs
    else
        echo -e "${RED}Error: Unsupported OS${NC}"
        echo "Please install Git LFS manually: https://git-lfs.github.com/"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Git LFS already installed${NC}"
fi

# Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install
echo ""

# Step 2: Check if dataset already exists
echo -e "${BLUE}Step 2: Checking CargoXray dataset...${NC}"
if [ -d "data/cargoxray/test" ] && [ -d "data/cargoxray_yolo" ]; then
    echo -e "${GREEN}✓ CargoXray dataset already exists${NC}"
    
    read -p "Dataset found. Re-download? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download. Using existing dataset."
    else
        rm -rf data/cargoxray data/cargoxray_yolo
        DOWNLOAD=true
    fi
else
    DOWNLOAD=true
fi

if [ "$DOWNLOAD" = true ]; then
    echo -e "${BLUE}Downloading CargoXray from Roboflow...${NC}"
    mkdir -p data/cargoxray
    cd data/cargoxray
    
    echo "Downloading (83MB)..."
    curl -L "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6" > roboflow.zip
    
    echo "Extracting..."
    unzip -q roboflow.zip
    rm roboflow.zip
    
    cd ../..
    echo -e "${GREEN}✓ Downloaded successfully${NC}"
    
    # Convert to YOLO format
    echo -e "${BLUE}Converting to YOLO format...${NC}"
    python scripts/convert_cargoxray_to_yolo.py \
        --input-dir data/cargoxray \
        --output-dir data/cargoxray_yolo
    
    echo -e "${GREEN}✓ Converted successfully${NC}"
fi

echo ""

# Step 3: Check dataset size
echo -e "${BLUE}Step 3: Checking dataset size...${NC}"
CARGO_SIZE=$(du -sh data/cargoxray | cut -f1)
YOLO_SIZE=$(du -sh data/cargoxray_yolo | cut -f1)

echo "Dataset sizes:"
echo "  data/cargoxray: ${CARGO_SIZE}"
echo "  data/cargoxray_yolo: ${YOLO_SIZE}"
echo ""

echo -e "${YELLOW}Note: GitHub LFS free tier = 1GB storage + 1GB bandwidth/month${NC}"
echo "Expected total: ~300MB (fits within free tier)"
echo ""

# Step 4: Check .gitattributes
echo -e "${BLUE}Step 4: Verifying .gitattributes...${NC}"
if [ ! -f ".gitattributes" ]; then
    echo -e "${RED}Error: .gitattributes not found${NC}"
    echo "This file should exist in the repo root."
    exit 1
fi

echo "Checking LFS patterns..."
if grep -q "data/.*\.jpg.*filter=lfs" .gitattributes; then
    echo -e "${GREEN}✓ .gitattributes configured correctly${NC}"
else
    echo -e "${YELLOW}Warning: .gitattributes may need updating${NC}"
fi
echo ""

# Step 5: Add files to Git
echo -e "${BLUE}Step 5: Adding files to Git...${NC}"

# Check if files are already committed
if git ls-files --error-unmatch data/cargoxray/ &> /dev/null; then
    echo -e "${YELLOW}Files already tracked by Git${NC}"
    read -p "Re-add anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping git add."
        SKIP_COMMIT=true
    fi
fi

if [ "$SKIP_COMMIT" != true ]; then
    echo "Adding files..."
    git add .gitattributes
    git add data/cargoxray/
    git add data/cargoxray_yolo/
    
    # Show what will be tracked by LFS
    echo ""
    echo "Files tracked by LFS:"
    git lfs ls-files | head -10
    if [ $(git lfs ls-files | wc -l) -gt 10 ]; then
        echo "... and $(( $(git lfs ls-files | wc -l) - 10 )) more"
    fi
    echo ""
    
    # Step 6: Commit
    echo -e "${BLUE}Step 6: Committing dataset...${NC}"
    git commit -m "feat: Add CargoXray dataset via Git LFS

- 659 X-ray images (462 train, 132 valid, 65 test)
- 16 cargo categories (textiles, tools, auto parts, etc.)
- Pre-converted to YOLO format
- Total size: ~300MB
- Tracked via Git LFS for automatic deployment"
    
    echo -e "${GREEN}✓ Committed successfully${NC}"
    echo ""
    
    # Step 7: Push
    echo -e "${BLUE}Step 7: Pushing to GitHub...${NC}"
    echo -e "${YELLOW}This will upload ~300MB to GitHub LFS${NC}"
    echo "Estimated time: 2-5 minutes (depending on connection)"
    echo ""
    
    read -p "Continue with push? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Pushing..."
        git push origin $(git rev-parse --abbrev-ref HEAD)
        
        echo ""
        echo -e "${GREEN}✓ Pushed successfully!${NC}"
    else
        echo "Skipped push. Run manually when ready:"
        echo "  git push origin $(git rev-parse --abbrev-ref HEAD)"
    fi
else
    echo "Skipped commit (files already tracked)"
fi

echo ""
echo -e "${BLUE}======================================================================"
echo "Setup Complete!"
echo -e "======================================================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Verify LFS storage usage:"
echo "   gh api /repos/OWNER/REPO/lfs/usage"
echo ""
echo "2. Deploy to CAI via GitHub Actions:"
echo "   gh workflow run deploy-to-cai.yml \\"
echo "     --field model_type=yolo \\"
echo "     --field dataset=cargoxray \\"
echo "     --field trigger_jobs=true"
echo ""
echo "3. Or trigger manually in GitHub UI:"
echo "   Actions → Deploy X-ray Detection to CAI → Run workflow"
echo ""
echo -e "${GREEN}Dataset is now ready for automated deployment!${NC}"
echo ""
