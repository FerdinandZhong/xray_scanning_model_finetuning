#!/bin/bash
#
# Deploy YOLO Fine-Tuning to Cloudera AI Workspace
#
# Usage:
#   ./scripts/deploy_to_cai.sh <dataset>
#
# Where <dataset> is one of:
#   - cargoxray: Quick baseline (30 min, 659 images)
#   - stcray: Production model (4 hours, 46k images)
#

set -e

DATASET=${1:-cargoxray}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "Deploy YOLO Fine-Tuning to CAI"
echo "======================================================================"
echo ""
echo "Dataset: ${DATASET}"
echo ""

# Validate dataset choice
if [[ "$DATASET" != "cargoxray" && "$DATASET" != "stcray" ]]; then
    echo -e "${RED}Error: Invalid dataset '${DATASET}'${NC}"
    echo ""
    echo "Valid options:"
    echo "  - cargoxray: Quick baseline (30 min)"
    echo "  - stcray: Production model (4 hours)"
    echo ""
    echo "Usage: ./scripts/deploy_to_cai.sh <dataset>"
    exit 1
fi

# Check if on correct branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: ${BRANCH}"
echo ""

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    git status -s
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Commit and push if needed
echo "Ensuring latest code is pushed..."
git add .
git commit -m "chore: Prepare for CAI deployment (${DATASET})" || echo "Nothing to commit"
git push origin "${BRANCH}"

echo ""
echo "======================================================================"
echo "Triggering GitHub Actions Workflow"
echo "======================================================================"
echo ""

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) not found${NC}"
    echo ""
    echo "Install with:"
    echo "  brew install gh"
    echo "  # or"
    echo "  sudo apt install gh"
    echo ""
    echo "Then authenticate:"
    echo "  gh auth login"
    exit 1
fi

# Trigger workflow
echo "Triggering deploy-to-cai.yml workflow..."
echo ""

gh workflow run deploy-to-cai.yml \
    --field model_type=yolo \
    --field dataset="${DATASET}"

if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✅ Workflow triggered successfully!${NC}"
    echo ""
    echo "Monitor progress:"
    echo "  gh run watch"
    echo ""
    echo "Or view in browser:"
    echo "  gh workflow view deploy-to-cai.yml --web"
    echo ""
    
    if [[ "$DATASET" == "cargoxray" ]]; then
        echo "Expected completion: ~1 hour"
        echo "  - Setup: 30 min"
        echo "  - Upload: 5 min"
        echo "  - Training: 30 min"
    else
        echo "Expected completion: ~5 hours"
        echo "  - Setup: 30 min"
        echo "  - Download: 30 min"
        echo "  - Training: 4 hours"
    fi
    echo ""
else
    echo -e "${RED}❌ Failed to trigger workflow${NC}"
    echo ""
    echo "Manual trigger:"
    echo "  1. Go to: https://github.com/YOUR_USER/YOUR_REPO/actions"
    echo "  2. Select 'Deploy to CAI' workflow"
    echo "  3. Click 'Run workflow'"
    echo "  4. Set:"
    echo "     - model_type: yolo"
    echo "     - dataset: ${DATASET}"
    exit 1
fi
