# Git LFS Setup for Datasets

Guide for using Git LFS to store CargoXray dataset in the repository for easy deployment to CAI.

## Why Git LFS?

**Benefits:**
- ✅ Dataset included in repo (no separate download)
- ✅ Version controlled
- ✅ Automatic deployment to CAI via GitHub Actions
- ✅ No need for external storage

**Limitations:**
- ⚠️ GitHub LFS free tier: 1GB storage, 1GB bandwidth/month
- ⚠️ CargoXray (~150MB) fits, STCray (~20GB) doesn't
- ⚠️ Paid plans available for larger datasets

---

## Initial Setup

### 1. Install Git LFS

```bash
# macOS
brew install git-lfs

# Linux
sudo apt-get install git-lfs

# Windows
# Download from: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install
```

### 2. Verify Installation

```bash
git lfs version
# Output: git-lfs/3.x.x
```

---

## Push CargoXray Dataset

### Step 1: Download & Convert CargoXray

```bash
# Download from Roboflow
cd data/cargoxray
curl -L "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6" > roboflow.zip
unzip roboflow.zip && rm roboflow.zip
cd ../..

# Convert to YOLO format
python scripts/convert_cargoxray_to_yolo.py \
  --input-dir data/cargoxray \
  --output-dir data/cargoxray_yolo
```

### Step 2: Check Dataset Size

```bash
# Check size before committing
du -sh data/cargoxray
du -sh data/cargoxray_yolo

# Expected:
#   data/cargoxray: ~150MB
#   data/cargoxray_yolo: ~150MB
#   Total: ~300MB (within GitHub free tier)
```

### Step 3: Configure Git LFS Tracking

The `.gitattributes` file is already configured, but verify:

```bash
cat .gitattributes

# Should include:
# data/**/*.jpg filter=lfs diff=lfs merge=lfs -text
# data/**/*.json filter=lfs diff=lfs merge=lfs -text
```

### Step 4: Add Dataset to Git

```bash
# Add CargoXray files
git add data/cargoxray/
git add data/cargoxray_yolo/

# Check what will be tracked by LFS
git lfs ls-files

# Should show:
# data/cargoxray/**/*.jpg
# data/cargoxray_yolo/**/*.jpg
# data/cargoxray/**/*.json
```

### Step 5: Commit & Push

```bash
# Commit dataset
git commit -m "feat: Add CargoXray dataset via Git LFS

- 659 X-ray images (train/valid/test splits)
- 16 cargo categories
- Pre-converted to YOLO format
- Total size: ~300MB"

# Push to GitHub (will use LFS for large files)
git push origin main

# Monitor LFS upload progress
# GitHub will show: "Uploading LFS objects: 100% (N/N), X MB"
```

---

## Verify Upload

### Check GitHub LFS Storage

```bash
# View LFS bandwidth and storage usage
gh api /repos/OWNER/REPO/lfs/usage

# Or check in GitHub UI:
# Settings → Billing → Git LFS data
```

### Verify Files in Repository

```bash
# Clone in a new location to test
cd /tmp
git clone https://github.com/OWNER/REPO.git test-clone
cd test-clone

# LFS files should be downloaded automatically
ls -lh data/cargoxray/test/*.jpg

# Verify YOLO format exists
cat data/cargoxray_yolo/data.yaml
```

---

## GitHub Actions Integration

### Workflow Changes

The workflow is already configured to use the dataset from the repo:

```yaml
# .github/workflows/deploy-to-cai.yml
setup-project:
  steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        lfs: true  # ← Downloads LFS files automatically
```

### Deployment Process

```
1. GitHub Actions triggers
   ↓
2. Checkout code (includes LFS files)
   ↓
3. CAI project setup
   ↓
4. Code + Dataset synced to CAI
   ↓
5. Training jobs use dataset directly
```

No separate download/upload step needed!

---

## Update Job Configuration

### For CargoXray (from Git LFS)

Update `cai_integration/jobs_config_yolo.yaml`:

```yaml
jobs:
  setup_environment:
    # ... existing config ...

  # No download_dataset job needed for CargoXray!
  # Dataset is already in repo via Git LFS

  yolo_training:
    name: "Train YOLO on CargoXray"
    script: "training/train_yolo.py"
    parent_job_key: "setup_environment"  # ← Direct parent
    environment:
      DATA_YAML: "data/cargoxray_yolo/data.yaml"  # ← Uses repo data
      MODEL_NAME: "yolov8n.pt"
      EPOCHS: "100"
```

### For STCray (Download from HuggingFace)

STCray is too large for Git LFS, keep existing download job:

```yaml
jobs:
  download_dataset:
    name: "Download STCray Dataset"
    script: "cai_integration/download_dataset.py"
    # Downloads from HuggingFace
```

---

## Managing LFS Storage

### Check Storage Usage

```bash
# Local LFS cache size
du -sh .git/lfs

# GitHub LFS quota
gh api /repos/OWNER/REPO/lfs/usage
```

### Clean Up Local LFS Cache

```bash
# Remove unreferenced LFS files
git lfs prune

# Clean old LFS files
git lfs prune --verify-remote
```

### Upgrade GitHub LFS Quota

If you need more storage/bandwidth:

```
GitHub Settings → Billing → Git LFS Data
- Storage: $0.07/GB/month
- Bandwidth: $0.07/GB
```

Example costs:
- CargoXray (300MB): Free tier sufficient
- STCray (20GB): $1.40/month + bandwidth

---

## Alternative: Selective LFS

Only track converted YOLO dataset (smaller):

```bash
# .gitattributes - only track YOLO format
data/cargoxray_yolo/**/*.jpg filter=lfs diff=lfs merge=lfs -text
data/cargoxray_yolo/**/*.txt filter=lfs diff=lfs merge=lfs -text

# .gitignore - exclude raw COCO format
data/cargoxray/
```

This saves ~150MB (only store YOLO format, not raw).

---

## Troubleshooting

### Issue 1: LFS Files Not Uploaded

```
Error: LFS files not pushed
```

**Fix:**
```bash
# Re-push LFS files explicitly
git lfs push origin main --all
```

### Issue 2: Checkout Doesn't Download LFS

```
Error: Files are LFS pointers, not actual files
```

**Fix:**
```bash
# Pull LFS files
git lfs pull

# Or clone with LFS
git clone https://github.com/OWNER/REPO.git
cd REPO
git lfs pull
```

### Issue 3: Exceeded LFS Bandwidth

```
Error: Bandwidth limit exceeded
```

**Fix:**
- Wait until next billing cycle (monthly reset)
- Upgrade to paid LFS plan
- Use external storage (S3, HuggingFace) for large files

### Issue 4: Files Too Large

```
Error: File size exceeds GitHub's limit
```

**Fix:**
- GitHub LFS max file size: 5GB
- For larger files, use external storage
- Split dataset into multiple files

---

## Best Practices

### 1. Use LFS for Binary Files Only

```
✅ Images (*.jpg, *.png)
✅ Model weights (*.pt, *.onnx)
✅ Compressed archives (*.zip, *.tar.gz)
❌ Text files (*.txt, *.py, *.yaml)
❌ Small config files
```

### 2. Keep Dataset Size Reasonable

```
✅ CargoXray: ~300MB (perfect for LFS)
⚠️ STCray: ~20GB (too large, use external storage)
```

### 3. Use `.gitattributes` Patterns

```bash
# Track entire directories
data/cargoxray/**/*.jpg filter=lfs diff=lfs merge=lfs -text

# Exclude specific paths
!data/cargoxray/README.txt
```

### 4. Document LFS Usage

Add to README:
```markdown
## Dataset Setup

This repo uses Git LFS for datasets. After cloning:

\`\`\`bash
git lfs pull
\`\`\`
```

---

## Complete Setup Commands

```bash
# 1. Install Git LFS
brew install git-lfs  # or: apt-get install git-lfs
git lfs install

# 2. Download & convert CargoXray
cd data/cargoxray
curl -L "https://app.roboflow.com/ds/BbQux1Jbmr?key=CmUGXQ0DU6" > roboflow.zip
unzip roboflow.zip && rm roboflow.zip
cd ../..
python scripts/convert_cargoxray_to_yolo.py

# 3. Add to Git with LFS
git add .gitattributes
git add data/cargoxray/
git add data/cargoxray_yolo/
git commit -m "feat: Add CargoXray dataset via Git LFS"

# 4. Push to GitHub
git push origin main

# 5. Verify
gh api /repos/OWNER/REPO/lfs/usage

# 6. Deploy via GitHub Actions
gh workflow run deploy-to-cai.yml \
  --field model_type=yolo \
  --field dataset=cargoxray \
  --field trigger_jobs=true
```

---

## Summary

**Benefits of Git LFS:**
- ✅ Dataset versioned with code
- ✅ Automatic deployment to CAI
- ✅ No separate download/upload steps
- ✅ Easy team collaboration

**Costs:**
- Free: 1GB storage + 1GB bandwidth/month
- Paid: $5/month for 50GB storage + 50GB bandwidth

**Recommended:**
- Use LFS for CargoXray (~300MB)
- Use HuggingFace for STCray (~20GB)
- Commit converted YOLO format only (smaller)

---

## Resources

- **Git LFS Docs**: https://git-lfs.github.com/
- **GitHub LFS Pricing**: https://docs.github.com/en/billing/managing-billing-for-git-large-file-storage
- **GitHub Actions Guide**: [docs/GITHUB_ACTIONS_DEPLOYMENT.md](GITHUB_ACTIONS_DEPLOYMENT.md)

---

**Questions?** Check the main [README.md](../README.md) or open an issue!
