# STCray Dataset Download Guide

This guide covers downloading the STCray dataset RAR files and setting up Git LFS tracking.

## Prerequisites

1. **HuggingFace Account with Dataset Access**
   - Create account at https://huggingface.co
   - Go to https://huggingface.co/datasets/Naoufel555/STCray-Dataset
   - Click "Agree and access repository" to get access
   - Wait for approval (usually instant)

2. **Install Required Tools**

```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Login to HuggingFace
huggingface-cli login

# Install Git LFS
# macOS:
brew install git-lfs

# Ubuntu/Debian:
sudo apt-get install git-lfs

# Verify installation
git-lfs version
```

3. **Install unrar (for extraction)**

```bash
# macOS:
brew install unrar

# Ubuntu/Debian:
sudo apt-get install unrar
```

---

## Step 1: Set Up Git LFS

Run the Git LFS setup script:

```bash
./scripts/setup_git_lfs.sh
```

This will:
- Initialize Git LFS in your repository
- Configure LFS tracking for RAR files
- Add `.gitattributes` to git

---

## Step 2: Download STCray RAR Files

### Option A: Download Train + Test (Recommended for most use cases)

```bash
./scripts/download_stcray_rar.sh
```

This downloads:
- `STCray_TrainSet.rar` (1.09 GB) - 30,044 training images
- `STCray_TestSet.rar` (988 MB) - 16,598 test images

**Total download:** ~2 GB

### Option B: Download Only Test Set (for quick testing)

```bash
./scripts/download_stcray_rar.sh --skip-train
```

**Total download:** ~988 MB

### Option C: Download Everything Including Augmented

```bash
./scripts/download_stcray_rar.sh --with-augmented
```

This downloads all files including:
- `STCray_Augmented.rar` (21.5 GB) - Augmented training data

**Total download:** ~23.5 GB ⚠️

### Custom Output Directory

```bash
./scripts/download_stcray_rar.sh --output-dir /custom/path
```

---

## Step 3: Extract RAR Files

After download completes:

```bash
cd data/stcray_raw

# Extract train set
unrar x STCray_TrainSet.rar

# Extract test set
unrar x STCray_TestSet.rar

# (Optional) Extract augmented set
# unrar x STCray_Augmented.rar
```

**Expected structure after extraction:**

```
data/stcray_raw/
├── STCray_TrainSet.rar          # Original RAR (tracked by Git LFS)
├── STCray_TestSet.rar           # Original RAR (tracked by Git LFS)
├── STCray_TrainSet/             # Extracted (not tracked by git)
│   ├── images/
│   └── annotations.json
└── STCray_TestSet/              # Extracted (not tracked by git)
    ├── images/
    └── annotations.json
```

---

## Step 4: Add RAR Files to Git with LFS

```bash
# Stage the RAR files (will use LFS automatically)
git add data/stcray_raw/*.rar

# Commit
git commit -m "Add STCray dataset RAR files"

# Push to remote (LFS files go to LFS storage)
git push
```

**Important:**
- ✅ RAR files are tracked by Git LFS (not in main repo)
- ✅ Extracted files are ignored by git (`.gitignore`)
- ✅ Only RAR archives are version controlled

---

## Step 5: Verify Download

Check that everything is in place:

```bash
# Check RAR files
ls -lh data/stcray_raw/*.rar

# Check extracted directories
ls -d data/stcray_raw/STCray_*/

# Check image counts
find data/stcray_raw/STCray_TrainSet/images -name "*.jpg" | wc -l  # Should be ~30,044
find data/stcray_raw/STCray_TestSet/images -name "*.jpg" | wc -l   # Should be ~16,598
```

---

## Disk Space Requirements

| Files | Download Size | Extracted Size | Total |
|-------|--------------|----------------|-------|
| **Train + Test** | ~2 GB | ~5 GB | ~7 GB |
| **With Augmented** | ~23.5 GB | ~50 GB | ~73.5 GB |

**Recommendation:** Start with Train + Test only unless you specifically need augmented data.

---

## Troubleshooting

### Issue: "Access to this resource is forbidden"

**Solution:** You haven't accepted the dataset terms yet.
1. Go to https://huggingface.co/datasets/Naoufel555/STCray-Dataset
2. Click "Agree and access repository"
3. Wait for approval (usually instant)
4. Try download again

### Issue: "huggingface-cli: command not found"

**Solution:** Install HuggingFace CLI:
```bash
pip install huggingface_hub[cli]
# or
pip install -U huggingface_hub[cli]
```

### Issue: "git-lfs: command not found"

**Solution:** Install Git LFS:
```bash
# macOS:
brew install git-lfs

# Ubuntu:
sudo apt-get install git-lfs

# Then initialize:
git lfs install
```

### Issue: "unrar: command not found"

**Solution:** Install unrar:
```bash
# macOS:
brew install unrar

# Ubuntu:
sudo apt-get install unrar
```

### Issue: RAR extraction fails or is corrupted

**Solution:**
1. Delete the corrupted RAR file
2. Re-download:
   ```bash
   rm data/stcray_raw/STCray_TrainSet.rar
   ./scripts/download_stcray_rar.sh
   ```

### Issue: Git push fails with "file too large"

**Cause:** Git LFS might not be properly initialized.

**Solution:**
```bash
# Re-initialize Git LFS
./scripts/setup_git_lfs.sh

# Verify RAR files are tracked by LFS
git lfs ls-files

# If not listed, track them manually:
git lfs track "data/stcray_raw/*.rar"
git add .gitattributes
git add data/stcray_raw/*.rar
git commit -m "Track RAR files with LFS"
```

---

## Next Steps

After downloading and extracting:

1. **Generate VQA Dataset** (local with Gemini):
   ```bash
   ./scripts/generate_vqa_gemini.sh
   ```

2. **Or process extracted data** (if you have a custom processor):
   ```bash
   python data/process_stcray_extracted.py \
     --input-dir data/stcray_raw/STCray_TrainSet \
     --output-dir data/stcray/train
   ```

3. **Upload to CAI for training** (GitHub Actions):
   - Push to GitHub (with LFS)
   - Use GitHub Actions to deploy to CAI
   - RAR files will be available in CAI workspace

---

## Git LFS Summary

**What's tracked by LFS:**
- `*.rar` files (dataset archives)
- `*.bin`, `*.safetensors` (model files, if added)
- Other large binary files

**What's ignored by git:**
- Extracted directories (`STCray_TrainSet/`, `STCray_TestSet/`)
- Generated VQA files (handled separately)
- Virtual environments
- Cache directories

**Storage locations:**
- Git LFS: RAR archives (~2-23 GB depending on what you download)
- Git regular: Code, scripts, configs
- Ignored: Extracted files, temporary files

This keeps your git repository fast while preserving the original dataset files! 🚀
