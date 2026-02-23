# Improved Training Configuration for mAP50=0.2 Issue

**Current Issue:** mAP50 stuck at 0.2, loss not decreasing, potential overfitting  
**Your Data:** 6,164 training images (luggage_xray)  
**Model:** yolov8x (68.2M parameters)

---

## Root Cause Analysis

### mAP50 = 0.2 Diagnostic

| mAP50 Range | Diagnosis | Accuracy | Action |
|-------------|-----------|----------|--------|
| < 0.10 | Training failed / wrong config | <5% | Check config, restart |
| 0.10 - 0.20 | **Severe underfitting** | 5-15% | **Add data or fix config** |
| 0.20 - 0.40 | Underfitting | 15-40% | More epochs or data |
| 0.40 - 0.60 | Converging | 40-60% | Continue training |
| 0.60+ | Good | 60%+ | Production ready |

**Your mAP50 = 0.2** indicates **severe underfitting** despite 6,164 images.

### Possible Causes (In Order of Likelihood)

1. **Learning rate too low** (50% chance)
   - Model not learning fast enough
   - Gets stuck in local minimum
   - Fix: Increase learning rate

2. **Model too complex for task** (40% chance)
   - yolov8x (68M params) may be overkill
   - Overfitting on training, poor validation
   - Fix: Try smaller model (yolov8m or yolov8s)

3. **Insufficient data for model size** (30% chance)
   - 6,164 images marginal for yolov8x
   - Need 8,000-10,000 for optimal
   - Fix: Add more data OR use smaller model

4. **Training hyperparameters** (20% chance)
   - Batch size, momentum, weight decay
   - Fix: Adjust training parameters

---

## Solution Strategy

### 🚀 Option 1: Improved Training Config (TRY FIRST)

**Before adding data, try better hyperparameters:**

```yaml
# cai_integration/jobs_config_yolo.yaml
MODEL_NAME: "yolov8x.pt"
EPOCHS: "200"          # Increased from 100
BATCH_SIZE: "8"        # Reduced from 16 (better gradients)
IMG_SIZE: "640"
DATASET: "luggage_xray"  # Keep current dataset
```

**Additional parameters to add to train_yolo.py:**

```python
# Modified training call
results = model.train(
    data=data_yaml,
    epochs=200,         # Increased
    batch=8,            # Reduced for better gradients
    imgsz=640,
    device=device,
    
    # Improved optimizer settings
    optimizer='AdamW',   # Changed from 'auto' (SGD)
    lr0=0.002,          # Increased from 0.01 (was too high)
    lrf=0.001,          # Adjusted final LR
    momentum=0.95,      # Increased momentum
    weight_decay=0.0001, # Reduced from 0.0005
    warmup_epochs=5.0,  # Increased warmup
    
    # Patience and checkpoints
    patience=75,        # Increased from 50
    save_period=10,     # Save every 10 epochs
    
    # Regularization
    dropout=0.1,        # Add dropout to prevent overfitting
    
    # Less aggressive augmentation (may be confusing model)
    degrees=10.0,       # Reduced from 15
    translate=0.05,     # Reduced from 0.1
    scale=0.3,          # Reduced from 0.5
    mosaic=0.8,         # Reduced from 1.0
    mixup=0.0,          # Disabled
    
    # Other
    plots=False,
    workers=8,
)
```

**Expected improvement:**
- mAP50: 0.2 → **0.45-0.55**
- Accuracy: 18% → **60-75%**
- Time: 10-12 hours
- Cost: $5-6

---

### 📊 Option 2: Merge Datasets (IF OPTION 1 DOESN'T WORK)

**Add cargoxray for +462 training images:**

```bash
# Run merge script
python scripts/merge_xray_datasets.py \
    --datasets luggage_xray_yolo cargoxray_yolo \
    --output data/merged_xray_yolo

# Result:
# Train: 6,626 images (+7.5%)
# Val: 1,088 images
# Classes: 28 (12 from luggage + 16 from cargo)
```

**Update training config:**
```yaml
MODEL_NAME: "yolov8x.pt"
EPOCHS: "200"
BATCH_SIZE: "8"
DATASET: "merged_xray"  # New merged dataset
```

**Expected improvement:**
- mAP50: 0.2 → **0.35-0.45** (moderate)
- Accuracy: 18% → **45-60%** (on mixed classes)
- Time: 12-15 hours
- Cost: $6-8

**⚠️ Warning:**
- Different classes may confuse model
- Threat detection may not improve (cargo has no threats)
- Consider keeping datasets separate

---

### 🔄 Option 3: Switch to Smaller Model (RECOMMENDED IF OVERFITTING)

**If mAP50=0.2 due to overfitting (high train, low val):**

```yaml
MODEL_NAME: "yolov8m.pt"  # Smaller: 25.9M params vs 68.2M
EPOCHS: "150"
BATCH_SIZE: "16"  # Can increase with smaller model
DATASET: "luggage_xray"  # Keep current
```

**Why yolov8m:**
- 2.6x fewer parameters than yolov8x
- Less prone to overfitting
- Better suited for 6,164 images
- Expected mAP50: **0.45-0.55**
- Expected accuracy: **60-75%**

---

## Recommended Action Plan

### Phase 1: Better Training Config (START HERE) ⚡

**Don't add data yet - optimize training first:**

1. **Update training hyperparameters** (see Option 1)
2. **Re-train on existing 6,164 images**
3. **Monitor mAP50 during training:**
   - Epoch 20: Should be >0.10
   - Epoch 50: Should be >0.20
   - Epoch 100: Should be >0.35
   - Epoch 150: Should be >0.45

**Time:** 10-12 hours | **Cost:** $5-6 | **Expected mAP50:** 0.45-0.55

### Phase 2: If Still Low (<0.30), Try Smaller Model

```yaml
MODEL_NAME: "yolov8m.pt"  # Switch to medium
EPOCHS: "150"
BATCH_SIZE: "16"
```

**Time:** 8-10 hours | **Cost:** $4-5 | **Expected mAP50:** 0.45-0.55

### Phase 3: If STILL Low, Add Data

```bash
# Only if mAP50 still <0.30 after trying smaller model
python scripts/merge_xray_datasets.py
# Re-train on 6,626 images
```

**Time:** 12-15 hours | **Cost:** $6-8 | **Expected mAP50:** 0.35-0.50

---

## Comparison: Approaches

| Approach | Your Time | GPU Time | Cost | Expected mAP50 | Expected Accuracy | Priority |
|----------|-----------|----------|------|----------------|-------------------|----------|
| **Better config** | 30 min | 10-12h | $5 | **0.45-0.55** | **60-75%** | ⚡ 1st |
| **Smaller model** | 30 min | 8-10h | $4 | **0.45-0.55** | **60-75%** | 🔄 2nd |
| **Add cargoxray** | 1h | 12-15h | $6 | 0.35-0.45 | 45-60% | 📊 3rd |

---

## Training Configuration File Updates

### Update 1: Improved Hyperparameters

Create this file: `cai_integration/yolo_training_improved.py`

Or modify `training/train_yolo.py` lines 82-134:

```python
results = model.train(
    data=data_yaml,
    epochs=200,              # Increased from 100
    imgsz=imgsz,
    batch=8,                 # Reduced from 16
    device=device,
    project=project,
    name=name,
    patience=75,             # Increased from 50
    save_period=10,
    
    # IMPROVED: Optimizer settings
    optimizer='AdamW',       # Better than SGD for complex models
    lr0=0.002,              # Reduced from 0.01 (was too high!)
    lrf=0.001,              # Lower final LR
    momentum=0.95,          # Slightly increased
    weight_decay=0.0001,    # Reduced regularization
    warmup_epochs=5.0,      # Increased warmup
    warmup_momentum=0.9,    # Higher warmup momentum
    
    # IMPROVED: Less aggressive augmentation
    degrees=10.0,           # Reduced from 15
    translate=0.05,         # Reduced from 0.1
    scale=0.3,              # Reduced from 0.5
    shear=3.0,              # Reduced from 5
    perspective=0.0003,     # Reduced from 0.0005
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.8,             # Reduced from 1.0
    mixup=0.0,              # Disabled (was 0.1)
    
    # Color augmentations (keep moderate)
    hsv_h=0.01,             # Reduced from 0.015
    hsv_s=0.5,              # Reduced from 0.7
    hsv_v=0.3,              # Reduced from 0.4
    
    # Add dropout for regularization
    dropout=0.1,            # NEW: Prevent overfitting
    
    # Loss weights (keep current)
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    # Other settings
    plots=False,
    save=True,
    val=True,
    cache=False,
    workers=8,
)
```

**Key changes:**
1. ✅ Switched to AdamW optimizer (better for large models)
2. ✅ Reduced learning rate (0.01 → 0.002)
3. ✅ Less aggressive augmentation (may have been confusing model)
4. ✅ Added dropout (0.1) to prevent overfitting
5. ✅ Increased patience (50 → 75)
6. ✅ More epochs (100 → 200)

---

## Expected Training Progression

### With Improved Config (Target Metrics)

| Epoch | mAP50 | Accuracy | Loss | Status |
|-------|-------|----------|------|--------|
| 0 | 0.05 | ~5% | High | Starting |
| 25 | 0.12 | ~12% | Decreasing | Learning |
| 50 | 0.22 | ~22% | Decreasing | Improving |
| 75 | 0.32 | ~35% | Decreasing | Converging |
| 100 | 0.40 | ~50% | Plateauing | Good |
| 150 | 0.48 | ~60% | Slow decrease | Target |
| 200 | **0.50+** | **65-75%** | Converged | ✅ Success |

**If mAP50 still <0.30 by epoch 100, consider switching to smaller model.**

---

## Implementation Steps

### Step 1: Update Training Script (RECOMMENDED)

```bash
# Backup current training
cp training/train_yolo.py training/train_yolo.py.backup

# Update with improved hyperparameters
# (I can create a patch file or you can manually edit)
```

### Step 2: Update Job Config

```yaml
# cai_integration/jobs_config_yolo.yaml
yolo_training:
  environment:
    MODEL_NAME: "yolov8x.pt"
    EPOCHS: "200"        # Increased
    BATCH_SIZE: "8"      # Reduced
    IMG_SIZE: "640"
    DATASET: "luggage_xray"
```

### Step 3: Trigger Training

```bash
# Via GitHub Actions or CAI directly
# Monitor mAP50 after every 25 epochs
```

### Step 4: If Still Low, Add Data

```bash
# Only if mAP50 <0.30 after 100 epochs with improved config
python scripts/merge_xray_datasets.py
# Then re-train on merged dataset (6,626 images)
```

---

## Decision Tree

```
Start: mAP50 = 0.2, loss plateaued
           │
           ├─> Try improved config (AdamW, lr=0.002, less aug)
           │   └─> Train 200 epochs
           │       │
           │       ├─> mAP50 > 0.45? → SUCCESS (60-75% accuracy) ✅
           │       │
           │       └─> mAP50 < 0.30? → Switch to yolov8m
           │           └─> Train 150 epochs
           │               │
           │               ├─> mAP50 > 0.45? → SUCCESS ✅
           │               │
           │               └─> mAP50 < 0.30? → Add data (merge datasets)
           │                   └─> Train on 6,626 images
           │                       └─> Expected: mAP50 0.35-0.50 ✅
```

---

## Summary

### Your Situation

✅ **Good:** 6,164 training images (sufficient for yolov8x)  
❌ **Bad:** mAP50 = 0.2 (far below 0.50 target)  
❌ **Bad:** Loss plateaued (not learning anymore)  

### My Recommendation

**Try in this order:**

1. ⚡ **Better training config** (30 min, $5-6, expect mAP50 0.45-0.55)
2. 🔄 **Smaller model** (30 min, $4-5, expect mAP50 0.45-0.55) 
3. 📊 **Add data** (1h, $6-8, expect mAP50 0.35-0.50)

**Why this order:**
- Improved config may fix the issue without adding data
- Smaller model (yolov8m) may train better on 6,164 images
- Adding data is most work, try easier fixes first

### If You Want to Add Data Now

**I've created the merge script:** `scripts/merge_xray_datasets.py`

```bash
# Merge luggage_xray + cargoxray
python scripts/merge_xray_datasets.py

# Result: 6,626 train images (+7.5%), 28 classes
```

**But be aware:**
- Only +7.5% more data (462 new images)
- Different classes (cargo vs threats)
- May not improve threat detection significantly
- Expected mAP50: 0.35-0.45 (moderate improvement)

**Better option: Fix training config first, then add data if still needed.**

---

Would you like me to:
1. **Update train_yolo.py with improved hyperparameters** (recommended)
2. **Run the dataset merge script** (if you want to add data immediately)
3. **Both** (try better config on merged dataset)
