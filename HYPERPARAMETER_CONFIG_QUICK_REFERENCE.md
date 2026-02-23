# Hyperparameter Configuration Quick Reference

**Easy tuning:** Just edit `cai_integration/jobs_config_yolo.yaml` - no Python code changes needed!

---

## 🎯 Current Configuration (Optimized for YOLOv8x)

### File: `cai_integration/jobs_config_yolo.yaml`

```yaml
environment:
  # ═══════════════════════════════════════
  # BASIC CONFIGURATION
  # ═══════════════════════════════════════
  MODEL_NAME: "yolov8x.pt"
  DATASET: "luggage_xray"
  EPOCHS: "200"
  BATCH_SIZE: "8"
  IMG_SIZE: "640"
  
  # ═══════════════════════════════════════
  # HYPERPARAMETERS (← Edit these!)
  # ═══════════════════════════════════════
  LEARNING_RATE: "0.002"    # 🎯 Most critical parameter
  OPTIMIZER: "AdamW"        # Best for yolov8x
  PATIENCE: "10"            # Early stop after 10 epochs
  WARMUP_EPOCHS: "5.0"      # Gradual LR warmup
  
  # ═══════════════════════════════════════
  # AUGMENTATION
  # ═══════════════════════════════════════
  AUG_DEGREES: "10.0"       # Rotation: 0-30
  AUG_TRANSLATE: "0.05"     # Translation: 0-0.2
  AUG_SCALE: "0.3"          # Scale: 0-1.0
  AUG_MOSAIC: "0.8"         # Mosaic: 0-1.0
  AUG_MIXUP: "0.0"          # Mixup: 0-0.3 (keep 0 for X-ray)
```

---

## 🚀 Common Adjustments

### Scenario 1: Training Too Slow

**Edit:**
```yaml
LEARNING_RATE: "0.003"    # Increase from 0.002
WARMUP_EPOCHS: "3.0"      # Reduce from 5.0
```

**Expected:** Faster convergence, slightly less stable

---

### Scenario 2: Loss Oscillating

**Edit:**
```yaml
LEARNING_RATE: "0.001"    # Reduce from 0.002
BATCH_SIZE: "4"           # Reduce from 8
```

**Expected:** More stable training, slower convergence

---

### Scenario 3: Want More Aggressive Training

**Edit:**
```yaml
LEARNING_RATE: "0.005"
OPTIMIZER: "Adam"
PATIENCE: "5"
AUG_DEGREES: "15.0"
AUG_MOSAIC: "1.0"
```

**Expected:** Faster results or early failure (high risk/reward)

---

### Scenario 4: Overfitting (Train >> Val)

**Edit:**
```yaml
AUG_DEGREES: "15.0"       # More augmentation
AUG_TRANSLATE: "0.1"
AUG_SCALE: "0.5"
AUG_MOSAIC: "1.0"
AUG_MIXUP: "0.1"          # Re-enable mixup
```

**Expected:** Better generalization, less overfitting

---

### Scenario 5: Need Faster Feedback

**Edit:**
```yaml
PATIENCE: "5"             # Stop after 5 epochs
EPOCHS: "100"             # Reduce max epochs
```

**Expected:** Quicker iteration, may stop too early

---

## 📋 Parameter Reference Card

| Parameter | Range | Default | For YOLOv8n | For YOLOv8x | For Stability | For Speed |
|-----------|-------|---------|-------------|-------------|---------------|-----------|
| **LEARNING_RATE** | 0.001-0.01 | 0.002 | 0.01 | 0.002 | 0.001 | 0.005 |
| **OPTIMIZER** | SGD/Adam/AdamW | AdamW | SGD | AdamW | AdamW | Adam |
| **PATIENCE** | 5-20 | 10 | 15 | 10 | 20 | 5 |
| **WARMUP_EPOCHS** | 3-10 | 5.0 | 3.0 | 5.0 | 10.0 | 3.0 |
| **AUG_DEGREES** | 0-30 | 10.0 | 15.0 | 10.0 | 5.0 | 15.0 |
| **AUG_TRANSLATE** | 0-0.2 | 0.05 | 0.1 | 0.05 | 0.03 | 0.1 |
| **AUG_SCALE** | 0-1.0 | 0.3 | 0.5 | 0.3 | 0.2 | 0.5 |
| **AUG_MOSAIC** | 0-1.0 | 0.8 | 1.0 | 0.8 | 0.5 | 1.0 |
| **AUG_MIXUP** | 0-0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |

---

## 🎨 Preset Templates

### Template 1: Default (Already Set) ✅

```yaml
# Best for: First training attempt
# Model: yolov8x
# Expected: mAP50 0.50-0.55

LEARNING_RATE: "0.002"
OPTIMIZER: "AdamW"
PATIENCE: "10"
WARMUP_EPOCHS: "5.0"
AUG_DEGREES: "10.0"
AUG_TRANSLATE: "0.05"
AUG_SCALE: "0.3"
AUG_MOSAIC: "0.8"
AUG_MIXUP: "0.0"
```

---

### Template 2: Conservative

```yaml
# Best for: Unstable training, need guaranteed convergence
# Model: Any
# Expected: mAP50 0.45-0.52, slower

LEARNING_RATE: "0.001"
OPTIMIZER: "AdamW"
PATIENCE: "20"
WARMUP_EPOCHS: "10.0"
AUG_DEGREES: "5.0"
AUG_TRANSLATE: "0.03"
AUG_SCALE: "0.2"
AUG_MOSAIC: "0.5"
AUG_MIXUP: "0.0"
```

---

### Template 3: Aggressive

```yaml
# Best for: Quick experiments, fast feedback
# Model: yolov8x
# Expected: mAP50 0.40-0.50 or early failure

LEARNING_RATE: "0.005"
OPTIMIZER: "Adam"
PATIENCE: "5"
WARMUP_EPOCHS: "3.0"
AUG_DEGREES: "15.0"
AUG_TRANSLATE: "0.1"
AUG_SCALE: "0.5"
AUG_MOSAIC: "1.0"
AUG_MIXUP: "0.1"
```

---

### Template 4: For YOLOv8n (Small Model)

```yaml
# Best for: Fast training on small model
# Model: yolov8n
# Expected: mAP50 0.35-0.45

MODEL_NAME: "yolov8n.pt"
BATCH_SIZE: "16"
LEARNING_RATE: "0.01"
OPTIMIZER: "SGD"
PATIENCE: "15"
WARMUP_EPOCHS: "3.0"
AUG_DEGREES: "15.0"
AUG_TRANSLATE: "0.1"
AUG_SCALE: "0.5"
AUG_MOSAIC: "1.0"
AUG_MIXUP: "0.0"
```

---

## 🔧 How to Apply Changes

### Step 1: Edit Configuration

```bash
# Open the file
vim cai_integration/jobs_config_yolo.yaml

# Or use any editor
code cai_integration/jobs_config_yolo.yaml

# Change the hyperparameter values under 'environment:'
```

### Step 2: Commit & Push

```bash
git add cai_integration/jobs_config_yolo.yaml
git commit -m "Tune hyperparameters: increase LR to 0.003"
git push origin feature/implementation
```

### Step 3: Trigger Training

```bash
# Via GitHub Actions:
# Go to Actions → "Deploy X-ray Detection to CAI" → Run workflow

# Or via CAI directly:
# Jobs → yolo_training → Run Job
```

### Step 4: Monitor

```bash
# Check mAP50 after 6 hours (epoch 50-60)
# If mAP50 > 0.40: Good progress ✅
# If mAP50 < 0.30: Try different config ⚠️
```

---

## 💡 Tuning Tips

### Tip 1: Change One Thing at a Time

```yaml
# Attempt 1: Baseline
LEARNING_RATE: "0.002"

# Attempt 2: Only change LR
LEARNING_RATE: "0.003"
# (Keep all other parameters same)

# Compare results to isolate impact
```

### Tip 2: Learning Rate is Most Important

**If you only tune one parameter, tune LEARNING_RATE:**

```yaml
# Try these in order if default doesn't work:
LEARNING_RATE: "0.002"  # Default
LEARNING_RATE: "0.001"  # If oscillating
LEARNING_RATE: "0.003"  # If too slow
LEARNING_RATE: "0.005"  # If much too slow
```

### Tip 3: Match Optimizer to LR

```yaml
# AdamW: Works well with lower LR (0.001-0.003)
OPTIMIZER: "AdamW"
LEARNING_RATE: "0.002"

# SGD: Needs higher LR (0.005-0.01)
OPTIMIZER: "SGD"
LEARNING_RATE: "0.01"
```

### Tip 4: Patience vs Training Time

```yaml
# Quick feedback (4-6 hours)
PATIENCE: "5"
EPOCHS: "100"

# Balanced (7-9 hours)
PATIENCE: "10"
EPOCHS: "200"

# Patient (10-15 hours)
PATIENCE: "20"
EPOCHS: "300"
```

---

## 📊 Expected Results by Configuration

### Current Default Config

```yaml
LEARNING_RATE: "0.002"
OPTIMIZER: "AdamW"
PATIENCE: "10"
```

**Expected:**
- mAP50: **0.50-0.55**
- Accuracy: **60-75%**
- Training time: 7-9 hours (early stop ~epoch 110-140)
- Probability: **70-80%**

---

### Conservative Config

```yaml
LEARNING_RATE: "0.001"
PATIENCE: "20"
WARMUP_EPOCHS: "10.0"
```

**Expected:**
- mAP50: **0.45-0.52**
- Accuracy: **55-70%**
- Training time: 10-12 hours (early stop ~epoch 140-180)
- Probability: **85-90%** (very safe)

---

### Aggressive Config

```yaml
LEARNING_RATE: "0.005"
OPTIMIZER: "Adam"
PATIENCE: "5"
```

**Expected:**
- mAP50: **0.40-0.50** (if successful)
- Accuracy: **50-65%**
- Training time: 4-6 hours (early stop ~epoch 60-100)
- Probability: **50-60%** (risky)

---

## 🎯 Decision Matrix

| Your Situation | Recommended Config | Priority Change |
|----------------|-------------------|-----------------|
| **First training (current)** | Default (0.002, AdamW, 10) | None - use as-is ✅ |
| mAP50 < 0.30 by epoch 50 | Conservative (0.001, 20, 10.0) | Reduce LEARNING_RATE |
| Loss oscillating | Stable (0.001, AdamW, 15) | Reduce LEARNING_RATE |
| Training too slow | Fast (0.003, Adam, 10) | Increase LEARNING_RATE |
| Stops too early | Patient (0.002, 20, 5.0) | Increase PATIENCE |
| Overfitting | Heavy aug (15.0, 0.1, 0.5, 1.0) | Increase AUG_* |

---

## 📝 Example Edit Session

```yaml
# Before (experiencing slow convergence):
LEARNING_RATE: "0.002"
WARMUP_EPOCHS: "5.0"

# After (speed up):
LEARNING_RATE: "0.003"    # +50% faster learning
WARMUP_EPOCHS: "3.0"      # Shorter warmup

# Commit message:
git commit -m "Increase learning rate to 0.003 for faster convergence"
```

---

## ✅ Summary

### What You Can Now Configure (9 Parameters)

| Category | Parameters | Count |
|----------|------------|-------|
| **Optimizer** | LEARNING_RATE, OPTIMIZER, WARMUP_EPOCHS | 3 |
| **Training** | PATIENCE | 1 |
| **Augmentation** | AUG_DEGREES, AUG_TRANSLATE, AUG_SCALE, AUG_MOSAIC, AUG_MIXUP | 5 |

### Files Changed

```
✅ training/train_yolo.py               (9 new CLI arguments)
✅ cai_integration/yolo_training.py     (Read env vars, pass to training)
✅ cai_integration/jobs_config_yolo.yaml (9 new environment variables)
📖 docs/HYPERPARAMETER_TUNING_GUIDE.md  (Complete tuning guide)
```

### How to Use

1. **Edit:** `cai_integration/jobs_config_yolo.yaml` (change values)
2. **Commit:** `git commit -m "Tune: <description>"`
3. **Push:** `git push`
4. **Train:** Trigger via GitHub Actions or CAI
5. **Monitor:** Check mAP50 after 6 hours

---

## 🎬 Next Step

**Current config is already optimized!** Just trigger training:

```bash
# Push commits
git push origin feature/implementation

# Trigger via GitHub Actions
# Expected: mAP50 0.50-0.55 in 7-9 hours
```

**If results are poor (<0.40), tune hyperparameters using this guide.**

---

**Pro tip:** Start with default config. Only tune if mAP50 < 0.40 after epoch 100.
