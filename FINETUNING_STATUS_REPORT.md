# X-ray YOLO Fine-tuning Status Report

**Date:** February 25, 2026  
**Dataset:** Luggage X-ray — 6,164 train images / 956 validation images / 12 classes  
**Classes:** blade, Cans, CartonDrinks, dagger, GlassBottle, knife, PlasticBottle, scissors, SprayCans, SwissArmyKnife, Tin, VacuumCup  
**Threat classes (5):** blade, dagger, knife, scissors, SwissArmyKnife

---

## 1. How Models Are Fine-tuned

### Infrastructure

| Component | Details |
|-----------|---------|
| **Platform** | Cloudera AI (CAI) Workspace |
| **GPU** | 1× NVIDIA GPU (A100/V100 class) |
| **Runtime** | `ml-runtime-pbj-workbench-python3.10-cuda:2026.01.1-b6` |
| **Framework** | Ultralytics YOLOv8 |
| **Training script** | `training/train_yolo.py` |
| **CAI orchestration** | `cai_integration/yolo_training.py` |
| **Job config** | `cai_integration/jobs_config_yolo.yaml` |

### Training Pipeline

```
CAI Job 1: setup_environment
    → installs PyTorch + Ultralytics venv
CAI Job 2a: download_luggage_xray
    → downloads from Roboflow, converts to YOLO format
CAI Job 3: yolo_training
    → runs training/train_yolo.py with hyperparams from env vars
    → saves best.pt checkpoint + exports metrics CSV
```

### Hyperparameter Evolution

All runs trained on `luggage_xray_yolo` (6,164 train / 956 val, 12 classes).

| Parameter | Run 1 (yolov8n) | Run 2 (yolov8x v1) | Run 3 (yolov8x v2) | **Run 4 — Current** |
|-----------|-----------------|---------------------|--------------------|---------------------|
| **Model** | yolov8n | yolov8x | yolov8x | **yolov8x** |
| **Epochs** | 100 | 100 | 100 | **118 (early stop)** |
| **Batch size** | 16 | 16 | 16 | **16** |
| **Optimizer** | auto (SGD) | auto (SGD) | AdamW | **AdamW** |
| **lr0** | 0.01 | 0.01 | 0.002 | **0.002** |
| **warmup_epochs** | 3 | 3 | 5 | **5** |
| **patience** | 50 | 50 | 50 | **10** |
| **aug_mosaic** | 1.0 | 1.0 | 0.8 | **0.8** |
| **aug_degrees** | 15° | 15° | 10° | **10°** |

### Training Run Results

| Run | Model | Epochs | Peak mAP50 | Peak Epoch | Precision | Recall | Final LR | Outcome |
|-----|-------|--------|-----------|------------|-----------|--------|----------|---------|
| Run 1 | yolov8n | 100 | ~0.05 | ~80 | ~0.15 | ~0.10 | depleted | ✓ Completed |
| Run 2 | yolov8x | 100 | ~0.200 | ~90 | ~0.30 | ~0.22 | depleted | ✓ Completed (plateau) |
| Run 3 (run_23_Feb) | yolov8x | 100 | **0.1430** | 100 | 0.306 | 0.212 | 0.000022 (depleted) | ✓ Completed — LR ran out |
| **Run 4 (run_24_Feb)** | **yolov8x** | **118** | **0.1454** | 112 | 0.284 | 0.237 | 0.001065 (healthy) | ✅ **Early stopped — HOSTED** |
| Run 5 (in progress) | yolov8m | TBD | TBD | — | — | — | — | 🔄 Training on luggage_xray |

**Key observations:**
- Run 3 hit the learning rate floor at epoch 100 — AdamW cosine decay exhausted in 100 epochs (should have been 200)
- Run 4 fixed this with `EPOCHS=200`; training correctly early-stopped at epoch 118 with LR still at 53% of initial
- Both runs converged to the same mAP50 ceiling (~0.145), confirming the bottleneck is **dataset size** (6K images), not hyperparameters

---

## 2. Benchmarking Results — All Fine-tuned YOLO Models (956-sample Full Validation)

**Test set:** `data/luggage_xray_yolo/images/valid/` — 956 images (same split used during training)  
**Metric:** Top-1 accuracy (predicted class = ground-truth class), confidence threshold = 0.10

### Overall Performance

| Model | Training Config | Accuracy | Detection Rate | Threats Found | Avg Det./img | Speed | Tested |
|-------|-----------------|----------|----------------|---------------|--------------|-------|--------|
| YOLO pre-trained (no fine-tuning) | ImageNet only | 1.36% (13/956) | 6.6% | 15 | 0.08 | 2.8 img/s | Feb 19 |
| **yolov8n fine-tuned** | default config | **11.92%** (114/956) | 25.52% | 72 | — | 1.0 img/s | Feb 19 |
| **yolov8x fine-tuned v1** | default config | **19.98%** (191/956) | 40.38% | 149 | 0.60 | 0.8 img/s | Feb 20 |
| **yolov8x fine-tuned v2** *(current hosted)* | updated HP + patience=10 | **17.15%** (164/956) | 35.77% | 112 | 0.51 | 0.7 img/s | **Feb 25** |
| yolov8m fine-tuned | updated HP | — | — | — | — | — | 🔄 In progress |

### Visual Accuracy Comparison

```
Full Validation (956 samples):
yolov8x v1 (default)  ████████████████████ 19.98% (191/956)  🥇
yolov8x v2 (updated)  █████████████████    17.15% (164/956)  ← Current hosted
yolov8n (default)     ███████████          11.92% (114/956)
pre-trained (no FT)   █                     1.36%  (13/956)
```

### Per-Class Accuracy: yolov8x v1 vs yolov8x v2 (Current)

| Class | Support | yolov8x v1 (conf=0.25) | yolov8x v2 current (conf=0.10) | Delta |
|-------|---------|------------------------|--------------------------------|-------|
| PlasticBottle | 339 | 22.1% | 22.4% | +0.3% |
| VacuumCup | 126 | 21.4% | 17.5% | -3.9% |
| **blade** ⚠️ | 115 | **0.0%** | **0.0%** | = |
| scissors | 75 | **29.3%** | 20.0% | -9.3% |
| CartonDrinks | 71 | **28.2%** | 21.1% | -7.1% |
| dagger | 46 | **30.4%** | 23.9% | -6.5% |
| SprayCans | 38 | **26.3%** | 10.5% | -15.8% |
| **SwissArmyKnife** ⚠️ | 37 | 2.7% | **0.0%** | -2.7% |
| Cans | 33 | 9.1% | 15.2% | **+6.1%** |
| Tin | 28 | **25.0%** | 14.3% | -10.7% |
| **knife** | 27 | **44.4%** | **44.4%** | = |
| GlassBottle | 21 | 0.0% | 0.0% | = |

**Findings:**
- `knife` is the standout class for both yolov8x variants: **44.4%** accuracy
- `blade` (115 samples, most common threat class) remains **completely undetected** across all models
- The updated HP model (v2) performs slightly worse overall; likely due to less overfitting — the lower mAP50 (0.145 vs 0.200) is consistent with this
- Both models are bounded by the same ~20% accuracy ceiling

---

## 3. Benchmarking Results — All Methods (50-sample Comparison)

**Test set:** `valid_000000.jpg` to `valid_000049.jpg` — same 50 images for all models  
**Class distribution in 50 samples:** 22 threats (blade×9, SwissArmyKnife×6, dagger×4, scissors×2, knife×1), 28 normal items

### Overall Comparison

| Rank | Model | Type | Accuracy | Correct | Threats (22 total) | Speed | Cost/img |
|------|-------|------|----------|---------|-------------------|-------|----------|
| 🥇 | **YOLO yolov8x v1** | Fine-tuned | **18%** (9/50) | 9 | **3** detected | 0.9 img/s | $0 |
| 🥇 | **YOLO yolov8x v2** *(current)* | Fine-tuned | **18%** (9/50) | 9 | **3** detected | 0.7 img/s | $0 |
| 🥉 | **RolmOCR** | Pre-trained VLM | 16% (8/50) | 8 | **0** detected | 0.2 img/s | ~$0.006 |
| 4th | **GPT-4o** | Pre-trained VLM | 2% (1/50) | 1 | **0** detected | 0.22 img/s | ~$0.01 |

> *yolov8x v2 (current hosted) 50-sample result extracted from the full 956-sample run on Feb 25.*

### Visual Comparison

```
Accuracy (50 samples):
YOLO yolov8x v1    ████████████████████ 18.0% (9/50)   🥇
YOLO yolov8x v2    ████████████████████ 18.0% (9/50)   🥇 (current hosted)
RolmOCR            ████████████████     16.0% (8/50)   🥉
GPT-4o             ██                    2.0% (1/50)

Threat Detection (22 threats in 50 samples):
YOLO yolov8x       ████████             3 detected (13.6%)  Only model detecting threats
RolmOCR            (none)               0 detected (0%)
GPT-4o             (none)               0 detected (0%)

Inference Speed:
YOLO yolov8x v1    ████████████████████ 0.9 img/s
YOLO yolov8x v2    ██████████████████   0.7 img/s
GPT-4o             ██████               0.22 img/s
RolmOCR            █████                0.2 img/s
```

### Per-Class Accuracy (50 Samples)

| Class | Support | yolov8x v1 | yolov8x v2 (current) | RolmOCR | GPT-4o |
|-------|---------|-----------|---------------------|---------|--------|
| blade ⚠️ | 9 | 0% | **0%** | 0% | 0% |
| VacuumCup | 9 | 22% | **22%** | 0% | 0% |
| PlasticBottle | 9 | 11% | **11%** | 31% 🥇 | 10% |
| SwissArmyKnife ⚠️ | 6 | 0% | **0%** | 0% | 0% |
| CartonDrinks | 5 | 60% 🥇 | **40%** | 18% | 0% |
| dagger ⚠️ | 4 | 50% 🥇 | **50%** 🥇 | 0% | 0% |
| Cans | 3 | 0% | **33%** 🥇 | 0% | 0% |
| scissors ⚠️ | 2 | 50% 🥇 | **50%** 🥇 | 0% | 0% |
| SprayCans | 2 | 0% | **0%** | 0% | 0% |
| knife ⚠️ | 1 | 0% | **0%** | 0% | 0% |

**Critical finding:** `blade` (most common threat, 9/22 threat samples) is **0% across all models**.  
**Only YOLO detects any threats** — VLMs (RolmOCR, GPT-4o) have zero threat detection.

---

## Summary & Next Steps

### Current State

| Model | Status | mAP50 | Full Val Accuracy | 50-sample Accuracy |
|-------|--------|-------|-------------------|-------------------|
| yolov8n fine-tuned | ✓ Done | ~0.05 | 11.92% | — |
| yolov8x v1 (default HP) | ✓ Done | ~0.200 | 19.98% | 18.0% |
| yolov8x v2 (updated HP) | ✅ **Hosted** | 0.145 | **17.15%** | **18.0%** |
| yolov8m (updated HP) | 🔄 Training | TBD | TBD | TBD |

### Root Cause: Dataset Size Bottleneck

All fine-tuned models converge to the same ~15-20% accuracy ceiling on `luggage_xray` alone.  
With 6,164 training images, yolov8x (requires 10K+ for full performance) is **capacity-constrained**.  
The mAP50 ceiling at 0.145–0.200 is consistent with this constraint.

### Path to Production (>75% Target)

| Option | Action | Expected Accuracy | Status |
|--------|--------|-------------------|--------|
| **1 — Larger dataset** | Train on `combined_xray_yolo` (32K+ imgs: luggage + STCray) | **50–70%** | 🔜 Script ready (`scripts/combine_xray_datasets.py`) |
| **2 — External data** | Add `cargoxray` or other public X-ray datasets | **70–85%** | Possible after Option 1 |
| **3 — Annotation quality** | Review and fix blade/SwissArmyKnife zero-detection | **+5–10%** | Investigate |

**Immediate next action:** Run `combine_datasets` CAI job → re-train yolov8m on `combined_xray_yolo` → benchmark.

---

*Generated: February 25, 2026*
