# RolmOCR Testing on CargoXray Dataset

Guide for testing RolmOCR model on the CargoXray dataset with confusion matrix analysis.

## Overview

This guide shows how to:
1. Test RolmOCR on 100 random CargoXray samples
2. Compute confusion matrix across 16 cargo categories
3. Analyze per-category performance
4. Compare with STCray results

## Why Test on CargoXray?

**CargoXray has simpler, clearer images** compared to STCray:
- ✅ Larger objects (easier to recognize)
- ✅ Less clutter and occlusion
- ✅ Better baseline for RolmOCR capabilities
- ✅ 16 common cargo items (textiles, tools, toys, etc.)

If RolmOCR performs well on CargoXray but poorly on STCray, it indicates that **baggage complexity** (not the model) is the bottleneck.

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install matplotlib seaborn openai scikit-learn

# Set JWT token for RolmOCR API
export JWT_TOKEN='your_stepfun_api_token_here'
```

### Run Test (100 Samples)

```bash
python scripts/test_rolmocr_cargoxray.py \
  --dataset-dir data/cargoxray \
  --split test \
  --num-samples 100 \
  --output-dir test_results/rolmocr_cargoxray
```

**Expected runtime**: ~5-10 minutes (depending on API speed)

## Script Options

### Basic Usage

```bash
python scripts/test_rolmocr_cargoxray.py [OPTIONS]
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-url` | `https://api.stepfun.com/v1` | RolmOCR API endpoint |
| `--model-id` | `stepfun-ai/GOT-OCR2_0` | Model to use |
| `--dataset-dir` | `data/cargoxray` | CargoXray dataset location |
| `--split` | `test` | Which split (train/valid/test) |
| `--num-samples` | `100` | Number of samples to test |
| `--output-dir` | `test_results/rolmocr_cargoxray` | Results output directory |
| `--seed` | `42` | Random seed for sampling |

### Examples

**Test on validation set:**
```bash
python scripts/test_rolmocr_cargoxray.py --split valid --num-samples 50
```

**Test all test images:**
```bash
python scripts/test_rolmocr_cargoxray.py --num-samples 65
```

**Quick test (10 samples):**
```bash
python scripts/test_rolmocr_cargoxray.py --num-samples 10
```

## Output Files

The script generates 4 output files:

### 1. `confusion_matrix.png` (Normalized)

16x16 confusion matrix showing **proportions** (0.0 to 1.0).

- **Rows**: True categories (ground truth)
- **Columns**: Predicted categories
- **Diagonal**: Correct predictions (darker = better)
- **Off-diagonal**: Misclassifications

**Example interpretation:**
```
Row "textiles", Column "clothes": 0.35
→ 35% of textile samples were misclassified as clothes
```

### 2. `confusion_matrix_raw.png` (Counts)

Same matrix but showing **raw counts** instead of proportions.

### 3. `rolmocr_cargoxray_results.json`

Detailed JSON results:

```json
{
  "summary": {
    "total_samples": 100,
    "accuracy": 0.45,
    "macro_avg": {
      "precision": 0.42,
      "recall": 0.38,
      "f1-score": 0.39
    }
  },
  "per_category": {
    "textiles": {
      "precision": 0.60,
      "recall": 0.55,
      "f1-score": 0.57,
      "support": 20
    },
    ...
  },
  "predictions": [
    {
      "image": "Pictures-1-1-_jpg.rf.xxx.jpg",
      "ground_truth": "textiles",
      "predicted": "clothes",
      "correct": false
    },
    ...
  ]
}
```

### 4. `rolmocr_cargoxray_report.txt`

Human-readable text report:

```
================================================================================
RolmOCR Performance on CargoXray Dataset
================================================================================

Total Samples: 100
Overall Accuracy: 45.00%

================================================================================
Per-Category Performance
================================================================================

textiles:
  Precision: 60.00%
  Recall: 55.00%
  F1-Score: 57.41%
  Support: 20

...
```

## CargoXray Categories (16)

| ID | Category | Examples | Typical Size |
|----|----------|----------|--------------|
| 0 | auto_parts | Engine parts, filters | Medium-Large |
| 1 | bags | Luggage, backpacks | Medium |
| 2 | bicycle | Bikes, bike frames | Large |
| 3 | car_wheels | Tires, rims | Large |
| 4 | clothes | Garments, apparel | Medium |
| 5 | fabrics | Fabric rolls, textiles | Medium-Large |
| 6 | lamps | Light fixtures | Small-Medium |
| 7 | office_supplies | Stationary, equipment | Small-Medium |
| 8 | shoes | Footwear | Small |
| 9 | spare_parts | Generic parts | Small-Medium |
| 10 | tableware | Dishes, utensils | Small-Medium |
| 11 | textiles | Textile materials | Medium-Large |
| 12 | tools | Hand tools | Small-Medium |
| 13 | toys | Toys, games | Small-Medium |
| 14 | unknown | Unclassified | Varies |
| 15 | xray_objects | Generic items | Varies |

## Expected Performance

### Optimistic Scenario (Best Case)

If RolmOCR is well-suited for cargo X-rays:

| Metric | Expected Value |
|--------|----------------|
| **Overall Accuracy** | 60-70% |
| **Macro Avg Precision** | 55-65% |
| **Macro Avg Recall** | 50-60% |
| **Macro Avg F1** | 52-62% |

**Strong categories**: textiles, tools, bicycle, car_wheels (large, distinctive)
**Weak categories**: unknown, xray_objects, office_supplies (ambiguous)

### Realistic Scenario (Most Likely)

Given RolmOCR is designed for OCR, not object detection:

| Metric | Expected Value |
|--------|----------------|
| **Overall Accuracy** | 30-45% |
| **Macro Avg Precision** | 25-40% |
| **Macro Avg Recall** | 20-35% |
| **Macro Avg F1** | 22-37% |

**Common errors**:
- Confusing textiles ↔ clothes ↔ fabrics
- Misclassifying bags ↔ unknown
- Generic predictions (everything → xray_objects)

### Pessimistic Scenario (Worst Case)

If RolmOCR struggles with X-ray images:

| Metric | Expected Value |
|--------|----------------|
| **Overall Accuracy** | 10-20% |
| **Macro Avg F1** | <15% |

**Indicators**:
- Most predictions are "unknown" or "xray_objects"
- Diagonal of confusion matrix is very light
- No clear pattern in misclassifications

## Interpreting Results

### High Accuracy (>60%)
✅ RolmOCR is effective on cargo X-rays
✅ Proceed to test on STCray
✅ Consider using RolmOCR for cargo screening

### Medium Accuracy (30-60%)
⚠️ RolmOCR has limited capability
⚠️ Better than random, but not production-ready
⚠️ YOLO detection models will outperform significantly

### Low Accuracy (<30%)
❌ RolmOCR is not suitable for X-ray object recognition
❌ Focus on YOLO instead
❌ RolmOCR better for OCR tasks (text extraction)

## Comparison: CargoXray vs STCray

After running both tests, compare:

| Metric | CargoXray | STCray | Insight |
|--------|-----------|--------|---------|
| Accuracy | ? | ? | If CargoXray >> STCray, complexity is the issue |
| Avg F1 | ? | ? | Measures overall robustness |
| Runtime | ~10 min | ~30 min | CargoXray is 3x faster (fewer samples) |

**Decision tree:**
1. **CargoXray good (>50%), STCray poor (<30%)**: Baggage images too complex for RolmOCR
2. **Both poor (<30%)**: RolmOCR not suitable for X-ray object detection
3. **Both good (>50%)**: RolmOCR is viable, consider for production

## Common Issues & Fixes

### Issue 1: JWT_TOKEN Not Set

```
Error: JWT_TOKEN environment variable not set
```

**Fix:**
```bash
export JWT_TOKEN='your_actual_token_here'
```

### Issue 2: API Rate Limiting

```
Error querying RolmOCR: Rate limit exceeded
```

**Fix:**
- Reduce `--num-samples` to 20-50
- Add delays between requests (modify script)
- Wait and retry

### Issue 3: Missing Dependencies

```
ModuleNotFoundError: No module named 'seaborn'
```

**Fix:**
```bash
pip install matplotlib seaborn
```

### Issue 4: Image Not Found

```
Warning: Image not found: data/cargoxray/test/xxx.jpg
```

**Fix:**
- Verify CargoXray is downloaded: `ls data/cargoxray/test/`
- Re-download if needed: See [CARGOXRAY_QUICKSTART.md](CARGOXRAY_QUICKSTART.md)

## Advanced Usage

### Test Specific Categories Only

Modify the script to filter samples:

```python
# In select_samples function, add filtering:
if normalize_category(category_name) in ['textiles', 'tools', 'toys']:
    samples.append((image_path, normalized_name, category_id))
```

### Adjust API Parameters

Modify `query_rolmocr` function:

```python
response = client.chat.completions.create(
    ...
    temperature=0.0,  # More deterministic
    max_tokens=4096,  # Longer responses
)
```

### Batch Processing

For faster testing (if API supports batching):
- Modify script to batch multiple images
- Process results in parallel

## Next Steps

### After Testing CargoXray

1. **Analyze results** - Check confusion matrix and report
2. **Compare with STCray** - Run same test on STCray
3. **Decide on approach**:
   - If RolmOCR works well → Continue with VLM approach
   - If RolmOCR struggles → Focus on YOLO detection

### If RolmOCR Works Well (>50% accuracy)

```bash
# Test on STCray
python test_rolmocr.py \
  --image-dir data/stcray_raw/STCray_TestSet/Images \
  --annotations data/stcray_processed/test/annotations.json \
  --num-samples 100
```

### If RolmOCR Struggles (<30% accuracy)

```bash
# Focus on YOLO training instead
python training/train_yolo.py \
  --data data/cargoxray_yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 100
```

## References

- **Script**: `scripts/test_rolmocr_cargoxray.py`
- **CargoXray Dataset**: [docs/CARGOXRAY_QUICKSTART.md](CARGOXRAY_QUICKSTART.md)
- **Dataset Comparison**: [docs/DATASETS_COMPARISON.md](DATASETS_COMPARISON.md)
- **RolmOCR**: https://api.stepfun.com

---

**Questions?** Check the main [README.md](../README.md) or open an issue!
