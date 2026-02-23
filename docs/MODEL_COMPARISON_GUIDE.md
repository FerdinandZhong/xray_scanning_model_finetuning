# Model Comparison Guide

This guide explains how to compare the YOLO API results with existing GPT-4 and RolmOCR test results.

## Overview

The comparison script (`scripts/compare_models_with_yolo.py`) will:

1. **Load existing results** - GPT-4 and RolmOCR test results from previous runs
2. **Test YOLO API** - Run the same images through your deployed YOLO API
3. **Calculate metrics** - Compare accuracy, predictions, and performance
4. **Generate report** - Create a comprehensive markdown comparison report

## Prerequisites

1. **Deployed YOLO API** - Your YOLO application must be running
2. **Existing test results** - GPT-4 and RolmOCR results in `test_results/` directory
3. **Test images** - Original X-ray images in `data/luggage_xray_yolo/images/valid/`

## Quick Start

### Basic Usage

Test with default settings (uses your deployed API URL):

```bash
python scripts/compare_models_with_yolo.py
```

### Custom API URL

If your API is at a different URL:

```bash
python scripts/compare_models_with_yolo.py \
  --api-url https://xray-yolo-api.[your-domain]
```

### Limit Number of Samples

Test with a subset of images (faster for quick checks):

```bash
# Test first 10 samples
python scripts/compare_models_with_yolo.py --num-samples 10

# Test first 20 samples
python scripts/compare_models_with_yolo.py --num-samples 20
```

### Custom Output Path

```bash
python scripts/compare_models_with_yolo.py \
  --output test_results/my_comparison.md
```

## Complete Example

```bash
python scripts/compare_models_with_yolo.py \
  --api-url https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site \
  --num-samples 50 \
  --output test_results/yolo_vs_vlm_comparison.md
```

## Output Files

The script generates two files:

1. **Comparison Report** (Markdown)
   - Location: `test_results/model_comparison_with_yolo.md` (default)
   - Contains: Accuracy metrics, category breakdown, sample-by-sample comparison

2. **YOLO Results** (JSON)
   - Location: `test_results/model_comparison_with_yolo_yolo_results.json`
   - Contains: Full detection results for each test image

## Report Contents

The comparison report includes:

### 1. Overall Performance

| Model | Accuracy | Correct | Total | Avg Detections | Threats Found |
|-------|----------|---------|-------|----------------|---------------|
| GPT-4o | 85.2% | 43/50 | 50 | N/A | N/A |
| RolmOCR | 78.4% | 39/50 | 50 | N/A | N/A |
| YOLO API | 72.0% | 36/50 | 50 | 3.2 | 8 |

### 2. Category-wise Comparison

Top predicted categories by each model (shows distribution of predictions).

### 3. Sample-by-Sample Results

Detailed comparison of predictions for each test image with ground truth.

### 4. Key Findings

- Best performing model
- Average detection statistics
- Performance characteristics of each model

### 5. Threat Detection

List of images where YOLO detected potential threats.

## Understanding the Results

### Accuracy Comparison

- **GPT-4o** - Visual Language Model with detailed scene understanding
- **RolmOCR** - OCR-focused VLM, good at text and objects
- **YOLO API** - Fast object detection, real-time capable

### Expected Performance

The pre-trained YOLO model (yolov8x.pt) is trained on COCO dataset, **not X-ray images**.

**Expected accuracy order:**
1. GPT-4o (~85%) - Best scene understanding
2. RolmOCR (~78%) - Good object recognition
3. YOLO API (~65-75%) - Not trained on X-ray data

**To improve YOLO accuracy:**
- Train YOLO on the luggage X-ray dataset
- Fine-tune with X-ray specific classes
- Use the trained model from `runs/detect/yolo_luggage_*/weights/best.pt`

### Speed vs Accuracy Trade-off

| Model | Inference Time | Accuracy | Use Case |
|-------|----------------|----------|----------|
| GPT-4o | 5-10 seconds | ~85% | High accuracy, offline analysis |
| RolmOCR | 5-8 seconds | ~78% | Text-heavy images |
| YOLO API | <100ms | ~70% | Real-time screening, high throughput |

## Troubleshooting

### API Connection Errors

```
❌ Health check failed: Connection refused
```

**Solutions:**
- Check if your YOLO application is running in CAI
- Verify the API URL is correct
- Test the health endpoint manually: `curl https://[api-url]/health`

### Missing Test Results

```
Warning: Results not found: test_results/gpt4_luggage/gpt4_luggage_results.json
```

**Solutions:**
- Ensure you've run the GPT-4 and RolmOCR tests first
- Check the paths in `test_results/` directory
- Run the VLM tests: `python scripts/test_luggage_vlm.py`

### Image Not Found

```
⚠ Image not found, skipping
```

**Solutions:**
- Verify images exist in `data/luggage_xray_yolo/images/valid/`
- Check the dataset was downloaded correctly
- Re-run data download: `python cai_integration/download_luggage_xray.py`

### API Timeout

```
⚠ API call failed: Timeout
```

**Solutions:**
- Increase timeout in the script (line with `timeout=30`)
- Check if the model is loaded in the API (may take 30-60s on first call)
- Test with fewer samples first: `--num-samples 5`

## Advanced Usage

### Modifying Category Mapping

The script maps YOLO detections to categories. Edit `_map_yolo_to_category()` to customize:

```python
def _map_yolo_to_category(self, detections: List[Dict[str, Any]]) -> str:
    """Map YOLO detections to a single category for comparison."""
    if not detections:
        return "unknown"
    
    # Use highest confidence detection
    best_detection = max(detections, key=lambda d: d['confidence'])
    return best_detection['label']
```

### Using Different Test Sets

To test on a different set of images, modify the script to:
1. Load your custom test results
2. Update `samples` to point to your images
3. Adjust ground truth labels as needed

### Batch Testing with Different Models

Test multiple YOLO models (trained vs pre-trained):

```bash
# Pre-trained model
python scripts/compare_models_with_yolo.py \
  --output test_results/comparison_pretrained.md

# Update your CAI app to use trained model
# Then run again
python scripts/compare_models_with_yolo.py \
  --output test_results/comparison_trained.md
```

## Next Steps

After running the comparison:

1. **Review the report** - Understand strengths/weaknesses of each model
2. **Train YOLO** - Fine-tune on X-ray dataset to improve accuracy
3. **Update deployment** - Deploy the trained YOLO model
4. **Re-compare** - Run comparison again with trained model
5. **Benchmark** - Measure inference speed and throughput

## Related Guides

- [API Testing Guide](API_TESTING_GUIDE.md) - Test individual endpoints
- [Deployment Guide](DEPLOYMENT_QUICK_START.md) - Deploy YOLO application
- [Training Guide](../README.md) - Train YOLO on X-ray dataset

## Example Output

Here's what a successful run looks like:

```
🔬 YOLO API Model Comparison
   API: https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site
   Samples: 50

✓ Loaded GPT-4 results: 50 samples
✓ Loaded RolmOCR results: 50 samples

============================================================
Testing YOLO API on Sample Images
============================================================

Checking API health...
✓ API is healthy

Testing 50 images...

[1/50] valid_000000.jpg (GT: CartonDrinks)... ✗ Predicted: bottle (3 detections)
[2/50] valid_000001.jpg (GT: Knife)... ✓ Predicted: knife (1 detections)
[3/50] valid_000002.jpg (GT: Laptop)... ✓ Predicted: laptop (2 detections)
...

✓ Completed 50 samples

============================================================
Summary
============================================================

GPT-4o:
  Accuracy: 86.0%
  Correct:  43/50

RolmOCR:
  Accuracy: 78.0%
  Correct:  39/50

YOLO API:
  Accuracy: 72.0%
  Correct:  36/50
  Avg Detections: 3.2
  Threats Found: 8

Generating comparison report...
✓ Report saved: test_results/model_comparison_with_yolo.md
✓ YOLO results saved: test_results/model_comparison_with_yolo_yolo_results.json

============================================================
✅ Comparison Complete!
============================================================

View report: test_results/model_comparison_with_yolo.md
```

## Tips

1. **Start small** - Test with 5-10 samples first to verify everything works
2. **Check API health** - Always verify the API is responding before full test
3. **Compare trained models** - After training, deploy and compare again
4. **Monitor performance** - Track inference time along with accuracy
5. **Review samples** - Look at individual predictions to understand model behavior

## Support

If you encounter issues:

1. Check the [API Testing Guide](API_TESTING_GUIDE.md) for API troubleshooting
2. Verify your YOLO application is running in CAI
3. Test the API manually with curl first
4. Check the application logs in CAI for errors
