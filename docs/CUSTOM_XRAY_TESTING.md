# Custom X-ray Image Testing Guide

Guide for testing RolmOCR on simpler container/baggage X-ray images.

## Why Test on Simpler Images?

STCray dataset is very challenging:
- Complex overlapping items
- Heavy occlusion and concealment
- Dense cluttered baggage
- 21 different threat categories

Testing on clearer, simpler X-ray scans helps evaluate baseline model performance.

## Workflow

### Step 1: Find and Download X-ray Images

**Recommended sources:**

1. **Google Images**
   - Search: "airport baggage x-ray knife" or "container x-ray scanning"
   - Filter: Tools → Usage rights → "Labeled for reuse"
   - Download 3-5 clear images with visible prohibited items

2. **Public Datasets**
   - OPIXray: Has publicly available sample images
   - SIXray: Academic use, some samples available
   - Government security websites: Often have sample X-ray images

3. **Stock Photo Sites**
   - Unsplash, Pexels, Pixabay
   - Search: "x-ray security scan", "airport scanner"

**Download using the helper script:**

```bash
# Method 1: Provide URLs directly
python3 scripts/download_sample_xrays.py \
    --urls \
        https://example.com/xray_knife.jpg \
        https://example.com/xray_gun.jpg \
        https://example.com/xray_scissors.jpg \
    --create-templates

# Method 2: Download manually, then create templates
mkdir -p data/test_xrays/images
# ... download images to data/test_xrays/images/ ...

python3 scripts/download_sample_xrays.py \
    --output-dir data/test_xrays \
    --create-templates
```

### Step 2: Create Annotations

**Option A: Interactive CLI Tool**

```bash
python3 scripts/create_manual_annotation.py \
    --image data/test_xrays/images/xray_knife.jpg

# Follow prompts:
# Item 1:
#   Category (1-19 or custom): 1        # Select "Knife"
#   Item name (default: Knife): Folding knife
#   Location (e.g., center): center
#   Confidence (0.0-1.0): 1.0
#
# Item 2:
#   Category: <press Enter to finish>
```

**Option B: Manual JSON Editing**

Edit the template file created in `data/test_xrays/annotations/`:

```json
{
  "image_filename": "xray_knife.jpg",
  "image_path": "images/xray_knife.jpg",
  "image_path_absolute": "/full/path/to/xray_knife.jpg",
  "categories": ["Knife"],
  "items": [
    {
      "category": "Knife",
      "item_name": "Folding knife",
      "location": "center",
      "confidence": 1.0
    }
  ],
  "num_annotations": 1,
  "notes": "Clear knife visible in center of scan",
  "source": "web_download",
  "annotator": "human"
}
```

### Step 3: Test RolmOCR

```bash
export JWT_TOKEN="your-jwt-token"

# Test with specific image and annotation
python3 test_rolmocr.py \
    --image data/test_xrays/images/xray_knife.jpg \
    --annotations data/test_xrays/annotations/xray_knife_annotation.json

# Test random image from test set
python3 test_rolmocr.py \
    --image-dir data/test_xrays/images
```

### Step 4: Analyze Results

The script will output:
- **Detected items** with categories, confidence, and locations
- **Ground truth comparison** with precision, recall, F1 score
- **Detailed metrics**: True positives, false positives, false negatives

Example output:
```
======================================================================
Performance Metrics
======================================================================
Precision: 100.0%
Recall: 100.0%
F1 Score: 1.000
Exact Match: ✓ Yes

Detailed Comparison:
  True Positives: ['Knife']
  False Positives: []
  False Negatives: []
```

## Annotation Guidelines

### Categories to Look For

**High Priority Threats:**
- Knife, Gun, Explosive, Blade, Weapon

**Medium Priority:**
- Scissors, Cutter, Razor, Needle, Syringe

**Common Items:**
- Battery, Lighter, Tool (Hammer, Wrench, Screwdriver, Pliers)

**Other Objects:**
- Handcuffs, Sharp Object, Metal Object

### Location Descriptors

Use these standard locations:
- `center` - Middle of scan
- `upper-left`, `upper`, `upper-right`
- `left`, `right`
- `lower-left`, `lower`, `lower-right`

### Confidence Scores

For manual annotations:
- `1.0` - Clearly visible, unambiguous
- `0.8-0.9` - Visible but partially occluded
- `0.6-0.7` - Difficult to identify, unclear

## Example Test Cases

### Easy Case: Single Knife
```json
{
  "categories": ["Knife"],
  "items": [{"category": "Knife", "item_name": "Kitchen knife", "location": "center"}]
}
```

### Medium Case: Multiple Items
```json
{
  "categories": ["Knife", "Scissors"],
  "items": [
    {"category": "Knife", "item_name": "Pocket knife", "location": "center"},
    {"category": "Scissors", "item_name": "Scissors", "location": "upper-right"}
  ]
}
```

### Hard Case: Overlapping Items
```json
{
  "categories": ["Gun", "Knife"],
  "items": [
    {"category": "Gun", "item_name": "Handgun", "location": "center", "confidence": 0.8},
    {"category": "Knife", "item_name": "Knife (partially hidden)", "location": "center", "confidence": 0.6}
  ]
}
```

## Comparison with STCray

**STCray (Complex)**:
- 46k images, 21 categories
- Multiple overlapping items per image
- Heavy occlusion and concealment
- Varied baggage contents

**Custom Test Images (Simpler)**:
- 3-10 images, clear examples
- 1-3 items per image
- Minimal occlusion
- Focused on specific threat types

## Expected Results

On simpler images, RolmOCR should achieve:
- **Precision**: 80-100% (if items are clearly visible)
- **Recall**: 70-90% (may miss some items)
- **F1 Score**: 0.75-0.95

If performance is still poor on clear images, it suggests:
- RolmOCR may not be trained for X-ray images (it's designed for OCR/documents)
- YOLO approach would be more suitable
- Consider using X-ray specific models

## Quick Start Example

```bash
# 1. Download a few clear X-ray images from Google
mkdir -p data/test_xrays/images
# ... save images to data/test_xrays/images/ ...

# 2. Annotate first image
python3 scripts/create_manual_annotation.py \
    --image data/test_xrays/images/clear_knife.jpg

# 3. Test RolmOCR
export JWT_TOKEN="your-token"
python3 test_rolmocr.py \
    --image data/test_xrays/images/clear_knife.jpg \
    --annotations data/test_xrays/images/clear_knife_annotation.json

# 4. Review metrics and compare with YOLO approach
```

## Next Steps

Based on test results:
- **If RolmOCR performs well** (F1 > 0.8): Consider fine-tuning on STCray
- **If RolmOCR performs poorly** (F1 < 0.5): Stick with YOLO approach
- **Mixed results**: Use YOLO for detection, RolmOCR for additional analysis

## Files Created

- `scripts/download_sample_xrays.py` - Download helper
- `scripts/create_manual_annotation.py` - Interactive annotation tool
- `test_rolmocr.py` - Updated with custom annotation support
- `data/test_xrays/` - Suggested directory structure

---

**Note**: RolmOCR is an OCR model (text extraction from documents), not specifically trained for X-ray object detection. If testing shows poor performance, the YOLO approach is recommended for production X-ray screening.
