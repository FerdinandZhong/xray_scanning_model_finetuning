# VQA Dataset Examples

This directory contains example datasets showing the format used for training.

## Dataset Format

Each line in the `.jsonl` file is a JSON object with the following structure:

```json
{
  "image_path": "path/to/image.jpg",
  "question": "What items are visible in this X-ray scan?",
  "answer": "Detected items: a folding knife.",
  "metadata": {
    "image_id": 1,
    "num_annotations": 1,
    "categories": ["Folding_Knife"],
    "has_occlusion": false,
    "question_type": "general",
    "bboxes": [[x, y, w, h]]
  }
}
```

## Field Descriptions

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_path` | string | Relative path to X-ray image |
| `question` | string | VQA question about the image |
| `answer` | string | Ground truth answer for training |
| `metadata` | object | Additional information (see below) |

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | int | Unique identifier for the image |
| `num_annotations` | int | Number of prohibited items detected |
| `categories` | list[string] | Item categories (e.g., "Folding_Knife") |
| `has_occlusion` | bool | Whether items are concealed/occluded |
| `question_type` | string | Type of question (general/specific/location/occlusion/detailed) |
| `bboxes` | list[list[int]] | Bounding boxes [x, y, width, height] |

## Question Types

### 1. General Questions
Ask about all items in the scan.

**Example:**
```json
{
  "question": "What items are visible in this X-ray scan?",
  "answer": "Detected items: a folding knife.",
  "metadata": {
    "question_type": "general"
  }
}
```

### 2. Specific Questions
Ask about a particular item type.

**Positive Example (item present):**
```json
{
  "question": "Is there a scissors in this scan?",
  "answer": "Yes, a scissors is detected at upper-right.",
  "metadata": {
    "question_type": "specific"
  }
}
```

**Negative Example (item absent):**
```json
{
  "question": "Is there a straight knife in this scan?",
  "answer": "No, there is no straight knife in this scan.",
  "metadata": {
    "question_type": "specific"
  }
}
```

### 3. Location Questions
Ask where items are located.

**Example:**
```json
{
  "question": "List all items detected in this scan with their locations.",
  "answer": "Items detected: Folding knife in center-left quadrant, Scissors in upper-right.",
  "metadata": {
    "question_type": "location"
  }
}
```

### 4. Occlusion Questions
Ask about concealed/hidden items.

**Example (with occlusion):**
```json
{
  "question": "Are there any concealed or partially visible items?",
  "answer": "Yes, partially concealed items detected: utility knife at center.",
  "metadata": {
    "question_type": "occlusion",
    "has_occlusion": true
  }
}
```

**Example (without occlusion):**
```json
{
  "question": "Are there any concealed or partially visible items?",
  "answer": "No, all items are clearly visible without concealment.",
  "metadata": {
    "question_type": "occlusion",
    "has_occlusion": false
  }
}
```

### 5. Detailed Questions
Ask for comprehensive description.

**Example:**
```json
{
  "question": "Provide a detailed description of all items in this X-ray scan.",
  "answer": "Detected 3 item(s): folding knife at center-left (partially concealed), scissors at upper-right, utility knife at lower section.",
  "metadata": {
    "question_type": "detailed",
    "num_annotations": 3
  }
}
```

## Item Categories

The dataset uses 5 prohibited item categories from OPIXray:

| Category ID | Category Name | Friendly Name |
|-------------|---------------|---------------|
| 1 | Folding_Knife | folding knife |
| 2 | Straight_Knife | straight knife |
| 3 | Scissor | scissors |
| 4 | Utility_Knife | utility knife |
| 5 | Multi-tool_Knife | multi-tool knife |

## Location Descriptions

Locations are described using a 3x3 grid:

```
upper-left    upper       upper-right
left          center      right
lower-left    lower       lower-right
```

Special cases:
- "center of the image" - middle of middle
- "center-left quadrant" - left side, middle vertically
- "upper section" - top third

## Answer Format Guidelines

### Item Recognition
- Start with "Detected items:" or "No prohibited items detected"
- Use friendly names (lowercase): "folding knife", not "Folding_Knife"
- Include location when relevant: "at center-left"
- Mention occlusion: "partially concealed"

**Good Examples:**
- ✅ "Detected items: a folding knife."
- ✅ "Detected items: a folding knife at center-left, partially concealed."
- ✅ "No prohibited items detected in this scan."

**Avoid:**
- ❌ "FOLDING_KNIFE detected" (use friendly name)
- ❌ "Risk level: HIGH" (no risk assessment in VLM training)
- ❌ "Declaration mismatch" (no declaration comparison)

### Multiple Items
List all items, separated by commas:

```
"Detected items: folding knife, scissors, utility knife."
```

With locations:
```
"Items detected: Folding knife in center-left quadrant, Scissors in upper-right."
```

## Statistics

Typical dataset distribution:
- **Total samples**: ~17,000 (8,885 images × 2 questions per image)
- **Question types**:
  - General: ~35%
  - Specific: ~30%
  - Location: ~15%
  - Occlusion: ~10%
  - Detailed: ~10%
- **With occlusion**: ~20%
- **With multiple items**: ~5%
- **No items (clean scans)**: ~10%

## Sample Dataset File

See [`vqa_dataset_samples.jsonl`](vqa_dataset_samples.jsonl) for 10 representative examples covering all question types.

## Creating Your Own Dataset

```bash
# Generate VQA pairs from OPIXray annotations
python data/create_vqa_pairs.py \
  --opixray-root data/opixray \
  --split all \
  --samples-per-image 2

# Output files:
# - data/opixray_vqa_train.jsonl (~12,000 samples)
# - data/opixray_vqa_val.jsonl (~2,600 samples)
# - data/opixray_vqa_test.jsonl (~2,600 samples)
```

## Important Notes

### What's IN the Training Data
✅ Item recognition: "What items are visible?"
✅ Location information: "Where are items located?"
✅ Occlusion detection: "Are items concealed?"

### What's NOT in the Training Data
❌ Declaration comparison: "Does this match the declaration?"
❌ Risk assessment: "What is the risk level?"
❌ Action recommendations: "Should we inspect?"

**Why?** These are handled in post-processing (`inference/postprocess.py`), not in the VLM. This separation gives better model performance and more flexibility.

## Validation

To validate your dataset:

```bash
# Count samples
wc -l data/opixray_vqa_*.jsonl

# Check format
head -1 data/opixray_vqa_train.jsonl | python -m json.tool

# Statistics
python -c "
import json
with open('data/opixray_vqa_train.jsonl') as f:
    data = [json.loads(line) for line in f]
print(f'Total: {len(data)}')
print(f'With items: {sum(1 for d in data if d[\"metadata\"][\"num_annotations\"] > 0)}')
print(f'With occlusion: {sum(1 for d in data if d[\"metadata\"][\"has_occlusion\"])}')
"
```

## Next Steps

1. Review the sample dataset: `examples/vqa_dataset_samples.jsonl`
2. Generate your own dataset: `python data/create_vqa_pairs.py`
3. Start training: `python training/train_local.py --config configs/train_local.yaml`
