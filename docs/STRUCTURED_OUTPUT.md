# Structured JSON Output with XGrammar

This guide explains how the X-ray scanning model generates structured JSON output with guaranteed schema compliance using XGrammar guided generation.

## Overview

The fine-tuned Qwen2.5-VL model can generate outputs in two formats:

1. **Natural Language** - Conversational answers for exploratory questions
2. **Structured JSON** - Machine-parseable format with guaranteed schema compliance (via XGrammar)

### Why Structured Output?

**Benefits:**
- **Guaranteed Valid JSON**: XGrammar ensures 100% valid JSON output at inference time
- **Consistent Format**: All outputs match the predefined schema
- **Confidence Scores**: Each detected item includes a confidence value (0.0-1.0)
- **Precise Locations**: Spatial information for each item (e.g., "upper-left", "center")
- **Easy Integration**: Direct JSON parsing, no regex or text extraction needed
- **Production-Ready**: Reliable output format for downstream systems

---

## Output Schema

All structured outputs follow this JSON schema:

```json
{
  "items": [
    {
      "name": "knife",
      "confidence": 0.95,
      "location": "center"
    }
  ],
  "total_count": 1,
  "has_concealed_items": false
}
```

### Schema Fields

#### `items` (array, required)
List of detected prohibited items.

**Item fields:**
- `name` (string): Item category
  - Valid values: `"knife"`, `"folding knife"`, `"straight knife"`, `"utility knife"`, `"multi-tool knife"`, `"scissors"`, `"gun"`, `"handgun"`, `"pistol"`, `"firearm"`, `"explosive"`, `"blade"`, `"weapon"`, `"prohibited item"`
- `confidence` (number, 0.0-1.0): Detection confidence
  - 0.85-0.95: Clear, unambiguous items
  - 0.70-0.84: Partially visible or occluded items
  - 0.50-0.69: Low confidence detections
- `location` (string): Spatial position in scan
  - Valid values: `"upper-left"`, `"upper"`, `"upper-right"`, `"left"`, `"center"`, `"right"`, `"lower-left"`, `"lower"`, `"lower-right"`, `"center-left"`, `"center-right"`, `"upper-center"`, `"lower-center"`

#### `total_count` (integer, required)
Total number of prohibited items detected. Must match `items` array length.

#### `has_concealed_items` (boolean, required)
Whether any items are partially hidden, concealed, or intentionally obscured.

### Full Schema File

See [`inference/output_schema.json`](../inference/output_schema.json) for the complete JSON Schema definition used by XGrammar.

---

## Training Data Format

The model is trained on a **mixed dataset**:
- **70% Natural Language** questions/answers
- **30% Structured JSON** questions/answers

### Natural Language Examples

```json
{
  "question": "What items can you see in this X-ray scan?",
  "answer": "I can see a folding knife in the center-left area and scissors in the upper-right quadrant.",
  "question_type": "general"
}
```

### Structured JSON Examples

```json
{
  "question": "List all items detected in this X-ray scan in JSON format.",
  "answer": "{\"items\": [{\"name\": \"folding knife\", \"confidence\": 0.92, \"location\": \"center-left\"}, {\"name\": \"scissors\", \"confidence\": 0.88, \"location\": \"upper-right\"}], \"total_count\": 2, \"has_concealed_items\": false}",
  "question_type": "structured_list"
}
```

**Key differences:**
1. Question explicitly requests JSON format
2. Answer is a JSON string (escaped quotes)
3. `question_type` is `"structured_list"`

---

## Generating VQA Dataset with Structured Output

### Step 1: Generate Mixed-Format VQA Data

Use the `--structured-ratio` parameter to control the mix of question types:

```bash
# Generate 2000-sample VQA dataset (70% natural, 30% JSON)
python data/llm_vqa_generator.py \
  --annotations data/stcray/annotations/stcray_train_processed.json \
  --images-dir data/stcray/images \
  --output data/stcray_vqa_2k_mixed.jsonl \
  --model gemini-2.0-flash \
  --samples-per-image 3 \
  --num-samples 2000 \
  --random-seed 42 \
  --structured-ratio 0.3
```

**Parameters:**
- `--num-samples`: Randomly sample N images from dataset (e.g., 2000)
- `--random-seed`: Seed for reproducible sampling (default: 42)
- `--structured-ratio`: Ratio of structured JSON questions (0.0-1.0, default: 0.0)
  - `0.0` = All natural language
  - `0.3` = 30% JSON, 70% natural
  - `1.0` = All structured JSON

### Step 2: Split into Train/Val

```bash
python data/split_vqa_dataset.py \
  --input data/stcray_vqa_2k_mixed.jsonl \
  --output-dir data \
  --output-prefix stcray_vqa_2k \
  --train-ratio 0.8 \
  --val-ratio 0.2 \
  --random-seed 42
```

**Output:**
- `data/stcray_vqa_2k_train.jsonl` (1600 samples, ~1120 natural + ~480 JSON)
- `data/stcray_vqa_2k_val.jsonl` (400 samples, ~280 natural + ~120 JSON)

### Step 3: Train Model

Train on the mixed dataset:

```bash
python training/train_qwen_vl.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --train_file data/stcray_vqa_2k_train.jsonl \
  --eval_file data/stcray_vqa_2k_val.jsonl \
  --output_dir models/qwen2.5-vl-7b-xray-finetuned \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-5 \
  --lora_rank 64
```

The model learns to:
1. Answer natural language questions conversationally
2. Generate valid JSON for structured questions
3. Include confidence scores and locations in JSON output

---

## Inference with XGrammar

### Using vLLM Engine (Recommended)

XGrammar is integrated with vLLM for **guaranteed valid JSON output**:

```python
from inference.vllm_engine import VLLMInferenceEngine

# Initialize engine
engine = VLLMInferenceEngine(
    model_path="models/qwen2.5-vl-7b-xray-finetuned",
    tensor_parallel_size=1,
)

# Generate structured JSON (with XGrammar)
result = engine.generate_structured(
    image_path="test_scan.jpg",
    prompt="List all items detected in this X-ray scan in JSON format.",
    max_tokens=500,
    temperature=0.7,
)

print(result)
# Output: {"items": [...], "total_count": 2, "has_concealed_items": false}
```

**How XGrammar Works:**
1. Loads the JSON schema from `inference/output_schema.json`
2. Constrains the model's output during generation
3. Guarantees the output matches the schema (no parsing errors)

### Testing XGrammar

Verify XGrammar is working correctly:

```bash
python inference/test_xgrammar.py \
  --model models/qwen2.5-vl-7b-xray-finetuned \
  --images data/stcray/images/img001.jpg data/stcray/images/img002.jpg \
  --verbose \
  --save-results xgrammar_test_results.json
```

**Expected output:**
```
✓ SUCCESS: XGrammar guided generation working correctly!
  All outputs are valid JSON (100% success rate)
✓ All outputs match the expected schema
```

### Natural Language Generation

For exploratory questions, use natural language generation:

```python
# Generate natural language (no XGrammar)
answer = engine.generate_natural(
    image_path="test_scan.jpg",
    prompt="Describe the items you see in this scan.",
    max_tokens=200,
    temperature=0.7,
)

print(answer)
# Output: "I can see a folding knife in the center-left area..."
```

### Fallback with Transformers (No XGrammar)

If vLLM is not available, the system falls back to Transformers:

```python
from inference.vllm_engine import TransformersInferenceEngine

# Fallback engine (no XGrammar, may have parsing errors)
engine = TransformersInferenceEngine(
    model_path="models/qwen2.5-vl-7b-xray-finetuned",
    device="cuda",
)

# Attempts JSON but not guaranteed
result = engine.generate_structured(image_path="test_scan.jpg")
```

**Note:** Without XGrammar, JSON parsing may occasionally fail. The system includes fallback text extraction in `inference/json_parser.py`.

---

## API Server Usage

The FastAPI server supports both structured and natural language generation:

### Structured Output (default)

```bash
curl -X POST http://localhost:8080/api/v1/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "scan_id": "scan_001",
    "image_base64": "<base64_encoded_image>",
    "use_structured": true
  }'
```

**Response:**
```json
{
  "scan_id": "scan_001",
  "risk_level": "medium",
  "detected_items": [
    {
      "name": "knife",
      "confidence": 0.92,
      "location": "center-left",
      "occluded": false
    }
  ],
  "item_details": [...],
  "recommended_action": "REVIEW",
  "reasoning": "Detected items: knife. Moderate risk - manual review recommended.",
  "processing_time_ms": 245.3,
  "used_structured_output": true
}
```

### Natural Language Output

```bash
curl -X POST http://localhost:8080/api/v1/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "scan_id": "scan_002",
    "image_base64": "<base64_encoded_image>",
    "use_structured": false,
    "question": "What do you see in this scan?"
  }'
```

---

## Sample Size Recommendations

For experimentation and testing, use random sampling to generate smaller VQA datasets:

| Sample Size | Use Case | Expected Training Time |
|-------------|----------|------------------------|
| **100 samples** | Quick testing, proof-of-concept | ~10 minutes |
| **2,000 samples** | Initial experimentation | ~1-2 hours |
| **10,000 samples** | Production model v1 | ~5-8 hours |
| **30,000+ samples** | Full production model | ~12-24 hours |

### Progressive Workflow

1. **100 samples**: Validate pipeline end-to-end
2. **2k samples**: Test structured output, tune hyperparameters
3. **10k samples**: First production candidate
4. **30k samples**: Final production model

Example:

```bash
# Step 1: Generate 100-sample test dataset
python data/llm_vqa_generator.py \
  --num-samples 100 \
  --structured-ratio 0.3 \
  ... # other args

# Step 2: If results are good, scale to 2k
python data/llm_vqa_generator.py \
  --num-samples 2000 \
  --structured-ratio 0.3 \
  ... # other args

# Step 3: Final model with full dataset
python data/llm_vqa_generator.py \
  --structured-ratio 0.3 \
  ... # no --num-samples = use all data
```

---

## Validation and Quality Assurance

### 1. Validate VQA Dataset

Check quality of generated VQA pairs:

```bash
python data/llm_vqa_generator.py \
  --validate \
  --output data/stcray_vqa_2k_mixed.jsonl
```

### 2. Test Structured Output

After training, verify XGrammar:

```bash
python inference/test_xgrammar.py \
  --model models/qwen2.5-vl-7b-xray-finetuned \
  --images data/stcray/test_images/*.jpg
```

**Success criteria:**
- ✓ 100% valid JSON outputs (with vLLM)
- ✓ 100% schema-compliant outputs
- ✓ Confidence scores in expected ranges
- ✓ Location values match schema enums

### 3. Manual Inspection

Review sample outputs:

```bash
python inference/vllm_engine.py \
  models/qwen2.5-vl-7b-xray-finetuned \
  data/stcray/images/img001.jpg
```

---

## Troubleshooting

### Issue: JSON parsing errors despite XGrammar

**Cause:** vLLM not installed or configured correctly

**Solution:**
```bash
# Ensure vLLM is installed
pip install vllm>=0.5.0

# Verify XGrammar support
python -c "from vllm import SamplingParams; print(hasattr(SamplingParams, 'guided_json'))"
# Should print: True
```

### Issue: Low confidence scores

**Cause:** Insufficient structured training data

**Solution:**
- Increase `--structured-ratio` to 0.5 or higher
- Generate more training samples
- Use clearer ground truth labels

### Issue: Incorrect item names

**Cause:** Schema enum doesn't match training data categories

**Solution:**
- Update `inference/output_schema.json` to include actual category names from STCray
- Regenerate VQA data with correct category mapping

### Issue: Schema validation errors

**Cause:** Model not fully learning the schema

**Solution:**
- Increase training epochs (3-5)
- Increase structured_ratio to 0.4-0.5
- Add explicit JSON formatting instruction in training prompts

---

## Best Practices

1. **Mixed Training Data**: Always use 20-40% structured JSON questions (0.2-0.4 ratio)
2. **XGrammar for Production**: Use vLLM + XGrammar for guaranteed valid JSON
3. **Confidence Calibration**: Review confidence scores on validation set, adjust thresholds
4. **Schema Versioning**: Keep `output_schema.json` in version control
5. **Fallback Strategy**: Implement text extraction fallback for edge cases
6. **Testing**: Run `test_xgrammar.py` after every model update
7. **Sample Incrementally**: Start small (100-2k), scale up progressively

---

## Related Documentation

- [VQA Generation Guide](VQA_GENERATION.md)
- [Gemini Local VQA](GEMINI_VQA_GENERATION.md)
- [Training Guide](../README.md#training)
- [API Server Documentation](../inference/api_server.py)

---

## Summary

| Feature | Natural Language | Structured JSON (XGrammar) |
|---------|------------------|----------------------------|
| **Output Format** | Text | Valid JSON |
| **Guaranteed Valid** | No | Yes (with vLLM) |
| **Confidence Scores** | Manual extraction | Direct field |
| **Location Info** | Text parsing | Direct field |
| **Use Case** | Exploratory Q&A | Production systems |
| **Training Data** | 70% of dataset | 30% of dataset |
| **Inference Engine** | Any | vLLM recommended |

For production deployments, **always use structured JSON output with vLLM and XGrammar** to ensure 100% reliable parsing and integration with downstream systems.
