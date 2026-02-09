# Architecture Update Summary

## What Changed?

The system has been redesigned to **separate item recognition (VLM) from declaration comparison (post-processing)**.

## Before vs After

### Before ❌
```
┌─────────────────────────────────────┐
│         VLM (Qwen2.5-VL)           │
│                                     │
│  • Recognize items                  │
│  • Compare with declaration         │
│  • Assess risk                      │
│  • Generate explanation             │
└─────────────────────────────────────┘
```

**Problems:**
- VLM tries to do too much
- Mixed responsibilities
- Hard to update business rules
- Need retraining for policy changes

### After ✅
```
┌─────────────────────┐
│  VLM (Qwen2.5-VL)  │  ← Focus: Item recognition ONLY
│  • Recognize items  │
│  • Detect occlusion │
└──────────┬──────────┘
           ↓
┌──────────────────────┐
│  Post-Processing     │  ← Declaration comparison
│  • Compare detected  │
│    vs declared items │
│  • Assess risk       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Business Rules      │  ← Action decision
│  • Determine action  │
│  • Generate alert    │
└──────────────────────┘
```

**Benefits:**
- ✅ VLM learns one task well
- ✅ Update rules without retraining
- ✅ Transparent decisions
- ✅ Easy to test and maintain

## Key File Changes

### 1. Data Pipeline

**`data/create_vqa_pairs.py`**

Before:
```python
# Generated declaration comparison questions
"Is the declared content 'clothing, electronics' consistent with the X-ray image?"
"No, undeclared items found: folding knife. Fraud detected."
```

After:
```python
# Focuses on item recognition only
"What items are visible in this X-ray scan?"
"Detected items: a folding knife at center-left, partially concealed."
```

### 2. Post-Processing (NEW)

**`inference/postprocess.py`** - New file!

```python
def process_vlm_response(vlm_answer, declared_items):
    # Step 1: Extract items from VLM answer
    detected_items = extract_items_from_text(vlm_answer)
    
    # Step 2: Compare with declaration
    comparison = compare_with_declaration(detected_items, declared_items)
    
    # Step 3: Assess risk
    risk_level = assess_risk_level(detected_items, comparison)
    
    # Step 4: Determine action
    action = determine_action(risk_level)
    
    return {
        "detected_items": detected_items,
        "risk_level": risk_level,
        "recommended_action": action,
        "reasoning": "..."
    }
```

### 3. API Server

**`inference/api_server.py`**

Before:
```python
# VLM did everything
result = vllm_server.inspect_xray(image, question, declaration)
```

After:
```python
# Step 1: VLM - item recognition
vlm_answer = vllm_server.generate("What items are visible?")

# Step 2: Post-processing - declaration comparison
result = process_vlm_response(vlm_answer, declared_items)
```

## Training Data Format

### Before ❌
```jsonl
{
  "question": "Is the declared content 'clothing' consistent with scan?",
  "answer": "No, undeclared knife detected. Fraud alert.",
  "metadata": {
    "declared_items": ["clothing"],
    "match_declaration": false
  }
}
```

### After ✅
```jsonl
{
  "question": "What items are visible in this X-ray scan?",
  "answer": "Detected items: a folding knife at center-left.",
  "metadata": {
    "categories": ["Folding_Knife"],
    "has_occlusion": true
  }
}
```

**Key difference:** No declaration info in training data!

## Question Types (After)

1. **General**: "What items are visible in this X-ray scan?"
2. **Specific**: "Is there a knife in this scan?"
3. **Location**: "Where are the items located?"
4. **Occlusion**: "Are any items concealed?"
5. **Detailed**: "Provide detailed description of all items"

**Removed:** Declaration comparison questions

## API Flow

### Request
```json
{
  "scan_id": "SCAN-001",
  "image_base64": "...",
  "declared_items": ["clothing", "electronics"]
}
```

### Processing Steps

**Step 1: VLM Inference**
```
Input: "What items are visible?"
Output: "Detected items: a folding knife at center-left."
```

**Step 2: Post-Processing**
```python
detected = ["folding knife"]
declared = ["clothing", "electronics"]

# Compare
undeclared = ["folding knife"]  # Not in declaration!

# Assess risk
risk_level = "high"  # Undeclared prohibited item

# Action
action = "PHYSICAL_INSPECTION"
```

### Response
```json
{
  "risk_level": "high",
  "detected_items": [{"item": "folding knife", "occluded": true}],
  "declaration_match": false,
  "reasoning": "Detected items: folding knife. ALERT: Undeclared items...",
  "recommended_action": "PHYSICAL_INSPECTION"
}
```

## Risk Assessment Logic

**Handled in post-processing (NOT in VLM):**

```python
def assess_risk_level(detected_items, declaration_comparison, has_occlusion):
    if not detected_items:
        return "low"
    
    # HIGH risk conditions
    if len(detected_items) > 2:
        return "high"  # Multiple prohibited items
    
    if has_occlusion:
        return "high"  # Concealed items
    
    if declaration_comparison["undeclared_items"]:
        return "high"  # Fraud indicator
    
    # MEDIUM risk
    if any("knife" in item for item in detected_items):
        return "medium"  # Single prohibited item
    
    return "low"
```

**Easy to update without retraining!**

## Migration Steps

1. **Regenerate training data**
   ```bash
   python data/create_vqa_pairs.py --opixray-root data/opixray --split all
   ```

2. **Train with new format**
   ```bash
   python training/train_local.py --config configs/train_local.yaml
   ```

3. **Deploy with post-processing**
   ```bash
   python inference/api_server.py --model outputs/qwen25vl_lora_phase1
   ```

## Performance Impact

| Component | Latency | Change |
|-----------|---------|--------|
| VLM Inference | 300-400ms | ~Same (simpler task may be faster) |
| Post-Processing | <10ms | +New (minimal overhead) |
| **Total** | **~350ms** | **✅ Still <500ms target** |

## Testing

### Test VLM (item recognition)
```bash
python training/train_local.py --config configs/train_local.yaml
python evaluation/eval_vqa.py --model outputs/qwen25vl_lora_phase1
```

### Test Post-Processing
```bash
python inference/postprocess.py  # Run built-in tests
```

### Test End-to-End
```bash
python inference/api_server.py --model outputs/qwen25vl_lora_phase1
curl -X POST http://localhost:8080/api/v1/inspect -d '{...}'
```

## Documentation

- **`ARCHITECTURE.md`** - Full system design (NEW)
- **`CHANGELOG.md`** - Version history and migration guide (NEW)
- **`README.md`** - Updated quick start
- **`QUICKSTART.md`** - Updated for Cloudera workspace

## Summary

**The VLM now does ONE thing well: recognize items in X-rays**

Everything else (declaration comparison, risk assessment, action decisions) is handled in fast, flexible post-processing.

This makes the system:
- ✅ More accurate (focused VLM task)
- ✅ More flexible (update rules without retraining)
- ✅ More transparent (clear AI vs rules separation)
- ✅ Easier to maintain (modular components)

**No API changes, better architecture!** 🎉
