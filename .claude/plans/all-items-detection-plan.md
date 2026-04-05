# All-Items Detection Plan: Zero-Shot VLM for Customs Declaration Verification

**Source spec:** .omc/specs/deep-interview-all-items-detection.md (12% ambiguity, 8-round deep interview)

## Executive Summary

Add a **second inference mode** ("all_items") alongside the existing threat detection mode. The base Qwen3-VL-2B (zero-shot, no fine-tuning) recognizes all visible item categories in X-ray images via a new prompt, outputs structured JSON via XGrammar, maps items to customs declaration categories, and flags mismatches against a passenger's digital declaration form.

---

## RALPLAN-DR Summary

### Principles
1. **Zero-shot first** -- No fine-tuning; leverage the base VLM's general knowledge with a well-designed prompt
2. **Additive, not disruptive** -- The all-items mode runs alongside threat detection; existing threat pipeline is untouched
3. **Schema-driven** -- XGrammar guided generation ensures valid JSON; a new schema file governs the all-items output
4. **Evaluate before optimizing** -- Qualitative check on 20 images before investing in annotation or fine-tuning

### Decision Drivers
1. **No training data exists** for all-items X-ray detection -> zero-shot is the only viable first step
2. **Existing XGrammar infrastructure** can be reused with a new schema -> minimal new code
3. **Declaration comparison already exists** in postprocess.py -> extend, don't rewrite

### Viable Options

**Option A: Dual-Schema Mode Routing (Recommended)**
- Create `output_schema_all_items.json` with open item categories (no enum restriction on name)
- Add `detection_mode` parameter to the API and engine
- Route to the appropriate schema + prompt based on mode
- Extend postprocessing with category mapping and declaration comparison

Pros:
- Clean separation: threat detection is completely untouched
- XGrammar works with both schemas independently
- Easy to iterate on all-items schema without affecting threats
- Minimal changes to existing code (additive)

Cons:
- Two schema files to maintain
- Model loaded once but called with different schemas -> minor complexity in engine

**Option B: Unified Schema with Mode Field**
- Extend the existing schema to support both modes
- Add conditional logic for item name validation

Pros:
- Single schema file

Cons:
- XGrammar enum constraints can't be conditional at runtime
- Mixing threat-specific and general fields in one schema is fragile
- Changes to the unified schema risk breaking threat detection

**Why Option B is not recommended:** XGrammar enforces the schema at generation time. A single schema with conditional enums is not supported by XGrammar -- it validates the full schema on every generation. Separate schemas avoid this fundamental constraint.

---

## Implementation Plan

### Phase 1: New Schema + Prompt

**Step 1.1: Create output_schema_all_items.json**

New schema with open item names (no enum restriction):
```json
{
  "title": "X-ray All Items Detection Output",
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "category": {"type": "string"},
          "description": {"type": "string"},
          "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
          "location": {"type": "string"}
        },
        "required": ["category", "description", "confidence"]
      }
    },
    "total_count": {"type": "integer", "minimum": 0},
    "summary": {"type": "string"}
  },
  "required": ["items", "total_count"]
}
```

Key differences from threat schema:
- `category` (string, unrestricted) replaces `name` (enum of threats)
- `description` added for richer item detail
- `summary` replaces `has_concealed_items` (a general text summary)
- `location` is free-form string (not enum) -- more flexible for general items

**Step 1.2: Design the all-items prompt**

```
List all items and objects visible in this X-ray baggage scan. For each item, provide:
- category: the broad item category (e.g., electronics, clothing, liquids, metal_tools, food, documents, toiletries, accessories)
- description: a brief description of the specific item
- confidence: how confident you are (0.0 to 1.0)
- location: where in the image the item appears

Include ALL visible items, not just prohibited ones. Respond in JSON format only.
```

### Phase 2: Engine + API Mode Routing

**Step 2.1: Update vllm_engine.py**

Add schema selection to `VLLMInferenceEngine`:
- `__init__`: Accept `schema_path` for default + `all_items_schema_path` for all-items mode
- New method `generate_all_items(image_path, prompt=ALL_ITEMS_PROMPT, ...)`:
  - Same as `generate_structured` but uses the all-items schema for `guided_json`
  - Returns dict with `items`, `total_count`, `summary`

**Step 2.2: Update api_server.py**

Add `detection_mode` to `InspectionRequest`:
- `detection_mode: str = "threats"` (default preserves backward compatibility)
- Values: `"threats"` (existing), `"all_items"` (new), `"both"` (run both)
- Route to appropriate engine method based on mode
- New response model `AllItemsResponse` for the all-items mode

**Step 2.3: Update postprocess.py**

Add customs category mapping and declaration comparison:

```python
# Standard customs declaration categories
CUSTOMS_CATEGORIES = {
    "electronics": ["laptop", "phone", "tablet", "camera", "charger", ...],
    "clothing": ["shirt", "pants", "shoes", "jacket", ...],
    "liquids": ["bottle", "water", "perfume", "spray", ...],
    "food": ["snack", "fruit", "canned", ...],
    "documents": ["book", "paper", "passport", ...],
    "toiletries": ["toothbrush", "razor", "shampoo", ...],
    "metal_tools": ["wrench", "pliers", "screwdriver", ...],
    "currency": ["coins", "bills", "cash", ...],
    "alcohol": ["wine", "beer", "spirits", ...],
    "tobacco": ["cigarettes", "cigars", ...],
}
```

New functions:
- `map_to_customs_categories(detected_items: List[Dict]) -> Dict[str, bool]`
  - Maps VLM item descriptions to customs categories using keyword matching
  - Returns `{category: detected_or_not}` dict
- `compare_with_declaration_categories(detected_categories: Dict[str, bool], declared_categories: Dict[str, bool]) -> Dict`
  - Returns `{matched, undeclared (detected but not declared), undetected (declared but not detected)}`
- `process_all_items_response(vlm_answer, declared_categories=None) -> Dict`
  - Main postprocessing for all-items mode

### Phase 3: Evaluation

**Step 3.1: Create qualitative evaluation script**

`scripts/eval_all_items_zero_shot.py`:
- Load 20 random X-ray images from STCray test set
- Run base Qwen3-VL-2B with the all-items prompt
- Display: image path, VLM output, human judgment prompt
- Save results to `test_results/all_items_qualitative/`

**Step 3.2: Verify zero-shot quality**

Run the evaluation:
```bash
python scripts/eval_all_items_zero_shot.py \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --data data/stcray_vlm/stcray_vlm_test.jsonl \
    --n 20 --output test_results/all_items_qualitative/
```

Qualitative pass criteria:
- VLM produces valid JSON for >80% of images
- VLM identifies at least 1 item per image (no empty results)
- Item descriptions are plausible for X-ray imagery
- Diverse categories appear across the 20 images

---

## Files to Modify

| File | Change |
|------|--------|
| `inference/vllm_engine.py` | Add `generate_all_items()` method, dual schema loading |
| `inference/api_server.py` | Add `detection_mode` parameter, route to all-items path |
| `inference/postprocess.py` | Add `CUSTOMS_CATEGORIES`, `map_to_customs_categories()`, `compare_with_declaration_categories()`, `process_all_items_response()` |

## New Files

| File | Purpose |
|------|---------|
| `inference/output_schema_all_items.json` | XGrammar schema for open-vocabulary item detection |
| `scripts/eval_all_items_zero_shot.py` | Qualitative evaluation of zero-shot all-items detection |

## Files NOT Modified

| File | Why |
|------|-----|
| `inference/output_schema.json` | Existing threat schema stays unchanged |
| `training/*` | No fine-tuning needed for zero-shot approach |
| `evaluation/eval_vlm_qlora.py` | Threat evaluation stays separate |
| `cai_integration/*` | No CAI changes for zero-shot inference |

---

## Acceptance Criteria

1. `output_schema_all_items.json` validates as JSON Schema
2. `generate_all_items()` produces valid JSON matching the schema on test images
3. API accepts `detection_mode="all_items"` and returns item categories
4. `map_to_customs_categories()` correctly maps at least 8 out of 10 standard categories
5. `compare_with_declaration_categories()` flags undeclared items correctly
6. Qualitative eval: VLM produces meaningful results on >80% of 20 test images
7. Existing threat detection mode is completely unaffected (regression test)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| VLM can't interpret X-ray imagery zero-shot | The VLM has general image understanding; X-rays are grayscale but items are still recognizable. Qualitative eval will reveal this quickly. |
| XGrammar rejects open-vocabulary schema | The all-items schema uses `"type": "string"` without enum -- XGrammar supports this. |
| Category mapping is inaccurate | Start with keyword matching; iterate to embeddings-based matching if needed. |
| Dual schema complicates the engine | Schema selection is a simple if/else; no shared state between modes. |
| Zero-shot accuracy too low | If qualitative eval fails, next step is synthetic labels via GPT-4o/Gemini. |

---

## ADR: Decision Record

**Decision:** Dual-schema mode routing (Option A) with zero-shot inference

**Drivers:** XGrammar cannot handle conditional enums in a single schema. Existing threat detection must remain untouched. No training data exists for all-items detection.

**Alternatives considered:**
- Unified schema (rejected: XGrammar can't do conditional enums at runtime)
- Fine-tuning first (rejected: no labeled training data; zero-shot is the correct first step)
- Separate model instance (rejected: wasteful; same base model with different prompt/schema)

**Why chosen:** Minimal code changes, clean separation of concerns, leverages existing XGrammar infrastructure, preserves backward compatibility

**Consequences:** Two schema files to maintain. If the all-items mode proves valuable, may want to unify the schemas later (after understanding the full requirements).

**Follow-ups:**
- If zero-shot accuracy >70%: invest in quantitative evaluation (annotate 50 images)
- If zero-shot accuracy <50%: generate synthetic training data via GPT-4o/Gemini
- If category mapping is poor: switch from keywords to embedding-based semantic matching
