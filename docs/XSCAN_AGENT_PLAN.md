# XScan-Agent: Innovation Plan & Implementation Strategy

**Title**: "XScan-Agent: Adaptive X-Ray Reasoning via Proposal-Guided VLM Fine-Tuning"

**Venue target**: AAAI 2027 Industry Track or WACV 2027 Applications Track

**Deliverables**: Publishable paper with full ablation + external baselines, AND a working end-to-end demo (X-ray + declaration -> mismatch report with evidence crops)

---

## Core Innovation

Two tightly coupled contributions:

### C1: Proposal-Guided Dual-Format VLM Fine-Tuning (Phases A+B)

YOLO spatial proposals are integrated into the VLM's input during training. The model learns two complementary task formats:
- **Full-scene analysis**: classify all regions in a complete X-ray scan with spatial proposal context
- **Focused ROI re-analysis**: classify a single cropped region with category disambiguation hints

```
Standard approach:  X-ray image -> VLM -> "what's in here?" -> text
Our approach:       X-ray image + YOLO proposals as spatial tokens + category hints
                    -> VLM -> per-region classification with confidence
```

**vs. OVXD** (CLIP adaptation): OVXD adapts image features only. We inject spatial structure into the VLM prompt.
**vs. RAXO** (training-free retrieval): RAXO retrieves at inference only. We fine-tune the VLM with proposal context, so it learns to leverage spatial grounding.
**vs. crop-and-classify**: We keep full image context and mark regions with spatial tokens.

### C2: Uncertainty-Gated Adaptive Perception Loop (Orchestrator + Phase C)

The VLM's generation confidence (token logprobs from vLLM) drives an adaptive re-analysis loop. Low-confidence classifications trigger focused re-analysis with category disambiguation hints (Phase C retrieval-as-text).

**Why logprobs, not trained confidence**: A 2B model cannot reliably learn calibrated meta-cognitive uncertainty. Token-level logprobs are a principled, training-free signal. Validated via calibration experiment (E_cal).

### C3: End-to-End Agentic Customs Verification Pipeline

First VLM-based customs verification system. Complete ablation showing each component's contribution.

---

## System Architecture

```
X-ray Image + Declaration Form
    |
    v
+======================================+
| 1. PERCEPTION LAYER                  |
|  Class-agnostic YOLO -> proposals    |
+==================+===================+
                   |
                   v
+======================================+
| 2. REASONING CORE (fine-tuned VLM)   |
|  Input: full image                   |
|    + YOLO proposals as spatial tokens |
|    + category hints (Phase C text)   |
|  Output: per-region classification   |
|    + token logprobs (from vLLM)      |
+==================+===================+
                   |
                   v
+======================================+
| 3. ADAPTIVE RE-ANALYSIS LOOP         |
|  For regions where entropy > thresh: |
|    - Zoom into ROI crop              |
|    - Lookup confusable categories    |
|    - Re-query VLM (focused format)   |
|    - Update if more confident        |
|  Max re-analysis: 3 regions          |
+==================+===================+
                   |
                   v
+======================================+
| 4. DECLARATION COMPARATOR            |
|  Category mapping (items -> customs) |
|  Set comparison (detected vs declared)|
|  Mismatch flagging with severity     |
+==================+===================+
                   |
                   v
+======================================+
| 5. OUTPUT                            |
|  - Item inventory + customs cats     |
|  - Declaration mismatch report       |
|  - Per-item evidence & confidence    |
|  - ROI crops for flagged items       |
+======================================+
```

**Design decisions:**
- **Declaration comparator is deterministic** (not VLM-driven): auditable set operations, not "the VLM thinks there's a mismatch"
- **Single re-analysis pass**: practical for deployment latency (~2-5s per image)
- **VLM as perception engine, orchestrator as controller**: cleanly separates model contribution (C1) from system contribution (C2) for independent evaluation

---

## Fine-Tuning Curriculum

### Phase A: X-Ray Domain Adaptation -- DONE

- QLoRA fine-tuning on STCray (46K images, 21 threat categories)
- Qwen3-VL-2B-Instruct, 4-bit NF4 quantization, LoRA r=16
- Checkpoint: `/home/cdsw/checkpoints/qwen3vl-2b-xray-qlora/final`
- Merged model: `/home/cdsw/models/qwen3vl-2b-xray-merged`
- The VLM learns to "see" X-ray imagery: shapes, materials, overlapping objects

### Phase B: Proposal-Guided Dual-Format Classification -- TODO

**Prerequisite: Class-Agnostic YOLO**
- Collapse ALL bbox annotations from STCray (46K) + HiXray (45K) into single "object" class
- Train YOLOv11-nano on ~90K images (~12h on T4)
- Produces a localization-only model -- classification is the VLM's job
- Script: `scripts/prepare_class_agnostic_yolo.py` (TODO)

**Dual-format VLM training:**

*Format 1: Full-scene analysis with proposals*
```
"Objects detected at: [R1: upper-left, 0.85 conf], [R2: center, 0.72 conf].
Classify each detected region and identify any additional items."
```
- VLM outputs per-region `{category, confidence}` as structured JSON
- Training data: class-agnostic proposals on HiXray (everyday items) + STCray (threats)

*Format 2: Focused ROI re-analysis with category hints*
```
"Identify this item in the X-ray crop.
Similar categories: laptop (large rectangular, dense), tablet (thin rectangular, moderate density).
Classify this item."
```
- VLM classifies individual ROIs with disambiguation context
- Training data: ROI crops with ground-truth confusable category hints from annotations

**Why dual format matters**: The adaptive loop calls the VLM in both modes -- full-scene (Pass 1) and focused ROI (Pass 2). Training on both formats means the model is optimized for exactly the tasks the agent needs.

**Training estimate**: ~135K samples (90K x 1.5 formats), 2-3 days on T4 for 3 epochs
**Fallback**: If epoch 1 takes >36h, subsample HiXray to 50% or reduce to 2 epochs with early stopping

**Scripts to create:**
- `scripts/prepare_class_agnostic_yolo.py` -- merge datasets -> single "object" class
- `data/create_pgrav_vqa.py` -- generate dual-format VQA training data
- CAI job config update for Phase B training pipeline

### Phase C: Retrieval-as-Text Category Hints -- INFERENCE ONLY

Phase C is NOT a training phase. It augments the VLM prompt at inference time with category disambiguation hints from a static lookup table.

**Category hints lookup schema** (`data/category_hints.json`):
```json
{
  "laptop": {
    "description": "Large rectangular electronic device, high density",
    "visual_cues": ["rectangular shape", "hinge visible", "dense material"],
    "confusables": ["tablet", "book", "cutting_board"]
  },
  "tablet": {
    "description": "Thin rectangular electronic device, moderate density",
    "visual_cues": ["thin profile", "uniform density", "no hinge"],
    "confusables": ["laptop", "book", "phone"]
  }
}
```

- ~50 entries covering HiXray (8 categories) + STCray (21 categories) + common everyday items
- Source: HiXray + STCray annotation labels, manually curated confusable pairs
- At inference: orchestrator looks up confusable categories for initial classification, injects as text into the focused ROI re-analysis prompt
- **No FAISS, no CLIP embeddings, no neural retrieval** -- just a Python dict lookup
- Script: `scripts/build_category_hints.py` (TODO, ~1 day of manual curation)

**Why this works**: Phase B Format 2 already trains the model on prompts with category hints. Phase C at inference simply provides real category hints into a format the model was specifically trained on.

---

## Adaptive Loop Specification

```python
def adaptive_analysis(image, declaration, yolo, category_hints, vlm, threshold):
    MAX_REANALYSIS = 3

    # --- Pass 1: Full-scene analysis (Phase B Format 1) ---
    proposals = yolo.detect(image)  # class-agnostic
    prompt = build_proposal_prompt(proposals)
    result = vlm.generate(image, prompt, return_logprobs=True)

    regions = parse_regions(result.text)
    for region in regions:
        region.entropy = compute_token_entropy(result.logprobs, region.span)

    # --- Identify uncertain regions ---
    uncertain = sorted(
        [r for r in regions if r.entropy > threshold],
        key=lambda r: r.entropy, reverse=True
    )[:MAX_REANALYSIS]

    if not uncertain:
        return regions

    # --- Pass 2: Focused re-analysis (Phase B Format 2 + Phase C hints) ---
    for region in uncertain:
        crop = zoom_crop(image, region.bbox, padding=0.15)
        hints = category_hints.confusables(region.category, k=5)  # static dict lookup
        focused_prompt = build_focused_prompt(hints)
        refined = vlm.generate(crop, focused_prompt, return_logprobs=True)

        refined_entropy = compute_token_entropy(refined.logprobs, ...)
        if refined_entropy < region.entropy:
            region.update(refined)
            region.reanalyzed = True

    return regions
```

**Stopping criteria**: Single pass, max 3 regions, <5s total budget
**Fallback**: If logprob calibration is poor (ECE > 0.15), use disagreement signal (per-ROI vs full-image classification mismatch)

---

## Implementation Phases

### Phase 1: YOLO-VLM Hybrid Pipeline -- DONE

| File | Status |
|------|--------|
| `inference/hybrid_pipeline.py` | Done (to refactor into adaptive loop) |
| `inference/roi_extractor.py` | Done (YOLO proposal generators) |
| `scripts/eval_all_items_zero_shot.py` | Done |

### Phase 2: Phase B Training

**Goal**: Train class-agnostic YOLO + proposal-guided dual-format VLM

**Steps**:
1. Download HiXray dataset (45K images, everyday items)
2. Create `scripts/prepare_class_agnostic_yolo.py` -- collapse STCray+HiXray bboxes to "object"
3. Train YOLOv11-nano class-agnostic (~12h T4)
4. Run class-agnostic YOLO on training images -> collect proposals
5. Create `data/create_pgrav_vqa.py` -- generate dual-format VQA data
6. Fine-tune VLM with QLoRA on dual-format data (~2-3 days T4)
7. Evaluate: compare Phase A vs Phase B on HiXray+STCray test sets

**Files to create:**
- `scripts/prepare_class_agnostic_yolo.py`
- `data/create_pgrav_vqa.py`
- `cai_integration/jobs_config_vlm_phase_b.yaml`

### Phase 3: Adaptive Loop + Logprob Extraction

**Goal**: Refactor pipeline into adaptive loop, validate logprob calibration

**Prerequisite: Logprob extraction** (must be implemented BEFORE adaptive loop)
- Add `logprobs` parameter to VLM API calls in `inference/hybrid_pipeline.py`
- The vLLM OpenAI-compatible API supports `logprobs: N` in request body
- Parse logprob values from response `choices[0].logprobs`
- This is a code change to existing files, not a new model

**Steps**:
1. Implement logprob extraction in `inference/hybrid_pipeline.py`
2. Build `data/category_hints.json` (~50 entries, Phase C static lookup)
3. Run E_cal: measure entropy-accuracy correlation on STCray validation
4. Establish entropy threshold (or fall back to disagreement signal)
5. Refactor `XScanAgentPipeline.run()` into adaptive loop controller
6. Run E_adapt: single-pass vs. adaptive loop comparison

**Files to modify:**
- `inference/hybrid_pipeline.py` -- adaptive loop controller + logprob extraction

**Files to create:**
- `data/category_hints.json` -- static category lookup
- `scripts/build_category_hints.py` -- generate lookup from annotations
- `evaluation/calibration_analysis.py` -- E_cal experiment
- `evaluation/eval_adaptive_loop.py` -- E_adapt experiment

### Phase 4: Experiments + Paper

**Goal**: Run full experiment suite, write paper

**Experiments:**

| Experiment | Config | Key Metric |
|------------|--------|------------|
| E1: YOLO baseline | YOLO only, 21 threat classes | mAP@0.5 |
| E2: Zero-shot VLM | No fine-tuning | Category accuracy |
| E3: + Phase A | X-ray domain adaptation | Accuracy delta |
| E4: + Phase B | Proposal-guided dual-format | Per-ROI accuracy, coverage |
| E5: + Phase C hints | + retrieval-as-text at inference | Confusion reduction |
| E6: + Adaptive loop | + uncertainty-gated re-analysis | Recall on ambiguous items |
| E7: End-to-end | Full pipeline + declaration | Declaration F1, undeclared recall |
| E_cal | Logprob calibration | ECE < 0.15 |
| E_adapt | Single-pass vs loop | Recall gain >= 5% |

**Minimum ablation delta for publishability**: >= 2% recall improvement OR >= 1% accuracy improvement per phase. If any phase shows <1% improvement, investigate before including in paper.

**Baselines:**

| Baseline | Approach | Time-box |
|----------|----------|----------|
| YOLO (ours) | Already trained | 0 days |
| OWL-ViT | Zero-shot on STCray test | 1 day |
| Grounding DINO | Zero-shot on STCray test | 1 day |
| OVXD | Reproduce if public repo works, else cite published | 2 days max |
| RAXO | Reproduce if public repo works, else cite published | 2 days max |

**Baseline rule**: If OVXD/RAXO code doesn't run on our test set within 2 days, cite their published numbers and note "numbers from original paper, different evaluation split."

**Ablation study:**
- Component: Base VLM -> +Phase A -> +Phase B -> +Phase C -> +Adaptive loop
- Dual-format: Phase B full-scene only vs. full-scene + focused ROI
- Phase C impact: with vs. without category hints at inference
- Loop threshold: sweep entropy threshold, report precision-recall curves
- Latency breakdown: per-component timing

---

## Datasets

| Dataset | Images | Status | Used in | Purpose |
|---------|--------|--------|---------|---------|
| STCray | 46,642 | Available | All phases | Threats (21 categories) |
| HiXray | 45,364 | To download | Phase B, class-agnostic YOLO | Everyday items (8 categories) |
| Declaration benchmark | 100 | To create | Phase 4 (E7) | Synthetic declarations for demo |

**Not needed** (removed from scope): SIXray (1M), ORXray (11K), DET-COMPASS (10K)

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup | Download HiXray, prepare class-agnostic YOLO data |
| 2 | Phase B.1 | Train class-agnostic YOLO (~12h) |
| 2-3 | Phase B.2 | Generate dual-format VQA data, train VLM (~2-3 days) |
| 4 | Phase 3 | Logprob extraction, category hints, adaptive loop |
| 5 | Phase 3 | Calibration experiment, loop validation |
| 6 | Phase 4 | Run ablation experiments (E1-E7) |
| 7 | Phase 4 | Run baselines (OWL-ViT, DINO, OVXD/RAXO) |
| 8 | Paper | Write paper + create demo visualizations |
| 9 | Paper | Internal review + submission |

---

## Current Code Inventory

| File | Status | Purpose |
|------|--------|---------|
| `training/train_vlm_qlora.py` | Done | VLM QLoRA training (Phase A, reuse for Phase B) |
| `training/vqa_dataset.py` | Done | Multimodal collate_fn with image_grid_thw + mm_token_type_ids |
| `inference/hybrid_pipeline.py` | Done (refactor) | Sequential pipeline -> adaptive loop controller |
| `inference/roi_extractor.py` | Done | YOLO proposal generators |
| `evaluation/eval_vlm_qlora.py` | Done | Base vs fine-tuned comparison |
| `data/convert_stcray_to_vlm.py` | Done | STCray -> VQA JSONL |
| `scripts/merge_lora_adapter.py` | Done | Merge LoRA for deployment |
| `cai_integration/jobs_config_vlm.yaml` | Done | Phase A CAI pipeline |
| `scripts/prepare_class_agnostic_yolo.py` | TODO | Merge datasets -> "object" class |
| `data/create_pgrav_vqa.py` | TODO | Dual-format VQA training data |
| `data/category_hints.json` | TODO | Static category lookup (~50 entries) |
| `scripts/build_category_hints.py` | TODO | Generate lookup from annotations |
| `evaluation/calibration_analysis.py` | TODO | Logprob calibration (E_cal) |
| `evaluation/eval_adaptive_loop.py` | TODO | Single-pass vs loop (E_adapt) |
| `evaluation/eval_declaration_matching.py` | TODO | Declaration benchmark (E7) |

---

## Non-Goals (Deferred to Future Work)

- VLM-to-YOLO distillation (transfer PG-RAV knowledge into fast YOLO)
- FAISS vector DB with CLIP embeddings for neural retrieval
- Multi-image VLM fine-tuning
- Training the VLM to output declaration reasoning or re-analysis flags
- Production deployment for real customs officers
- Training on SIXray (1M images) or DET-COMPASS (370 classes)
