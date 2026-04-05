# XScan-Agent: Innovation Plan & Benchmarking Strategy

## Paper Concept

**Title**: "XScan-Agent: Adaptive X-Ray Reasoning via Proposal-Guided Retrieval-Augmented VLM Fine-Tuning"

**Venue target**: AAAI 2027 Industry Track or WACV 2027 Applications Track

**Positioning**: Industry system paper -- end-to-end agentic pipeline from raw X-ray scan to customs declaration mismatch alerts. Key differentiator vs. prior work (OVXD, RAXO): not just detection accuracy, but an **adaptive verification workflow** where the fine-tuned model's perception directly drives an uncertainty-gated analysis loop.

**Core narrative**: We fine-tune a VLM specifically for the agentic X-ray task -- the model is trained on both full-scene analysis AND focused ROI re-analysis with retrieval context, so the adaptive loop feeds the model exactly the task formats it was trained on. The innovation is the **tight coupling between model training and agent design**.

---

## Core Innovation: PG-RAV with Adaptive Agentic Loop

The key innovation has two tightly coupled parts:

### Part 1: Proposal-Guided Retrieval-Augmented VLM (PG-RAV)

YOLO spatial proposals and retrieval exemplar evidence are **integrated into the VLM's input during training**, not just at inference. The model learns to use structured spatial context and exemplar comparisons as part of its reasoning.

Standard approach (what everyone does):
```
X-ray image -> VLM -> "what's in here?" -> text output
```

Our approach:
```
X-ray image
  + YOLO spatial annotations as structured prompt tokens
  + Retrieved exemplar labels as text context
  -> VLM -> per-region classification with confidence
```

**Why this is better than existing work:**
- **vs. OVXD** (CLIP adaptation): OVXD adapts image features only. We inject spatial structure (YOLO proposals) AND retrieval evidence into the VLM prompt, giving the model explicit "where to look" and "what similar items look like."
- **vs. RAXO** (training-free retrieval): RAXO retrieves at inference only. We fine-tune the VLM WITH retrieval context in the prompt, so it learns to leverage exemplar comparisons.
- **vs. crop-and-classify**: Instead of independently classifying each ROI, we keep the full image context and mark regions with spatial tokens.

### Part 2: Uncertainty-Gated Adaptive Perception Loop

The fine-tuned VLM's generation confidence (measured via token logprobs from vLLM) drives an adaptive re-analysis loop. Low-confidence classifications trigger focused re-analysis: zoom into the ROI, retrieve more specific exemplars, re-query the VLM with a focused prompt.

**Why this is novel**: The adaptive loop is not generic -- it feeds the VLM exactly the task format it was trained on (Phase B trains on both full-scene and focused-ROI prompts). The fine-tuning directly improves the loop's re-analysis quality, creating a virtuous cycle between model training and agent behavior.

**Why logprobs, not trained confidence**: A 2B model cannot reliably learn calibrated meta-cognitive uncertainty (this is an emergent capability at larger scales). Token-level logprobs from vLLM are a principled, training-free uncertainty signal. We validate this with a calibration experiment (E_cal).

---

## System Architecture

```
X-ray Image + Declaration Form
    |
    v
+======================================+
| 1. PERCEPTION LAYER                  |
|                                      |
|  YOLO (class-agnostic) -> proposals  |
|  FAISS retrieval -> exemplar labels  |
+==================+===================+
                   |
                   v
+======================================+
| 2. REASONING CORE (fine-tuned VLM)   |
|                                      |
| Input: full image                    |
|   + YOLO proposals as spatial tokens |
|   + retrieval exemplar labels        |
|                                      |
| Output: per-region classification    |
|   + items YOLO missed                |
|   + token logprobs (from vLLM)       |
+==================+===================+
                   |
                   v
+======================================+
| 3. ADAPTIVE RE-ANALYSIS LOOP         |
|    (uncertainty-gated)               |
|                                      |
|  For each region where               |
|  logprob_entropy > threshold:        |
|    - Zoom into ROI crop              |
|    - Retrieve focused exemplars (k=5)|
|    - Re-query VLM with focused prompt|
|    - Update classification           |
|                                      |
|  Max re-analysis: 3 regions          |
|  Max iterations: 1 pass              |
+==================+===================+
                   |
                   v
+======================================+
| 4. DECLARATION COMPARATOR            |
|    (deterministic, auditable)        |
|                                      |
|  Category mapping (items -> customs) |
|  Set comparison (detected vs declared)|
|  Mismatch flagging with severity     |
+==================+===================+
                   |
                   v
+======================================+
| 5. OUTPUT                            |
|  - Complete item inventory           |
|  - Declaration mismatch report       |
|  - Per-item evidence & confidence    |
|  - ROI crops for flagged items       |
+======================================+
```

**Design decisions:**
- **Declaration comparator is deterministic** (not VLM-driven): In a deployable customs system, decision logic must be auditable and traceable. Set operations on detected vs. declared categories are verifiable; "the VLM thinks there's a mismatch" is not.
- **Single re-analysis pass**: Practical for deployment latency (~2s per image). Multiple iterations would push beyond acceptable customs throughput.
- **VLM as perception engine, orchestrator as controller**: The VLM provides calibrated perception signals; the controller implements the adaptive loop. This separation allows the model contribution (PG-RAV) and the system contribution (adaptive loop) to be independently evaluated.

---

## Adaptive Loop Specification

```python
def adaptive_analysis(image, declaration, yolo, faiss_db, vlm, threshold):
    """
    Uncertainty-gated adaptive perception loop.
    
    The VLM is called twice in different modes it was trained for:
    1. Full-scene analysis (Phase B full-image training format)
    2. Focused ROI re-analysis (Phase B per-ROI training format)
    """
    MAX_REANALYSIS = 3  # max uncertain regions to re-analyze
    
    # --- Pass 1: Full-scene analysis ---
    proposals = yolo.detect(image)  # class-agnostic YOLO
    exemplars = faiss_db.retrieve(proposals, k=3)  # per-region exemplar labels
    
    # Build proposal-guided prompt (matches Phase B training format)
    prompt = build_pgrav_prompt(proposals, exemplars)
    result = vlm.generate(image, prompt, return_logprobs=True)
    
    # Parse per-region classifications with logprob entropy
    regions = parse_regions(result.text)
    for region in regions:
        region.entropy = compute_token_entropy(result.logprobs, region.span)
    
    # --- Identify uncertain regions ---
    uncertain = sorted(
        [r for r in regions if r.entropy > threshold],
        key=lambda r: r.entropy, reverse=True
    )[:MAX_REANALYSIS]
    
    if not uncertain:
        return regions  # high confidence, no re-analysis needed
    
    # --- Pass 2: Focused re-analysis (single pass) ---
    for region in uncertain:
        # Zoom into ROI (matches Phase B per-ROI training format)
        crop = zoom_crop(image, region.bbox, padding=0.15)
        
        # Retrieve more specific exemplars for this region
        focused_exemplars = faiss_db.retrieve_for_crop(crop, k=5)
        
        # Focused prompt with richer retrieval context
        focused_prompt = build_focused_prompt(focused_exemplars)
        refined = vlm.generate(crop, focused_prompt, return_logprobs=True)
        
        # Update only if re-analysis is more confident
        refined_entropy = compute_token_entropy(refined.logprobs, ...)
        if refined_entropy < region.entropy:
            region.update(refined)
            region.reanalyzed = True
    
    return regions
```

**Branching conditions:**
- `logprob_entropy > threshold`: triggers re-analysis (threshold determined by E_cal experiment)
- Re-analysis produces zoomed crop + 5 exemplars (vs. 3 in initial pass)
- Update only if re-analysis has lower entropy (prevents degradation)

**Stopping criteria:**
- Single re-analysis pass (no recursive loops)
- Maximum 3 regions re-analyzed per image
- Total pipeline budget: <5s per image

**Fallback uncertainty signal**: If logprob calibration is poor (ECE > 0.15), fall back to **disagreement signal**: when per-ROI classification differs from full-image pass for the same spatial region, flag as uncertain.

---

## Fine-Tuning Curriculum

### Phase A: X-Ray Domain Adaptation (supervised)
- Standard QLoRA on STCray threat detection (already built)
- The VLM learns to "see" X-ray imagery -- shapes, materials, overlaps
- Loss: standard causal LM on structured JSON answers

### Phase B: Proposal-Guided Classification (the core model innovation)

**Two training formats that match the two inference modes:**

*Format 1: Full-scene analysis with proposals*
- Run class-agnostic YOLO on training images -> collect bbox proposals
- Inject proposals into VLM prompt as spatial tokens:
  ```
  "Objects detected at: [R1: upper-left, 0.85 conf], [R2: center, 0.72 conf].
  Classify each detected region and identify any additional items."
  ```
- VLM trained to output per-region structured classification
- Training data: class-agnostic proposals on HiXray (everyday items) + STCray (threats)

*Format 2: Focused ROI re-analysis*
- Cropped ROI images with focused prompt:
  ```
  "Identify this item in the X-ray crop. Similar items from database:
  laptop (92%), tablet (78%), book (65%). Classify this item."
  ```
- VLM trained to classify individual ROIs with retrieval context
- Training data: ROI crops from same datasets, with FAISS exemplar labels injected

**Why two formats matter for the agent**: The adaptive loop calls the VLM in both modes -- full-scene (Pass 1) and focused ROI (Pass 2). By training on both formats, the model is optimized for exactly the tasks the agent needs.

### Phase B.1: Class-Agnostic YOLO Training

- Collapse ALL bbox annotations from SIXray (1M) + STCray (46K) + HiXray (45K) into single "object" class
- Train YOLOv11 for generic "objectness" in X-ray imagery
- This is a localization-only model -- classification is the VLM's job

### Phase C: Retrieval-Augmented Fine-Tuning (trained, not inference-only)

- For each training sample, retrieve K=3 exemplar crops from FAISS DB
- Inject retrieved exemplar labels as text into VLM prompt:
  ```
  "Reference items: [laptop (92% match), water bottle (78% match), pliers (65% match)].
  Now classify all items in this X-ray scan."
  ```
- Fine-tune with QLoRA so the model learns to leverage retrieval evidence
- **Key difference from current plan**: Phase C is now a training phase, not training-free. The model learns to use retrieval context, not just receive it.
- Tip-Adapter style (training-free logit adjustment) becomes a comparison baseline

**What Phase C changes in the training pipeline:**
- `vqa_dataset.py` collate_fn: extend to include exemplar labels in the text prompt (no multi-image changes needed -- exemplar labels are text, not images)
- Training data: same as Phase B but with retrieval labels prepended to prompts
- Estimated additional engineering: 2-3 days

### Ablation Structure (proves each component matters)

| Config | What's different | Expected result |
|--------|-----------------|-----------------|
| Base VLM (zero-shot) | No fine-tuning, no proposals, no retrieval | Baseline: ~8 categories, high hallucination |
| + Phase A | X-ray domain adaptation | Better X-ray understanding, less hallucination |
| + Phase B (full-scene) | Proposal-guided full-scene classification | Per-region classification, fewer misses |
| + Phase B (focused) | + Focused ROI re-analysis training | Better per-ROI accuracy |
| + Phase C | + Retrieval-as-prompt fine-tuning | Reduced confusion between similar items |
| + Adaptive loop | + Uncertainty-gated re-analysis | Higher recall on ambiguous items |
| Single-pass (no loop) | Full model, but no adaptive re-analysis | Measures adaptive loop contribution |

---

## Core Contributions

| # | Contribution | Novelty |
|---|-------------|---------|
| C1 | PG-RAV: Proposal-guided retrieval-augmented VLM fine-tuning with dual-format training | Novel fusion: YOLO proposals + retrieval exemplars as VLM input during training. Dual-format (full-scene + focused ROI) directly serves the adaptive loop. |
| C2 | Uncertainty-gated adaptive perception loop | Logprob-based confidence drives focused re-analysis of ambiguous regions. The loop feeds the model task formats it was specifically trained on. |
| C3 | End-to-end agentic customs verification pipeline with comprehensive ablation | First VLM-based customs verification system. Complete ablation showing each component's contribution to end-to-end performance. |

---

## Experiments & Ablations

### Core Experiments

| Experiment | Config | Datasets | Key Metric |
|------------|--------|----------|------------|
| E1: Closed-set YOLO baseline | YOLO only, 21 threat classes | STCray test | mAP@0.5 |
| E2: Zero-shot VLM | VLM full-image, no fine-tune | STCray test | Category accuracy |
| E3: PG-RAV (Phase A+B) | Proposal-guided VLM | STCray + HiXray test | Per-ROI accuracy, category coverage |
| E4: + Retrieval (Phase C) | PG-RAV + retrieval-as-prompt | STCray + HiXray test | Accuracy delta from retrieval |
| E5: + Adaptive loop | Full pipeline with re-analysis | STCray + HiXray test | Recall gain on ambiguous items |
| E6: End-to-end declaration | Full pipeline + declaration comparator | Declaration benchmark | Declaration match F1, undeclared recall |
| E7: VLM->YOLO distillation | Distilled YOLO, 50 classes | STCray expanded | Accuracy retention, speed |

### Calibration & Adaptive Loop Experiments

| Experiment | Purpose | Method | Success Criteria |
|------------|---------|--------|-----------------|
| E_cal: Logprob calibration | Validate logprobs as uncertainty signal | Plot entropy vs. accuracy on STCray val, compute ECE | ECE < 0.15, monotonic entropy-accuracy curve |
| E_adapt: Single-pass vs. adaptive | Measure adaptive loop contribution | Compare single-pass vs. loop at fixed precision (0.85) | Recall improvement >= 5% on ambiguous items |
| E_fallback: Disagreement signal | Fallback if logprobs poorly calibrated | Use per-ROI vs. full-image disagreement as uncertainty | Comparable performance to logprob threshold |

### Ablation Study

- **Component contribution**: Base VLM -> +Phase A -> +Phase B -> +Phase C -> +Adaptive loop
- **Dual-format impact**: Phase B full-scene only vs. Phase B full-scene + focused ROI
- **Retrieval training impact**: Phase C trained vs. Phase C training-free (Tip-Adapter baseline)
- **Retrieval DB size**: 1K, 5K, 10K, 45K exemplars
- **Adaptive loop threshold**: Sweep entropy threshold, report precision-recall curves
- **Latency breakdown**: Per-component timing (YOLO, retrieval, VLM pass 1, re-analysis)

### Baselines

| Baseline | Type | What it shows |
|----------|------|---------------|
| YOLO (ours) | Closed-set | What current deployment can do |
| OWL-ViT | RGB-trained OvOD | Domain gap from RGB to X-ray |
| Grounding DINO | RGB-trained OvOD | Domain gap from RGB to X-ray |
| OVXD | X-ray-adapted OvOD | State-of-art X-ray open-vocab |
| RAXO | Training-free X-ray OvOD | Retrieval-based comparison point |

---

## Endpoints Available

| Service | URL | Purpose |
|---------|-----|---------|
| YOLO detection | `POST .../yolo/v1/detect` (multipart file upload) | Fast object proposals with bounding boxes |
| VLM (Qwen3-VL-2B-FP8) | `POST .../qwen3-vl/v1/chat/completions` (OpenAI-compatible, supports logprobs) | Open-vocab item classification |

---

## Implementation Phases

### Phase 1: YOLO-VLM Hybrid Pipeline (DONE)

**Status**: Core pipeline built and tested.

**What exists**:
- `inference/roi_extractor.py` -- YOLO API, local YOLO, and grid-fallback proposal generators
- `inference/hybrid_pipeline.py` -- XScanAgentPipeline class (to be refactored into adaptive loop)
- `scripts/eval_all_items_zero_shot.py` -- Zero-shot evaluation

**Known issues to fix**:
- VLM over-generates items (~100 per image due to repetition in 2B model)
- Need deduplication/limiting in the full-image pass

### Phase 2: Exemplar DB + Retrieval Infrastructure

**Goal**: Build the FAISS vector DB for retrieval-augmented inference and Phase C fine-tuning.

**Steps**:
1. Download HiXray dataset
2. Extract labeled ROI crops from HiXray + STCray
3. Embed crops using CLIP vision encoder (ViT-B/32) -> FAISS index
4. Build retrieval API

**Files to create**:
- `inference/exemplar_db.py` -- Build and query FAISS vector DB
- `scripts/build_exemplar_db.py` -- Offline DB construction

### Phase 3: PG-RAV Fine-Tuning (core model innovation)

**Phase A** (already done): QLoRA on STCray threat detection

**Phase B**: Proposal-guided dual-format classification
1. Train class-agnostic YOLO on SIXray + STCray + HiXray
2. Run YOLO on training images -> collect proposals
3. Create dual-format training data:
   - Full-scene: proposals + exemplar labels in prompt -> per-region classification
   - Focused ROI: crop + exemplar labels in prompt -> single-item classification
4. Fine-tune with QLoRA on dual-format dataset

**Phase C**: Retrieval-augmented fine-tuning
1. For each training sample, retrieve K=3 exemplars from FAISS DB
2. Inject exemplar labels as text into prompt
3. Fine-tune with QLoRA (extends Phase B prompts with retrieval context)

**Files to create**:
- `scripts/prepare_class_agnostic_yolo.py` -- Merge datasets -> single "object" class
- `training/train_yolo_class_agnostic.py` -- Train YOLOv11 class-agnostic
- `data/create_pgrav_vqa.py` -- Generate dual-format VQA training data
- `training/train_vlm_pgrav.py` -- PG-RAV fine-tuning (builds on train_vlm_qlora.py)

### Phase 4: Adaptive Loop Implementation + Calibration

**Goal**: Refactor pipeline into adaptive loop, validate logprob calibration.

**Steps**:
1. Add logprob extraction to VLM calls (vLLM's `logprobs` parameter)
2. Run E_cal: measure entropy-accuracy correlation on STCray validation
3. Establish entropy threshold (or fall back to disagreement signal)
4. Refactor `XScanAgentPipeline.run()` into adaptive loop controller
5. Run E_adapt: single-pass vs. adaptive loop comparison

**Files to modify**:
- `inference/hybrid_pipeline.py` -- Refactor into adaptive loop (see pseudocode above)
- `inference/vllm_engine.py` -- Add logprob extraction support

**Files to create**:
- `evaluation/calibration_analysis.py` -- Logprob calibration experiment
- `evaluation/eval_adaptive_loop.py` -- Single-pass vs. adaptive comparison

### Phase 5: VLM-to-YOLO Distillation + Domain Gap Analysis

**Distillation**:
1. Run PG-RAV on training images -> generate expanded labels (50 categories)
2. Train YOLOv11 on expanded labels
3. Target: >80% of PG-RAV accuracy at 100x speed

**Domain Gap Analysis**:
1. Same 100 images, same prompt -> compare: RGB photo vs X-ray scan
2. Categorize failure modes: overlap confusion, material ambiguity, unfamiliar shapes
3. Quantify: which categories suffer most from domain gap?

### Phase 6: Paper Experiments & Writing

Run full experiment table (E1-E7 + E_cal + E_adapt + E_fallback), ablation study, and baselines.

---

## Datasets Required

| Dataset | Images | Status | Phase | Purpose |
|---------|--------|--------|-------|---------|
| STCray | 46,642 | In repo | All | Primary train/test (threats) |
| HiXray | 45,364 | To download | 2,3 | Everyday items |
| SIXray | 1,059,231 | To download | 3 | Class-agnostic YOLO training (1M images) |
| ORXray | 10,933 | To download | 5 | Cross-dataset evaluation |
| DET-COMPASS | ~10K (370 classes) | To request | 6 | Large-scale open-vocab eval |
| Declaration benchmark | 100 | To create | 4 | All-items ground truth + synthetic declarations |

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Phase 1 | YOLO-VLM hybrid pipeline (DONE) |
| 3 | Phase 2 | Exemplar DB + retrieval infrastructure |
| 4-5 | Phase 3 | PG-RAV dual-format fine-tuning (Phases A-B-C) |
| 6 | Phase 4 | Adaptive loop + calibration validation |
| 7 | Phase 5 | VLM-to-YOLO distillation |
| 8-9 | Phase 6 | Full experiments + ablations |
| 10 | - | Paper writing + figures |
| 11 | - | Internal review + submission |

---

## Current Code Inventory

| File | Status | Purpose |
|------|--------|---------|
| `inference/hybrid_pipeline.py` | Done (to refactor) | XScanAgentPipeline -> adaptive loop controller |
| `inference/roi_extractor.py` | Done | YOLO API / local YOLO / grid-fallback proposals |
| `inference/vllm_engine.py` | Existing (to extend) | vLLM inference (add logprob extraction) |
| `inference/postprocess.py` | Existing | Threat postprocessing |
| `scripts/eval_all_items_zero_shot.py` | Done | Zero-shot evaluation |
| `scripts/visualize_vlm_samples.py` | Done | Data quality visualization |
| `scripts/verify_image_attention.py` | Done | Verification gate for VLM training |
| `training/train_vlm_qlora.py` | Fixed | VLM QLoRA training with SFTTrainer |
| `training/vqa_dataset.py` | Fixed | Multimodal collate_fn with image_grid_thw |
| `inference/exemplar_db.py` | TODO | FAISS vector DB for retrieval |
| `scripts/build_exemplar_db.py` | TODO | Build exemplar DB |
| `data/create_pgrav_vqa.py` | TODO | Generate dual-format VQA training data |
| `training/train_vlm_pgrav.py` | TODO | PG-RAV fine-tuning script |
| `evaluation/calibration_analysis.py` | TODO | Logprob calibration experiment (E_cal) |
| `evaluation/eval_adaptive_loop.py` | TODO | Single-pass vs. adaptive comparison (E_adapt) |
| `evaluation/eval_declaration_matching.py` | TODO | Declaration benchmark runner |
| `scripts/generate_vlm_labels.py` | TODO | Generate VLM labels for YOLO distillation |
| `training/distill_vlm_to_yolo.py` | TODO | YOLO training on VLM-expanded labels |
| `evaluation/analyze_domain_gap.py` | TODO | RGB vs X-ray failure analysis |
