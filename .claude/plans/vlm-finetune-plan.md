# VLM Fine-Tune Plan: Qwen3-VL-2B QLoRA on STCray

**Revision:** v2 (incorporates Architect + Critic feedback from iteration 1)

## Executive Summary

Fine-tune Qwen3-VL-2B-Instruct using QLoRA for multi-object X-ray threat detection on the STCray dataset (46,642 images, 21 categories). The codebase has substantial infrastructure already built, but **the training data pipeline has 4 critical bugs that prevent actual multimodal training**. This plan prioritizes fixing those bugs with a rigorous verification gate, then executing the training pipeline end-to-end.

---

## RALPLAN-DR Summary

### Principles
1. **Fix before build** -- The existing multimodal training pipeline is broken; fix it before adding features
2. **Verify before committing** -- Prove images are actually attended to before launching a 36-hour training run
3. **Validate incrementally** -- Run a small-scale smoke test before committing to full training
4. **Reproducibility** -- Every training run must be fully reproducible (fixed seeds, logged hyperparameters, saved configs)
5. **Minimal viable first** -- Get one working end-to-end run before optimizing hyperparameters

### Decision Drivers
1. **Training correctness** -- The collate_fn doesn't pass images through the processor correctly; `image_grid_thw` is never provided; the model currently trains on text-only inputs
2. **Time/cost efficiency** -- Full training is 36-48 hours on T4 GPU; we need verification + smoke test to avoid wasting a 2-day run on broken code
3. **Evaluation fidelity** -- The eval script uses proper multimodal messages but training doesn't, creating a train/eval mismatch

### Viable Options

**Option A: Fix-and-Run with SFTTrainer (Recommended)**
- Fix the 4 critical bugs in vqa_dataset.py collate_fn
- Adopt SFTTrainer + DataCollatorForCompletionOnlyLM for label masking (already imported, never used)
- Add verification gate proving images are attended to
- Add a validation split
- Run smoke test (50 train + 10 val samples, 1 epoch)
- Execute full training on CAI
- Evaluate and iterate

Pros:
- Fastest path to a working fine-tuned model
- SFTTrainer handles label masking correctly via response template detection (eliminates hand-rolled Bug 2 fix)
- Leverages existing infrastructure (CAI pipeline, eval script, inference engine)
- Minimal new code; mostly fixing existing code

Cons:
- Doesn't explore alternative training frameworks (e.g., LLaMA-Factory, Axolotl)
- Single hyperparameter configuration initially
- We own the training code maintenance

**Option B: Rewrite with LLaMA-Factory**
- Replace custom training code with LLaMA-Factory framework
- Use their proven Qwen-VL training pipeline
- Configure via YAML

Pros:
- Battle-tested training code for Qwen VL models
- Built-in hyperparameter sweep, DPO, RLHF support
- Active community and documentation

Cons:
- Requires significant rework of CAI integration (job configs, environment setup, argument passing)
- Adds external dependency with its own release cadence
- Existing eval/inference code may need adaptation to LLaMA-Factory's output format
- Learning curve for new framework

**Why Option B is not recommended:** The existing codebase has the right architecture with specific, identifiable bugs. The Architect confirmed these are fixable without a rewrite. LLaMA-Factory would require reworking CAI job configs, adapting eval scripts, and learning a new framework. The core value of Option A is that `SFTTrainer` (already imported in the training script) solves the label masking problem, and the remaining fixes are in the collate_fn -- a single function rewrite, not a framework migration.

---

## Critical Bugs to Fix

### Bug 1: collate_fn doesn't create multimodal inputs (vqa_dataset.py:98-194)

**Current behavior:** The collate_fn builds chat messages with text-only content, tokenizes with `processor.tokenizer()`, and processes images separately with `processor.image_processor()`. For Qwen3-VL, this means images are never integrated into the token stream -- the model trains on text only.

**Required fix:** Use the unified `processor()` call with multimodal messages that include `{"type": "image"}` content blocks, matching how the eval script (eval_vlm_qlora.py:361-379) correctly does it.

### Bug 2: Label masking doesn't account for chat template tokens (vqa_dataset.py:172-185)

**Current behavior:** Masks the first N tokens where N = len(tokenize(prompt)). But the chat template adds role markers, system tokens, etc., so the actual prompt boundary is wrong.

**Required fix:** Adopt `SFTTrainer` with `DataCollatorForCompletionOnlyLM` (both already imported at train_vlm_qlora.py:32 but unused). This uses a response template marker (e.g., `"assistant\n"`) to detect the boundary automatically, eliminating hand-rolled label masking entirely.

**Design decision:** We choose SFTTrainer over hand-rolled masking because:
- It is already imported in the training script (line 32), indicating the original developer intended to use it
- It handles chat template token boundaries correctly by design
- It is maintained by the TRL library team and tested against Qwen models
- It eliminates an entire class of masking bugs

### Bug 3: Training messages missing image content (vqa_dataset.py:121-131)

**Current behavior:** Messages are `{"role": "user", "content": prompt}` (string only).

**Required fix:** Messages should be `{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}` to include the image token. This is the root cause of Bug 1 -- without the image content block, `apply_chat_template` never inserts `<|vision_start|>...<|vision_end|>` placeholder tokens.

### Bug 4: `image_grid_thw` never passed to model (NEW -- identified by Architect)

**Current behavior:** The collate_fn (vqa_dataset.py:188-194) returns only `input_ids`, `attention_mask`, `pixel_values`, and `labels`. Grep for `image_grid_thw` across the entire codebase returns zero hits.

**Why this is critical:** Qwen3-VL's `forward()` requires `image_grid_thw` (a tensor of shape `(num_images, 3)` encoding temporal/height/width grid dimensions) for its rotary position embedding over image patches. Without it, even if `pixel_values` are correctly generated, the model cannot compute spatial position encodings for image patches.

**Required fix:** The unified `processor()` call returns `image_grid_thw` as part of its output. The collate_fn must include this key in its return dict. HuggingFace Trainer passes all keys from the collate output as `**kwargs` to `model.forward()` (since `remove_unused_columns=False` is already set in TrainingArguments).

---

## Implementation Plan

### Phase 1: Fix Training Pipeline

**Step 1.1: Fix collate_fn in vqa_dataset.py**
- Rewrite collate_fn to construct multimodal messages with `{"type": "image"}` content blocks
- Use unified `processor()` call with both `text=` and `images=` arguments
- Ensure return dict includes: `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `labels`
- Handle batching correctly (Qwen3-VL concatenates pixel_values along a special dimension; `image_grid_thw` tracks per-image dimensions)
- Remove the hand-rolled label masking code (will be handled by SFTTrainer)

**Step 1.2: Switch to SFTTrainer in train_vlm_qlora.py**
- Replace `Trainer` (line 378) with `SFTTrainer` (already imported at line 32)
- Configure `DataCollatorForCompletionOnlyLM` with response template marker (e.g., the assistant role token)
- Add early stopping: `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `early_stopping_patience=3`
- Remove unused imports if any remain after refactoring

**Step 1.3: Verification gate (MUST PASS before any training)**

This gate proves images are actually being attended to. Run these checks on the fixed collate_fn:

```bash
python -c "
from training.vqa_dataset import collate_fn, XrayVQADataset
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct', trust_remote_code=True)
# ... load 2 samples, call collate_fn ...
batch = collate_fn(samples, processor, 2048, True)

# Check 1: image_grid_thw is present
assert 'image_grid_thw' in batch, 'FAIL: image_grid_thw missing from collate output'

# Check 2: input_ids contain vision placeholder tokens
image_pad_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
assert (batch['input_ids'] == image_pad_id).any(), 'FAIL: no vision tokens in input_ids'

# Check 3: image_grid_thw has expected shape (num_images, 3)
assert batch['image_grid_thw'].dim() == 2 and batch['image_grid_thw'].shape[1] == 3, \
    f'FAIL: unexpected image_grid_thw shape: {batch[\"image_grid_thw\"].shape}'

print('ALL CHECKS PASSED: images are correctly integrated into the token stream')
"
```

Additionally, after the smoke test model is trained (Step 1.5), verify image-dependence:
```bash
# Feed same prompt with 2 different images, assert outputs differ
python scripts/verify_image_attention.py --model checkpoints/smoke-test/final --images img1.jpg img2.jpg
```

**Step 1.4: Add train/val split**
- Modify convert_stcray_to_vlm.py to add `--val-ratio 0.1` flag
- 90/10 stratified split of training data -> train (33,584) / val (3,732)
- Stratified by category to maintain distribution
- Output: `stcray_vlm_train.jsonl`, `stcray_vlm_val.jsonl`, `stcray_vlm_test.jsonl`

**Step 1.5: Data quality visualization**

Before training, visually verify annotations:
```bash
python scripts/visualize_vlm_samples.py \
  --data data/stcray_vlm/stcray_vlm_train.jsonl \
  --n 20 --output data_check/
```
- Renders 20 random images with ground-truth bounding box overlays
- Manual inspection: do boxes match visible objects? Are categories correct?
- Must pass visual inspection before proceeding

**Step 1.6: Smoke test**
- Create a tiny test: 50 train samples, 10 val samples, 1 epoch, 10 steps
- Run: `python training/train_vlm_qlora.py --train-data data/smoke_train.jsonl --eval-data data/smoke_val.jsonl --num-train-epochs 1 --output-dir checkpoints/smoke-test`
- **Pass criteria:**
  - Training loss at step 10 is < 80% of loss at step 1 (verified from TensorBoard logs or stdout)
  - Checkpoint saves successfully to `checkpoints/smoke-test/`
  - `image_grid_thw` appears in training batch (logged by collate_fn)
- Run eval: `python evaluation/eval_vlm_qlora.py --finetuned-model checkpoints/smoke-test/final --test-data data/smoke_val.jsonl --skip-base`
- Run image-dependence verification (Step 1.3 second check)

### Phase 2: Execute Training on CAI

**Step 2.1: Prepare CAI environment**
- Update jobs_config_vlm.yaml with any changed CLI args (SFTTrainer params, val split)
- Ensure Git LFS data (STCray RAR files) will be available
- Test environment setup script with updated requirements

**Step 2.2: Run full training pipeline**
- Job 1: Download STCray dataset (30 min)
- Job 2: Convert to VLM format + val split (15 min)
- Job 3: Train Qwen3-VL-2B QLoRA, 3 epochs on 33,584 images (~36-45 hours)
  - Batch size 2, gradient accumulation 4 (effective batch 8)
  - Learning rate 2e-4 with 100-step warmup
  - LoRA r=16, alpha=32, dropout=0.05
  - Targets: q_proj, k_proj, v_proj, o_proj
  - Checkpoints every 500 steps, eval every 500 steps
  - `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`
  - `early_stopping_patience=3` (stops if eval_loss doesn't improve for 3 consecutive evals)
- Job 4: Evaluate base vs fine-tuned (2 hours)

**Step 2.3: Monitor training**
- Watch TensorBoard logs: training loss should decrease steadily through epoch 1
- **Alert threshold:** If train loss at step 500 is > 90% of step 1 loss, stop and investigate
- Check GPU utilization stays >80% via monitoring dashboard
- **Overfitting signal:** If eval_loss increases for 2 consecutive eval points while train_loss decreases, early stopping will trigger at patience=3

### Phase 3: Evaluate and Iterate

**Step 3.1: Run comprehensive evaluation**
- Base vs fine-tuned comparison (precision, recall, F1, IoU)
- Per-category breakdown (21 threat categories)
- Multi-object detection accuracy (images with 3+ objects)
- JSON parsing success rate
- Inference latency comparison

**Step 3.2: Analyze results and decide next steps**
- If F1 >= 0.50: Strong success, proceed to inference optimization
- If F1 0.35-0.50: Good first run, iterate on hyperparameters (try LoRA r=32, lr=1e-4, more epochs, add gate/up/down_proj targets)
- If F1 < 0.35: Investigate -- check per-category breakdown for systematic failures, verify data quality on failing categories, consider adding vision encoder to LoRA targets

**Step 3.3: Package for deployment**
- Merge LoRA adapters into base model (optional, for faster inference)
- Export for vLLM inference engine
- Update inference/vllm_engine.py with model path
- Test end-to-end: image in -> structured JSON out

---

## Files to Modify

| File | Change | Details |
|------|--------|---------|
| `training/vqa_dataset.py` | Rewrite collate_fn | Multimodal messages, unified processor() call, include `image_grid_thw` in output, remove hand-rolled label masking |
| `training/train_vlm_qlora.py` | Switch to SFTTrainer | Replace Trainer with SFTTrainer, configure DataCollatorForCompletionOnlyLM, add early stopping params (`load_best_model_at_end`, `metric_for_best_model`, `early_stopping_patience`) |
| `data/convert_stcray_to_vlm.py` | Add val split | Add `--val-ratio` flag, stratified split output |
| `cai_integration/jobs_config_vlm.yaml` | Update args | Reflect new CLI args from SFTTrainer and val split |

## New Files to Create

| File | Purpose |
|------|---------|
| `scripts/verify_image_attention.py` | Verification gate: feeds 2 images with same prompt, asserts different outputs |
| `scripts/visualize_vlm_samples.py` | Data quality: renders N samples with bounding box overlays |

## Files NOT Modified (already correct)

| File | Why |
|------|-----|
| `evaluation/eval_vlm_qlora.py` | Already uses correct multimodal message format with `{"type": "image"}` blocks |
| `inference/vllm_engine.py` | Inference pipeline is independent of training fixes |
| `inference/postprocess.py` | No changes needed |

---

## Acceptance Criteria

### Phase 1 Gate (must pass before CAI training)
1. **Verification gate passes:** collate_fn output contains `image_grid_thw`, `input_ids` contain `<|image_pad|>` token IDs, `image_grid_thw` shape is `(N, 3)`
2. **Data visualization passes:** 20 random samples inspected, bounding boxes align with visible objects
3. **Smoke test passes:** Training loss at step 10 is < 80% of loss at step 1; checkpoint saves successfully
4. **Image-dependence verified:** Same prompt with 2 different images produces different model outputs

### Phase 2 Gate (training success)
5. Full CAI training completes without OOM or crashes
6. Eval loss shows improvement over training (lower than base model eval loss)

### Phase 3 Gate (model quality -- initial run)
7. Fine-tuned model achieves F1 >= 0.35 on STCray test set (base is ~0.21) -- this is the minimum "training is working" threshold
8. **Stretch goal:** F1 >= 0.50 (achievable with hyperparameter iteration)
9. JSON parsing success rate > 80% (base is ~45%)
10. Multi-object detection accuracy > 50% on images with 3+ objects

---

## Risk Mitigation

| Risk | Mitigation | Specific Action |
|------|-----------|-----------------|
| T4 OOM during training | QLoRA + gradient checkpointing keeps VRAM ~6-8GB | If OOM: reduce batch_size to 1, reduce max_seq_length to 1024 |
| Training loss doesn't decrease | Images not actually attended to | Run verification gate (Step 1.3); if gate passes but loss flat, reduce lr to 1e-4 |
| Images ignored despite loss decrease | Model learns text patterns only | Step 1.3 image-dependence check catches this; re-verify after smoke test |
| Poor generalization (overfit) | Train/val gap widens | Early stopping (patience=3) triggers automatically; additionally monitor: if train_loss < 0.5 * eval_loss, investigate |
| STCray data quality issues | Bad annotations corrupt training | Run `scripts/visualize_vlm_samples.py --n 20` and manually inspect before training |
| CAI job timeout | Training takes longer than 48h | Checkpoint resume enabled; increase timeout to 72h if needed |
| Class imbalance in 21 categories | Rare categories underperform | Check per-category F1 in eval; consider class-weighted loss or oversampling in iteration 2 |

---

## ADR: Decision Record

**Decision:** Fix existing custom training pipeline (Option A) with SFTTrainer adoption, rather than rewriting with LLaMA-Factory (Option B)

**Drivers:** The existing codebase has the right architecture with 4 specific, identifiable bugs. SFTTrainer (already imported) solves the label masking problem. The remaining fixes are in the collate_fn -- a single function rewrite. The CAI integration, eval pipeline, and inference engine are all built around the current code structure.

**Alternatives considered:**
- LLaMA-Factory (rejected: requires reworking CAI job configs, adapting eval/inference scripts, new dependency with its own release cadence; the core bugs are fixable without a framework switch)
- Axolotl (rejected: less mature Qwen-VL support compared to direct HuggingFace/TRL usage)
- Full fine-tune without LoRA (rejected: T4 GPU 16GB VRAM cannot fit Qwen3-VL-2B in FP16 for training; QLoRA is necessary)
- Hand-rolled label masking (rejected in favor of SFTTrainer: SFTTrainer handles chat template boundaries correctly by design, is maintained by TRL team, and is already imported in the codebase)

**Why chosen:** Minimal changes, fastest path to working model, preserves existing infrastructure investment, leverages already-imported SFTTrainer for robust label masking

**Consequences:** We own the training code and must maintain it. If Qwen releases a new VL model version, we update our code rather than a framework config. SFTTrainer dependency pins us to TRL library releases.

**Follow-ups:**
- After first successful training run, evaluate whether hyperparameter sweep justifies adopting a framework for that specific capability
- If F1 < 0.35 after first run, reconsider LLaMA-Factory as it may have optimizations we're missing
- Monitor TRL library for breaking changes in SFTTrainer API
