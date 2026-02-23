# CAI Integration for X-ray VQA Fine-tuning

This directory contains Cloudera AI Workspace (CAI) integration for automated X-ray VQA model fine-tuning.

## Overview

The CAI integration provides a complete pipeline for fine-tuning Qwen2.5-VL on X-ray baggage scanning data:

1. **Environment Setup** - Create reusable Python virtual environment
2. **Dataset Download** - Download STCray dataset from HuggingFace
3. **VQA Generation** - Generate VQA pairs using external Qwen2.5-VL vLLM server
4. **Model Fine-tuning** - Fine-tune Qwen2.5-VL-7B with LoRA (multi-GPU, checkpointing)

## Prerequisites

### CAI Workspace Requirements
- Python 3.10+
- 2x NVIDIA GPUs (24GB VRAM each) for fine-tuning
- 100GB+ storage
- Runtime: `ml-runtime-pbj-jupyterlab-python3.10-cuda:2025.09.1-b5`

### External Services
- **vLLM Server** with Qwen2.5-VL-7B-Instruct
  - Must be accessible from CAI workspace
  - OpenAI-compatible API endpoint
  - Example: `http://your-vllm-server:8000/v1`

### API Credentials
```bash
export CML_HOST="https://your-cai-workspace.cloudera.com"
export CML_API_KEY="your-api-key"
```

Get API key from CAI: User Settings > API Keys

## Directory Structure

```
cai_integration/
├── README.md                      # This file
├── jobs_config.yaml               # Job definitions
├── create_jobs.py                 # Create jobs via CAI API
├── trigger_jobs.py                # Trigger and monitor jobs
├── setup_environment.py           # Environment setup job (Python wrapper)
├── setup_environment.sh           # Environment setup logic (bash)
├── download_dataset.py            # Dataset download job
├── generate_vqa.py                # VQA generation job
├── finetune_model.py              # Fine-tuning job
└── config/
    └── cai_train_config.yaml      # Training config for CAI
```

## Quick Start

### Step 1: Configure vLLM Endpoint

Edit `jobs_config.yaml` and update the vLLM API endpoint:

```yaml
jobs:
  generate_vqa:
    environment:
      VLLM_API_BASE: "http://your-vllm-server:8000/v1"  # CHANGE THIS
      MODEL_NAME: "Qwen/Qwen2.5-VL-7B-Instruct"
      SAMPLES_PER_IMAGE: "3"
      API_KEY: ""  # Optional: for OpenAI/Claude/authenticated vLLM
```

### Step 2: Create Jobs

```bash
# Set CAI credentials
export CML_HOST="https://your-workspace.cloudera.com"
export CML_API_KEY="your-api-key"
export CML_PROJECT_ID="your-project-id"

# Create all jobs
python3 cai_integration/create_jobs.py --project-id $CML_PROJECT_ID
```

This creates 4 jobs with automatic dependencies:
- `setup_environment` (root)
- `download_dataset` (depends on setup_environment)
- `generate_vqa` (depends on download_dataset)
- `finetune_model` (depends on generate_vqa)

### Step 3: Trigger Pipeline

```bash
# Trigger root job (child jobs auto-trigger when parent succeeds)
python3 cai_integration/trigger_jobs.py --project-id $CML_PROJECT_ID
```

This triggers `setup_environment`, and the rest of the pipeline auto-triggers via CAI dependencies.

### Step 4: Monitor Progress

Monitor jobs in CAI UI:
1. Navigate to **Jobs** > **Job Runs**
2. Watch the pipeline execution:
   - setup_environment (1h)
   - download_dataset (1h)
   - generate_vqa (3-5h)
   - finetune_model (6-12h)

**Total pipeline time**: 11-19 hours

## Job Details

### Job 1: Setup Python Environment

**Duration**: 30-60 minutes (first run), <1 minute (subsequent runs)  
**Resources**: 4 CPU, 16GB RAM  

Creates `/home/cdsw/.venv` with all dependencies (PyTorch, Transformers, PEFT, etc.).

**Skip logic**: If `.venv` exists with required packages, setup is skipped (fast).

**Force reinstall**:
```bash
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job setup_environment \
  --env FORCE_REINSTALL=true
```

### Job 2: Download STCray Dataset

**Duration**: 30-60 minutes  
**Resources**: 4 CPU, 8GB RAM  
**Output**: `/home/cdsw/data/stcray/` (train + test splits)

Downloads ~46k X-ray images with annotations from HuggingFace.

### Job 3: Generate VQA Dataset

**Duration**: 3-5 hours  
**Resources**: 8 CPU, 16GB RAM (no GPU, uses external vLLM)  
**Output**: 
- `/home/cdsw/data/stcray_vqa_train.jsonl` (~90k pairs)
- `/home/cdsw/data/stcray_vqa_val.jsonl` (~48k pairs)

Connects to external Qwen2.5-VL vLLM server to generate VQA pairs.

**Checkpointing**: Saves progress every 100 images, auto-resumes if interrupted.

**Configuration**:
- `VLLM_API_BASE`: vLLM server endpoint (REQUIRED)
- `MODEL_NAME`: Model name (default: Qwen/Qwen2.5-VL-7B-Instruct)
- `SAMPLES_PER_IMAGE`: VQA pairs per image (default: 3)
- `API_KEY`: API key for OpenAI/Claude/authenticated vLLM (optional)

### Job 4: Fine-tune Model

**Duration**: 6-12 hours  
**Resources**: 16 CPU, 64GB RAM, 2x GPUs  
**Output**: `/home/cdsw/outputs/qwen25vl_stcray_lora/`

Fine-tunes Qwen2.5-VL-7B with LoRA on generated VQA dataset.

**Checkpointing**: Saves checkpoint every 500 steps.

**Resume from checkpoint**:
```bash
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job finetune_model \
  --env RESUME_FROM_CHECKPOINT=/home/cdsw/outputs/qwen25vl_stcray_lora/checkpoint-2000
```

## Advanced Usage

### Trigger Specific Job

```bash
# Trigger only VQA generation (assumes dataset already downloaded)
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job generate_vqa
```

### Override Environment Variables

```bash
# Use different vLLM endpoint
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job generate_vqa \
  --env VLLM_API_BASE=http://new-server:8000/v1 \
  --env SAMPLES_PER_IMAGE=5
```

### Resume Failed Fine-tuning

If fine-tuning job times out or fails:

```bash
# 1. Find latest checkpoint in CAI workspace
# In CAI Session, run: ls -d /home/cdsw/outputs/qwen25vl_stcray_lora/checkpoint-*

# 2. Resume from checkpoint
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job finetune_model \
  --env RESUME_FROM_CHECKPOINT=/home/cdsw/outputs/qwen25vl_stcray_lora/checkpoint-2500
```

## Configuration Files

### `jobs_config.yaml`

Defines all jobs with dependencies, resources, and environment variables.

**Key fields**:
- `script`: Python entry point (relative to project root)
- `parent_job_key`: Dependency (null for root job)
- `cpu`, `memory`, `gpu`: Resource allocation
- `timeout`: Max execution time (seconds)
- `runtime_identifier`: CAI runtime image
- `environment`: Environment variables

### `config/cai_train_config.yaml`

Training configuration with CAI-specific absolute paths:
- `train_file: /home/cdsw/data/stcray_vqa_train.jsonl`
- `eval_file: /home/cdsw/data/stcray_vqa_val.jsonl`
- `output_dir: /home/cdsw/outputs/qwen25vl_stcray_lora`

## Troubleshooting

### Issue: Job creation fails

**Check**:
1. CML_HOST and CML_API_KEY are set correctly
2. Project ID is correct
3. You have permissions to create jobs

```bash
# Verify credentials
curl -H "Authorization: Bearer $CML_API_KEY" \
  "$CML_HOST/api/v2/projects/$CML_PROJECT_ID"
```

### Issue: Environment setup times out

**Solution**: Increase timeout in `jobs_config.yaml`:
```yaml
setup_environment:
  timeout: 5400  # 1.5 hours instead of 1 hour
```

### Issue: VQA generation fails - vLLM not accessible

**Check**:
1. vLLM server is running and accessible
2. VLLM_API_BASE is correct in `jobs_config.yaml`
3. Network connectivity from CAI to vLLM server

**Test connectivity from CAI Session**:
```bash
curl http://your-vllm-server:8000/v1/models
```

### Issue: Fine-tuning job fails - Out of memory

**Solution 1**: Reduce batch size in `config/cai_train_config.yaml`:
```yaml
per_device_train_batch_size: 1  # Instead of 2
gradient_accumulation_steps: 16  # Instead of 8
```

**Solution 2**: Request more GPUs in `jobs_config.yaml`:
```yaml
finetune_model:
  gpu: 4  # Instead of 2
```

### Issue: Fine-tuning job times out

**Solution**: Resume from checkpoint:
```bash
# Find checkpoint
ls -d /home/cdsw/outputs/qwen25vl_stcray_lora/checkpoint-*

# Resume
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job finetune_model \
  --env RESUME_FROM_CHECKPOINT=/home/cdsw/outputs/qwen25vl_stcray_lora/checkpoint-XXXX
```

## File Locations (CAI Workspace)

All files are stored under `/home/cdsw/`:

```
/home/cdsw/
├── .venv/                                    # Virtual environment (reused)
├── data/
│   ├── stcray/                               # Downloaded dataset
│   │   ├── train/
│   │   │   ├── images/                       # ~30k images
│   │   │   └── annotations.json
│   │   └── test/
│   │       ├── images/                       # ~16k images
│   │       └── annotations.json
│   ├── stcray_vqa_train.jsonl                # Generated VQA training set
│   └── stcray_vqa_val.jsonl                  # Generated VQA validation set
└── outputs/
    └── qwen25vl_stcray_lora/                 # Fine-tuned model
        ├── adapter_config.json
        ├── adapter_model.bin
        ├── checkpoint-500/
        ├── checkpoint-1000/
        └── checkpoint-1500/
```

## Pushing Model to HuggingFace

After fine-tuning completes, push the model to HuggingFace for persistence:

```bash
# In CAI Session
source /home/cdsw/.venv/bin/activate

# Login to HuggingFace
huggingface-cli login

# Upload model
huggingface-cli upload \
  your-org/qwen25vl-xray-stcray-finetuned \
  /home/cdsw/outputs/qwen25vl_stcray_lora \
  --repo-type model
```

## Cost Estimation

| Job | Duration | GPU Hours | Notes |
|-----|----------|-----------|-------|
| setup_environment | 1h | 0 | CPU only, first run only |
| download_dataset | 1h | 0 | CPU only |
| generate_vqa | 3-5h | 0 | CPU only, uses external vLLM |
| finetune_model | 6-12h | 12-24 | 2x GPUs |
| **Total** | **11-19h** | **12-24** | GPU hours for fine-tuning only |

**Cost (AWS p3.2xlarge equivalent)**: ~$36-72 per full run

## Next Steps After Fine-tuning

1. **Evaluate Model** (in CAI Session):
   ```bash
   source /home/cdsw/.venv/bin/activate
   python evaluation/eval_vqa.py \
     --model outputs/qwen25vl_stcray_lora \
     --test-file data/stcray_vqa_val.jsonl
   ```

2. **Deploy Inference** (in CAI Application):
   ```bash
   python inference/api_server.py \
     --model outputs/qwen25vl_stcray_lora \
     --host 127.0.0.1 \
     --port 8100
   ```

3. **Push to HuggingFace** (for persistence):
   ```bash
   huggingface-cli upload \
     your-org/qwen25vl-xray-finetuned \
     outputs/qwen25vl_stcray_lora
   ```

## Environment Variables Reference

### Job Configuration (`jobs_config.yaml`)

| Variable | Job | Default | Description |
|----------|-----|---------|-------------|
| `FORCE_REINSTALL` | setup_environment | `false` | Force full venv reinstall |
| `VLLM_API_BASE` | generate_vqa | (none) | vLLM server endpoint (REQUIRED) |
| `MODEL_NAME` | generate_vqa | `Qwen/Qwen2.5-VL-7B-Instruct` | Model name for VQA |
| `SAMPLES_PER_IMAGE` | generate_vqa | `3` | VQA pairs per image |
| `API_KEY` | generate_vqa | (empty) | API key for OpenAI/Claude/authenticated vLLM |
| `RESUME_FROM_CHECKPOINT` | finetune_model | (empty) | Checkpoint path to resume |
| `CONFIG_FILE` | finetune_model | `cai_integration/config/cai_train_config.yaml` | Training config |

### Runtime Overrides

Pass via `trigger_jobs.py --env`:

```bash
# Example: Use different vLLM server
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job generate_vqa \
  --env VLLM_API_BASE=http://new-server:8000/v1
```

## Job Dependencies

```
setup_environment (root)
    ↓
download_dataset
    ↓
generate_vqa
    ↓
finetune_model
```

When you trigger `setup_environment`, CAI automatically:
1. Runs setup_environment
2. On success, triggers download_dataset
3. On success, triggers generate_vqa
4. On success, triggers finetune_model

**You only need to trigger the root job!**

## Best Practices

1. **Test with small dataset first**:
   - Modify `download_dataset.py` to use `--max-samples 100`
   - Verify pipeline works end-to-end
   - Then run full pipeline

2. **Monitor GPU usage**:
   - Check CAI job logs for GPU utilization
   - Adjust batch size if OOM errors occur

3. **Backup checkpoints**:
   - Push checkpoints to HuggingFace periodically
   - In case of CAI workspace issues

4. **Reuse virtual environment**:
   - Once `setup_environment` succeeds, it's reused
   - Saves 30-60 minutes per subsequent pipeline run

5. **External vLLM server**:
   - Keep vLLM server running during VQA generation
   - Monitor vLLM logs for errors
   - Test connectivity before triggering VQA job

## Development Workflow

### Iterate on VQA Generation

```bash
# 1. Test with small sample
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job generate_vqa \
  --env MAX_IMAGES=100  # Test mode

# 2. Review output in CAI Session
source /home/cdsw/.venv/bin/activate
python data/llm_vqa_generator.py \
  --output data/stcray_vqa_train.jsonl \
  --validate

# 3. If good, run full generation
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job generate_vqa
```

### Iterate on Training Hyperparameters

```bash
# 1. Edit config/cai_train_config.yaml
# 2. Re-trigger fine-tuning job
python3 cai_integration/trigger_jobs.py \
  --project-id $CML_PROJECT_ID \
  --job finetune_model
```

## Comparison: CAI vs Local

| Aspect | CAI Integration | Local Scripts |
|--------|----------------|---------------|
| Setup | Automated jobs | Manual execution |
| Monitoring | CAI UI | Terminal logs |
| Resumption | Built-in (checkpoints) | Manual |
| Scheduling | CAI dependencies | Manual sequence |
| Resources | CAI-managed | Local GPU |
| Best for | Production, automation | Development, testing |

## Support

For issues:
1. Check CAI job logs: Jobs > Job Runs > [job] > Logs
2. Verify vLLM connectivity: `curl $VLLM_API_BASE/models`
3. Check file paths: All paths should be absolute (`/home/cdsw/...`)
4. Review this README and troubleshooting section

## References

- Main project: [`../README.md`](../README.md)
- Architecture: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- Qwen vLLM guide: [`../docs/QWEN_VL_VLLM_GUIDE.md`](../docs/QWEN_VL_VLLM_GUIDE.md)
- Complete workflow: [`../docs/COMPLETE_WORKFLOW.md`](../docs/COMPLETE_WORKFLOW.md)
