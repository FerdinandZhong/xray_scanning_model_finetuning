# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating deployment of the X-ray VQA fine-tuning project to Cloudera AI Workspace.

## Workflows

### Deploy to CAI ([deploy-to-cai.yml](workflows/deploy-to-cai.yml))

**Purpose:** Automated deployment of X-ray VQA fine-tuning project to Cloudera AI instances.

**Trigger:** Manual workflow dispatch

**Parameters:**
- `vllm_endpoint`: External vLLM server endpoint (e.g., `http://server:8000/v1`)
  - **Optional but recommended**: Jobs will be created even if not provided
  - If not provided: Placeholder used, must configure in CAI before running
  - If not provided: Automatic triggering will be skipped
- `force_reinstall`: Force environment reinstall (default: false)
- `trigger_jobs`: Automatically trigger pipeline after setup (default: false)
  - Only works if vLLM endpoint is provided
- `samples_per_image`: VQA samples per image (default: 3)
- `api_key`: API key for OpenAI/Claude/authenticated vLLM (optional)

**Steps:**
1. **Validate**: Check configuration files and YAML syntax (warns if vLLM endpoint missing)
2. **Setup Project**: Create/find CAI project and sync from GitHub
3. **Create Jobs**: Always creates 4 CAI jobs (uses placeholder if vLLM endpoint missing)
4. **Trigger Pipeline**: Only triggers if vLLM endpoint provided AND trigger_jobs=true
5. **Skip Trigger** (conditional): Shows manual setup instructions if trigger requested but endpoint missing

**Usage:**
```bash
# Via GitHub UI
1. Go to Actions tab
2. Select "Deploy X-ray VQA to CAI"
3. Click "Run workflow"
4. Fill in vLLM endpoint and other options
5. Click "Run workflow"

# Via GitHub CLI (with vLLM endpoint)
gh workflow run deploy-to-cai.yml \
  -f vllm_endpoint=http://your-server:8000/v1 \
  -f force_reinstall=false \
  -f trigger_jobs=false \
  -f samples_per_image=3 \
  -f api_key=""  # Optional: use for authenticated APIs

# Or without vLLM endpoint (configure later in CAI)
gh workflow run deploy-to-cai.yml \
  -f vllm_endpoint="" \
  -f trigger_jobs=false
```

**Required Secrets:**
- `CML_HOST`: CAI instance URL (e.g., https://ml.example.cloudera.site)
- `CML_API_KEY`: CAI API key with project creation permissions
- `GH_PAT`: GitHub Personal Access Token (for private repos, optional)

## Setup Instructions

### 1. Configure Repository Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

**For CAI Deployment:**
```
CML_HOST=https://ml.example.cloudera.site
CML_API_KEY=your-cml-api-key
GH_PAT=your-github-token (optional, for private repos)
```

**Get CAI API Key:**
1. Log into your CAI workspace
2. Go to User Settings → API Keys
3. Click "Create API Key"
4. Copy the generated key
5. Add as `CML_API_KEY` secret in GitHub

**Get GitHub PAT (for private repos):**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select `repo` scope
4. Copy the token
5. Add as `GH_PAT` secret in GitHub

### 2. Enable GitHub Actions

1. Go to repository Settings → Actions → General
2. Enable "Allow all actions and reusable workflows"
3. Set "Workflow permissions" to "Read and write permissions"

### 3. Test the Workflow

```bash
# Run via GitHub Actions UI or CLI
gh workflow run deploy-to-cai.yml \
  -f vllm_endpoint=http://test-server:8000/v1 \
  -f trigger_jobs=false
```

## Detailed Workflow Execution

### Deploy Workflow Execution Model

**Trigger:** Manual via GitHub UI or CLI

**Parameters:**
```
vllm_endpoint: string (required)
force_reinstall: boolean (false)
trigger_jobs: boolean (false)
samples_per_image: string ('3')
```

**Job Execution:** Sequential chain with dependencies
```
Manual Trigger (GitHub Actions UI/CLI)
    ↓
1. validate (5 min)
   - Check configuration files exist
   - Validate YAML syntax
    ↓
2. setup-project (30 min)
   - Creates CAI project "xray-scanning-model"
   - Outputs: project_id
    ↓
3. create-jobs (10 min)
   - Depends on: setup-project
   - Updates jobs_config.yaml with vLLM endpoint
   - Creates 4 CAI jobs via API
    ↓
4. trigger-pipeline (5 min, optional)
   - Depends on: setup-project, create-jobs
   - Only runs if: trigger_jobs == true
   - Triggers root job via CAI API
    ↓
Total time: ~45 minutes (setup only)
            ~50 minutes (setup + trigger)
```

**Output Data Flow:**
```yaml
setup-project:
  outputs:
    project_id: ${{ steps.setup.outputs.project_id }}

create-jobs:
  needs: setup-project
  steps:
    - uses: project_id from setup-project
      run: create_jobs.py --project-id ${{ needs.setup-project.outputs.project_id }}

trigger-pipeline:
  needs: [setup-project, create-jobs]
  if: github.event.inputs.trigger_jobs == 'true'
  steps:
    - run: trigger_jobs.py --project-id ${{ needs.setup-project.outputs.project_id }}
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  GitHub Repository                       │
└────────────┬────────────────────────────────────────────┘
             │
             └─► Manual: Deploy to CAI (Workflow Dispatch)
                 ↓
                 Deploy Workflow (deploy-to-cai.yml) - MANUAL
                 ├─► Validate (5 min)
                 │   ↓
                 ├─► Setup Project (30 min)
                 │   ↓ (outputs: project_id)
                 ├─► Create Jobs (10 min)
                 │   ↓ (sequential)
                 └─► Trigger Pipeline (5 min, optional)
                 ↓
                 ✅ CAI Project Ready
                    ↓
                    Manual: Trigger in CAI UI
                    ↓
                    Fine-tuning Pipeline (11-19 hours)
```

## CAI Deployment Flow

The deploy-to-cai.yml workflow orchestrates the following on your CAI instance:

```
1. Setup Project
   ├─► Create/find CAI project "xray-scanning-model"
   ├─► Configure git sync from GitHub
   └─► Wait for git clone to complete

2. Create Jobs
   ├─► Job 1: setup_environment (install deps)
   ├─► Job 2: download_dataset (get STCray)
   ├─► Job 3: generate_vqa (create VQA pairs)
   └─► Job 4: finetune_model (train model)

3. Execute Jobs (Manual or Auto)
   If trigger_jobs == true:
      setup_environment
         ↓
      download_dataset (auto-triggered by CAI)
         ↓
      generate_vqa (auto-triggered by CAI)
         ↓
      finetune_model (auto-triggered by CAI)
         ↓
      ✅ Fine-tuned model ready

4. Monitor
   └─► Check CAI UI → Jobs → Job Runs
```

## What Happens in CAI

### Phase 1: Project Setup (30 min)
- GitHub Actions calls CAI API to create project "xray-scanning-model"
- CAI clones GitHub repo to `/home/cdsw`
- All project files available in CAI workspace
- Includes: code, configs, scripts, requirements

### Phase 2: Job Creation (10 min)
- GitHub Actions calls CAI API to create 4 jobs
- Jobs defined from `cai_integration/jobs_config.yaml`
- vLLM endpoint injected from workflow input
- Job dependencies configured automatically

### Phase 3: Execution (Manual)
- User navigates to CAI UI → Jobs
- Triggers root job (setup_environment)
- Child jobs auto-trigger when parent succeeds:
  - setup_environment (1h)
  - download_dataset (1h)
  - generate_vqa (3-5h)
  - finetune_model (6-12h)
- Total pipeline time: 11-19 hours

## Monitoring Workflows

### View Workflow Runs

```bash
# List recent workflow runs
gh run list

# View specific run
gh run view <run-id>

# Watch a running workflow
gh run watch <run-id>

# View logs
gh run view <run-id> --log
```

### GitHub Actions UI

1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select a workflow from the left sidebar
4. View run history and logs

### Job Summaries

Each workflow creates a detailed summary with:
- Status of each job
- Project IDs (for CAI deployment)
- Configuration parameters (vLLM endpoint, etc.)
- Error messages (if any)
- Next steps

## Troubleshooting

### Deployment Fails

**Check CAI credentials:**
```bash
# Verify CML_HOST is accessible
curl -I $CML_HOST

# Verify CML_API_KEY works
curl -H "Authorization: Bearer $CML_API_KEY" \
  "$CML_HOST/api/v2/projects" | jq '.projects[0].name'
```

**Check workflow logs:**
```bash
gh run view --log
```

**Common issues:**
- API key invalid/expired → Update `CML_API_KEY` secret
- Project not found → Check `CML_HOST` configuration
- Git clone timeout → Increase timeout in setup_project.py
- vLLM endpoint unreachable → Verify vLLM server is running

### Validation Fails

**Configuration file not found:**
- Ensure all CAI integration files are committed
- Check paths: `cai_integration/jobs_config.yaml`, etc.

**YAML syntax error:**
```bash
# Test locally
python -c "import yaml; yaml.safe_load(open('cai_integration/jobs_config.yaml'))"
```

### Project Creation Fails

**Check CAI project permissions:**
- User must have project creation permissions
- Check CAI workspace quotas (CPU, memory, storage)

**Private repository access:**
- Add `GH_PAT` secret with `repo` scope
- Ensure PAT is valid and not expired

### Job Creation Fails

**Check jobs_config.yaml:**
- Validate YAML syntax
- Ensure all required fields are present
- Check runtime identifiers are valid for your CAI

**API quota exceeded:**
- Wait and retry
- Contact CAI admin to increase quotas

## Best Practices

1. **Test with Small Dataset First**:
   - Use `samples_per_image=1` for initial test
   - Verify pipeline works end-to-end
   - Then run full pipeline with `samples_per_image=3`

2. **Don't Auto-Trigger Jobs**:
   - Set `trigger_jobs=false` (default)
   - Manually trigger in CAI UI after verifying setup
   - Saves GPU hours if configuration is wrong

3. **Verify vLLM Endpoint**:
   - Test vLLM server before deployment
   - Use curl or Python to verify API works
   - Check network connectivity from CAI

4. **Monitor Job Progress**:
   - Check CAI UI regularly
   - Review job logs for errors
   - Set up email notifications in CAI

5. **Version Control**:
   - All changes to `jobs_config.yaml` should be committed
   - Tag releases for production deployments
   - Document any manual configuration changes

## Security

**Secrets Management:**
- Never commit secrets to repository
- Use GitHub Secrets for sensitive data
- Rotate API keys regularly (every 90 days)
- Use minimal permission scopes

**Code Security:**
- Review all code changes before deploying
- Test in staging environment first
- Enable GitHub security features:
  - Dependabot for dependency updates
  - Secret scanning
  - Code scanning (optional)

## Workflow Trigger Reference

### Deploy Workflow (`deploy-to-cai.yml`)

| Trigger | Type | How | Parameters |
|---------|------|-----|------------|
| workflow_dispatch | Manual | GitHub UI or CLI | vllm_endpoint (required), force_reinstall, trigger_jobs, samples_per_image |
| No automatic trigger | - | Must be manual | - |

**Manual Trigger via GitHub UI:**
1. Go to `Actions` tab
2. Select `Deploy X-ray VQA to CAI`
3. Click `Run workflow` button
4. Fill in vLLM endpoint (required)
5. Adjust other parameters (optional)

**Manual Trigger via CLI:**
```bash
gh workflow run deploy-to-cai.yml \
  -f vllm_endpoint=http://your-vllm-server:8000/v1 \
  -f force_reinstall=false \
  -f trigger_jobs=false \
  -f samples_per_image=3
```

## How Workflows Use Secrets

### Secret Storage
1. Repository Settings → `Secrets and variables` → `Actions`
2. Add each secret with name and value
3. Referenced in workflows as `${{ secrets.SECRET_NAME }}`

### Deploy Workflow
```yaml
env:
  CML_HOST: ${{ secrets.CML_HOST }}
  CML_API_KEY: ${{ secrets.CML_API_KEY }}
  GH_PAT: ${{ secrets.GH_PAT || secrets.GITHUB_TOKEN }}
```

## How Workflows Access GitHub Context

### Available Context Variables
```yaml
github.repository      # "FerdinandZhong/xray_scanning_model_finetuning"
github.ref            # "refs/heads/main"
github.ref_name       # "main"
github.sha            # commit SHA
github.event_name     # "workflow_dispatch"
github.event.inputs   # manual input parameters
```

### Example Usage in Workflows
```yaml
# Use in log message
- run: echo "Deploying ${{ github.ref_name }} to CAI"

# Access manual inputs
- run: |
    python deploy.py \
      --vllm-endpoint=${{ github.event.inputs.vllm_endpoint }}
```

## Environment Variable Flow

```
GitHub Workflow Input (vllm_endpoint)
    ↓
GitHub Actions Environment Variable
    ↓
Python script updates jobs_config.yaml
    ↓
CAI API creates jobs with updated config
    ↓
CAI jobs use VLLM_API_BASE from config
    ↓
VQA generation connects to vLLM server
```

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Triggers Reference](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows)
- [GitHub CLI Manual](https://cli.github.com/manual)
- [CAI Integration Guide](../cai_integration/README.md)
- [Complete Workflow](../docs/COMPLETE_WORKFLOW.md)
- [Qwen vLLM Guide](../docs/QWEN_VL_VLLM_GUIDE.md)

## Support

For issues with workflows:
- Check workflow logs in Actions tab
- Review CAI project logs
- Verify secrets are configured correctly
- Test vLLM endpoint connectivity
- See [CAI Integration README](../cai_integration/README.md) for detailed troubleshooting
