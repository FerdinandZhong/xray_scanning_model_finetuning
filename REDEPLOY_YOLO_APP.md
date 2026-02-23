# Redeploy YOLO Application with Unauthenticated Access

The YOLO application currently requires authentication, which blocks API testing scripts. You need to redeploy with the updated configuration.

## What Changed

Added `bypass_authentication: True` to the application configuration to allow unauthenticated API access for testing.

## Option 1: Redeploy via GitHub Actions (Recommended)

1. Go to **Actions** tab → **Deploy X-ray Detection to CAI**
2. Click **Run workflow**
3. Configure:
   - **model_type**: `yolo`
   - **yolo_model**: `yolov8x.pt` (or your preferred model)
   - **trigger_jobs**: `false` (skip training, just deploy)
4. Click **Run workflow**

This will update the application with the new `bypass_authentication` setting.

## Option 2: Manual Update via CAI Job

Run the deployment job in CAI:

```bash
# In CAI workspace terminal
python cai_integration/deploy_yolo_application.py \
  --api-key $CDSW_APIV2_KEY \
  --domain $CDSW_DOMAIN \
  --project xray-scanning-model \
  --model yolov8x.pt
```

## Option 3: Direct API Call

Update the application directly via CAI API (requires project and application IDs).

## Verify the Change

After redeployment, test unauthenticated access:

```bash
# Should return JSON without redirect
curl https://xray-yolo-api.ml-12abb479-548.qzhong-1.a465-9q4k.cloudera.site/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "yolov8x.pt",
  ...
}
```

## Then Run Comparison

Once the application is redeployed with unauthenticated access:

```bash
python scripts/compare_models_with_yolo.py --num-samples 50
```

## Alternative: Use Authenticated Access

If you prefer to keep authentication enabled, we can modify the comparison script to include authentication headers. However, this requires a CAI API key or session token.
