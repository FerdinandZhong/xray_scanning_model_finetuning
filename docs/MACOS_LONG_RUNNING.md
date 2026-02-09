# Running Long Processes on MacBook Without Interruption

Guide to prevent your MacBook from sleeping during long-running VQA generation tasks.

## Problem

When running VQA generation (1-2 hours), your MacBook may:
- Go to sleep and interrupt the process
- Stop network connections to AI Gateway
- Lose progress (though checkpoints help)

## Solutions

### Option 1: Use `caffeinate` (RECOMMENDED - Automatic)

The `generate_vqa_gemini.sh` script automatically uses macOS's built-in `caffeinate` command to prevent sleep.

**Automatic (default):**
```bash
# Just run the script - it will prevent sleep automatically
./scripts/generate_vqa_gemini.sh
```

**Manual control:**
```bash
# Disable automatic sleep prevention
USE_CAFFEINATE=no ./scripts/generate_vqa_gemini.sh

# Explicitly enable (default behavior)
USE_CAFFEINATE=yes ./scripts/generate_vqa_gemini.sh
```

**How it works:**
- `caffeinate -i` prevents idle sleep while the process runs
- Automatically stops when the process completes
- No system settings changes needed
- MacBook can still sleep manually (closing lid, menu)

### Option 2: Use `caffeinate` Manually

If running Python scripts directly:

```bash
# Prevent sleep for the entire command
caffeinate -i python data/llm_vqa_generator.py \
  --annotations data/stcray/train/annotations.json \
  --images-dir data/stcray/train/images \
  --output data/stcray_vqa_train.jsonl \
  --model gemini-2.0-flash-exp \
  --samples-per-image 3 \
  --api-base https://ai-gateway.dev.cloudops.cloudera.com/v1

# Or keep terminal awake for a specific duration
caffeinate -t 7200  # 7200 seconds = 2 hours
```

**Caffeinate options:**
- `-i` - Prevent idle sleep (RECOMMENDED)
- `-d` - Prevent display sleep
- `-m` - Prevent disk sleep
- `-s` - Prevent system sleep (requires AC power)
- `-t <seconds>` - Timeout after specified seconds
- `-w <PID>` - Wait for process with PID to exit

### Option 3: System Settings (Manual)

Change power settings temporarily:

**For macOS Ventura/Sonoma (13+):**
1. System Settings → Lock Screen
2. Set "Turn display off on battery when inactive" to "Never"
3. Set "Turn display off on power adapter when inactive" to "Never"
4. Go to Battery → Options
5. Disable "Enable Power Nap"

**For macOS Monterey and earlier:**
1. System Preferences → Battery (or Energy Saver)
2. Battery tab: Set "Turn display off after" to "Never"
3. Power Adapter tab: Set "Turn display off after" to "Never"
4. Uncheck "Put hard disks to sleep when possible"

**Important:** Remember to restore these settings after generation completes!

### Option 4: Third-Party Apps

**Amphetamine (Free, Mac App Store):**
- More control than `caffeinate`
- Schedule wake times
- Trigger-based automation
- [Download from App Store](https://apps.apple.com/us/app/amphetamine/id937984704)

**KeepingYouAwake (Free, Open Source):**
- Simple menu bar app
- One-click sleep prevention
- [Download from GitHub](https://github.com/newmarcel/KeepingYouAwake)

### Option 5: Use `tmux` or `screen` (For Resilience)

Run the process in a persistent terminal session:

**Using tmux (recommended):**
```bash
# Install tmux
brew install tmux

# Start a new tmux session
tmux new -s vqa_generation

# Run VQA generation (with caffeinate)
caffeinate -i ./scripts/generate_vqa_gemini.sh

# Detach from session (Ctrl+B, then D)
# Session continues in background

# Reattach later
tmux attach -t vqa_generation

# List all sessions
tmux ls

# Kill session when done
tmux kill-session -t vqa_generation
```

**Using screen:**
```bash
# Start screen session
screen -S vqa_generation

# Run VQA generation
caffeinate -i ./scripts/generate_vqa_gemini.sh

# Detach (Ctrl+A, then D)

# Reattach later
screen -r vqa_generation

# List sessions
screen -ls
```

**Benefits:**
- Process continues even if terminal closes
- Can disconnect and reconnect
- Survives SSH disconnections (if running remotely)
- Can resume from different location

### Option 6: Run in Background with `nohup`

Run the process in the background, immune to hangups:

```bash
# Run in background with output to log file
nohup caffeinate -i ./scripts/generate_vqa_gemini.sh > vqa_generation.log 2>&1 &

# Check process
jobs
ps aux | grep generate_vqa_gemini

# Monitor progress
tail -f vqa_generation.log

# Kill if needed
kill %1  # or kill <PID>
```

## Recommended Approach

**For best results, combine multiple strategies:**

```bash
# 1. Start tmux session (for resilience)
tmux new -s vqa_gen

# 2. Run script with caffeinate (automatic)
./scripts/generate_vqa_gemini.sh

# 3. Detach from tmux (Ctrl+B, then D)
# 4. Close laptop lid if needed - process continues!

# Later: reattach to check progress
tmux attach -t vqa_gen
```

## Troubleshooting

### Process Still Got Interrupted

**Check if caffeinate is working:**
```bash
# In another terminal, check if caffeinate is running
ps aux | grep caffeinate

# You should see something like:
# user  12345  caffeinate -i python data/llm_vqa_generator.py ...
```

**Check system logs:**
```bash
# View power management logs
log show --predicate 'subsystem == "com.apple.powermanagement"' --last 1h
```

### MacBook Still Sleeps When Closing Lid

**Closing the lid will always sleep on MacBooks** (by design). Solutions:

1. **Keep lid open** (best for short jobs)
2. **Use external monitor** (clamshell mode - keeps running)
3. **Use `caffeinate -s`** (prevents sleep, requires AC power)
4. **Run on desktop/server instead** (for very long jobs)

```bash
# Prevent sleep even when closing lid (requires AC power)
caffeinate -s ./scripts/generate_vqa_gemini.sh
```

### Network Connection Lost During Sleep

If your MacBook sleeps, network connections to AI Gateway will drop.

**Solution:**
- Use `caffeinate` to prevent sleep
- Generator auto-retries failed API calls (3 attempts)
- Checkpoint system saves progress every 100 images
- Resume from checkpoint if interrupted

### Battery Drains Too Fast

VQA generation is CPU/network intensive:

**Solutions:**
1. **Plug into AC power** (recommended)
2. **Reduce batch size:** `SAMPLES_PER_IMAGE=1` instead of 3
3. **Run overnight when plugged in**
4. **Generate in smaller chunks:**
   ```bash
   # Generate 5000 images at a time
   python data/llm_vqa_generator.py \
     --max-images 5000 \
     --output data/stcray_vqa_train_part1.jsonl \
     ...
   ```

### Permission Error with `caffeinate`

```
caffeinate: command not found
```

**Solution:** `caffeinate` is built into macOS (10.8+). If missing:
- Update macOS
- Or use System Settings method instead

## Verification

**Check if your MacBook is staying awake:**

```bash
# Monitor power assertions
pmset -g assertions | grep -i "caffeinate\|preventuseridledisplaysleep"

# Should show something like:
#   pid 12345(caffeinate): [0x000a...] PreventUserIdleDisplaySleep named: "caffeinate -i command"

# Check current power settings
pmset -g
```

## Best Practices

1. **Always plug into AC power** for long-running tasks
2. **Use the automatic `caffeinate` in the script** (easiest)
3. **Use tmux/screen for extra safety** (can detach/reattach)
4. **Monitor first 5-10 minutes** to ensure it's working
5. **Check for network connectivity** to AI Gateway
6. **Verify checkpoints are being saved** (every 100 images)

## Cost Estimation for Interruptions

VQA generation has built-in resilience:

- **Checkpoints every 100 images** (automatic)
- **API call retries** (3 attempts per image)
- **Resume from checkpoint** (just re-run the script)

**If interrupted:**
- Progress is saved in `.checkpoint` file
- Re-running skips already processed images
- No extra cost for resumed generation
- Only generates remaining images

## Quick Reference

| Method | Complexity | Reliability | Best For |
|--------|-----------|-------------|----------|
| Script's auto-caffeinate | ⭐ Easy | ⭐⭐⭐ High | Most users (default) |
| Manual caffeinate | ⭐⭐ Medium | ⭐⭐⭐ High | Custom commands |
| System Settings | ⭐ Easy | ⭐⭐ Medium | Quick fix |
| tmux + caffeinate | ⭐⭐⭐ Advanced | ⭐⭐⭐⭐ Very High | Power users |
| Third-party apps | ⭐⭐ Medium | ⭐⭐⭐ High | GUI preference |

## Summary

**For most users (RECOMMENDED):**
```bash
# 1. Plug into AC power
# 2. Just run the script - sleep prevention is automatic
./scripts/generate_vqa_gemini.sh

# 3. Keep laptop lid open or use external monitor
# 4. Check progress occasionally
```

**For power users:**
```bash
# 1. Start tmux session
tmux new -s vqa

# 2. Run script (automatic caffeinate)
./scripts/generate_vqa_gemini.sh

# 3. Detach (Ctrl+B, D)
# 4. Close terminal, do other work

# 5. Reattach to check progress
tmux attach -t vqa
```

The script handles sleep prevention automatically, so most users don't need to do anything extra! Just make sure you're plugged into power.
