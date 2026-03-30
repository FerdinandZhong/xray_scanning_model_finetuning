#!/usr/bin/env python3
"""
Combined launcher for the X-Ray Detection stack in CAI.

Starts two servers in a single CAI Application:
  1. YOLO detection backend   — inference/yolo_api_server.py  on 127.0.0.1:8100
  2. Detection UI (frontend)  — cai_integration/xray_detection_ui.py
                                 on 127.0.0.1:$CDSW_APP_PORT

CAI's reverse proxy exposes only CDSW_APP_PORT to the outside world.
The UI proxies detection requests internally to 127.0.0.1:8100 so the
model weights never need to be reachable from the internet directly.

Execution order
───────────────
  [1] Backend process launched in background (Popen)
  [2] Health-check loop waits until /health responds 200 (up to 120 s)
  [3] UI process launched in foreground; this process blocks on it
  [4] On exit / KeyboardInterrupt both processes are terminated cleanly

Environment Variables (all optional):
  MODEL_PATH       Path to .pt weights (default: auto-detect latest run)
  BACKEND          ultralytics | onnx           (default: ultralytics)
  CONF_THRESHOLD   Detection confidence cutoff  (default: 0.25)
  IOU_THRESHOLD    NMS IoU threshold            (default: 0.45)
  DEVICE           0 | cpu                      (default: 0)
  BACKEND_PORT     Internal port for YOLO API   (default: 8100)
  CDSW_APP_PORT    Injected by CAI for the UI   (default: 8101)
"""

import os
import sys
import subprocess
import time
import requests as _requests

# ── Resolve project root ──────────────────────────────────────────────────────
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.getcwd()

os.chdir(project_root)
sys.path.insert(0, project_root)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.getenv("MODEL_PATH", "")
BACKEND         = os.getenv("BACKEND", "ultralytics")
CONF_THRESHOLD  = os.getenv("CONF_THRESHOLD", "0.25")
IOU_THRESHOLD   = os.getenv("IOU_THRESHOLD", "0.45")
DEVICE          = os.getenv("DEVICE", "0")
BACKEND_PORT    = int(os.getenv("BACKEND_PORT", "8100"))
UI_PORT         = int(os.getenv("CDSW_APP_PORT", "8101"))
HEALTH_TIMEOUT  = 120   # seconds to wait for backend to become healthy

BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"


def find_latest_model() -> str:
    """Return the path to the most recently trained best.pt, or '' if none."""
    runs_dir = os.path.join(project_root, "runs", "detect")
    if not os.path.isdir(runs_dir):
        return ""
    candidates = []
    for entry in os.scandir(runs_dir):
        if entry.is_dir():
            pt = os.path.join(entry.path, "weights", "best.pt")
            if os.path.exists(pt):
                candidates.append((os.path.getmtime(pt), pt))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def wait_for_backend(timeout: int = HEALTH_TIMEOUT) -> bool:
    """Poll /health until the backend responds 200 or timeout expires."""
    deadline = time.time() + timeout
    interval = 2
    while time.time() < deadline:
        try:
            r = _requests.get(f"{BACKEND_URL}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
        interval = min(interval * 1.4, 8)   # gentle back-off, cap at 8 s
    return False


# ── Banner ────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  X-Ray Detection Stack — Combined CAI Application Launcher")
print("=" * 62)

# ── Resolve model path ────────────────────────────────────────────────────────
if not MODEL_PATH:
    MODEL_PATH = find_latest_model()

if MODEL_PATH and os.path.exists(MODEL_PATH):
    print(f"  Model:       {MODEL_PATH}")
elif MODEL_PATH:
    print(f"  Model:       {MODEL_PATH} (pre-trained, will auto-download)")
else:
    print("  ERROR: No trained model found.")
    print("  Run the yolo_training job first, or set MODEL_PATH explicitly.")
    sys.exit(1)

print(f"  Backend:     {BACKEND_URL}  (internal, not exposed)")
print(f"  UI:          http://127.0.0.1:{UI_PORT}  (CAI-exposed port)")
print(f"  Device:      {DEVICE}")
print()

backend_proc = None
ui_proc      = None

try:
    # ── [1] Start YOLO backend in background ─────────────────────────────────
    print("[1/3] Starting YOLO detection backend...")
    backend_script = os.path.join(project_root, "inference", "yolo_api_server.py")
    backend_cmd = [
        sys.executable, backend_script,
        "--model",          MODEL_PATH,
        "--backend",        BACKEND,
        "--conf-threshold", CONF_THRESHOLD,
        "--iou-threshold",  IOU_THRESHOLD,
        "--device",         DEVICE,
        "--host",           "127.0.0.1",
        "--port",           str(BACKEND_PORT),
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=project_root,
    )
    print(f"  Backend PID: {backend_proc.pid}")
    print()

    # ── [2] Wait for backend to be healthy ───────────────────────────────────
    print(f"[2/3] Waiting for backend health check (timeout {HEALTH_TIMEOUT} s)…")
    if wait_for_backend():
        print(f"  ✓ Backend healthy at {BACKEND_URL}/health")
    else:
        print(f"  ✗ Backend did not become healthy within {HEALTH_TIMEOUT} s — aborting.")
        backend_proc.terminate()
        sys.exit(1)
    print()

    # ── [3] Start UI in foreground ────────────────────────────────────────────
    print("[3/3] Starting Detection UI (foreground)…")
    ui_script = os.path.join(project_root, "cai_integration", "xray_detection_ui.py")
    ui_env = {
        **os.environ,
        "YOLO_API_URL":  BACKEND_URL,
        "CDSW_APP_PORT": str(UI_PORT),
    }
    ui_cmd = [sys.executable, ui_script]
    ui_proc = subprocess.Popen(
        ui_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=project_root,
        env=ui_env,
    )
    print(f"  UI PID: {ui_proc.pid}")
    print()
    print("  Detection UI:  http://127.0.0.1:{UI_PORT}/")
    print("  YOLO API docs: {BACKEND_URL}/docs")
    print()

    # Block until the UI process exits
    exit_code = ui_proc.wait()
    if exit_code != 0:
        print(f"\nUI process exited with code {exit_code}")
        sys.exit(exit_code)

except KeyboardInterrupt:
    print("\n\nShutting down…")

finally:
    for proc, name in [(ui_proc, "UI"), (backend_proc, "Backend")]:
        if proc and proc.poll() is None:
            print(f"  Terminating {name} (PID {proc.pid})…")
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
