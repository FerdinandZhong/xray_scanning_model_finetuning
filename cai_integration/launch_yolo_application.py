#!/usr/bin/env python3
"""
Python launcher for YOLO API Application in CAI.
This directly starts the FastAPI server for CAI Applications.
"""

import os
import sys

# Get the project root directory
# In CAI's Jupyter environment, __file__ is not defined, so use cwd
try:
    # Try to use __file__ if available (when run as script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback for Jupyter/IPython environment (CAI Applications)
    # CAI sets the working directory to the project root
    project_root = os.getcwd()

os.chdir(project_root)
print("=" * 60)
print("Starting YOLO X-ray Detection API")
print("=" * 60)
print()

# Get configuration from environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'runs/detect/xray_detection/weights/best.pt')
BACKEND = os.getenv('BACKEND', 'ultralytics')
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.25'))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.45'))
DEVICE = os.getenv('DEVICE', '0')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8080'))

print("Configuration:")
print(f"  Project Root:     {project_root}")
print(f"  Model Path:       {MODEL_PATH}")
print(f"  Backend:          {BACKEND}")
print(f"  Conf Threshold:   {CONF_THRESHOLD}")
print(f"  IOU Threshold:    {IOU_THRESHOLD}")
print(f"  Device:           {DEVICE}")
print(f"  Host:             {HOST}")
print(f"  Port:             {PORT}")
print()

# Check if model is pre-trained
PRETRAINED_MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov11n.pt", "yolov11s.pt", "yolov11m.pt", "yolov11l.pt", "yolov11x.pt"
]

if os.path.exists(MODEL_PATH):
    print(f"✓ Using local model: {MODEL_PATH}")
elif MODEL_PATH in PRETRAINED_MODELS:
    print(f"✓ Using pre-trained model: {MODEL_PATH}")
    print("  (Ultralytics will download automatically if needed)")
    # Pre-download model to avoid timeout on first request
    try:
        print("  Downloading model...")
        from ultralytics import YOLO
        YOLO(MODEL_PATH)
        print(f"  ✓ Model ready (cached at ~/.cache/ultralytics/)")
    except Exception as e:
        print(f"  ⚠ Warning: Could not pre-download model: {e}")
        print(f"  Will download on first API request")
else:
    print(f"ERROR: Model not found: {MODEL_PATH}")
    print()
    print("Available options:")
    print("  - Train a model first")
    print(f"  - Use pre-trained: {', '.join(PRETRAINED_MODELS[:3])}, ...")
    sys.exit(1)

print()

# Check virtual environment
venv_paths = [".venv_yolo/bin/activate", ".venv/bin/activate"]
venv_active = any(os.path.exists(p) for p in venv_paths)
if venv_active:
    print("✓ Virtual environment detected")
else:
    print("⚠ No virtual environment found, using system Python")

# Check dependencies
print("Checking dependencies...")
missing_deps = []

try:
    import fastapi
    import uvicorn
    from ultralytics import YOLO
except ImportError as e:
    print(f"⚠ Missing core dependency: {e}")
    missing_deps.extend(["fastapi", "uvicorn[standard]", "pillow", "ultralytics"])

try:
    import multipart
except ImportError:
    print("⚠ Missing python-multipart (required for file uploads)")
    missing_deps.append("python-multipart")

if missing_deps:
    print(f"Installing {len(missing_deps)} missing packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing_deps)
    print("✓ Dependencies installed")
else:
    print("✓ All dependencies available")

print()
print("=" * 60)
print("Starting API Server")
print("=" * 60)
print()
print("Endpoints will be available at:")
print(f"  Health Check:     http://{HOST}:{PORT}/health")
print(f"  API Docs:         http://{HOST}:{PORT}/docs")
print(f"  OpenAI API:       http://{HOST}:{PORT}/v1/chat/completions")
print(f"  Direct Detection: http://{HOST}:{PORT}/v1/detect")
print()

# Set environment variables for the FastAPI app startup
os.environ['MODEL_PATH'] = MODEL_PATH
os.environ['BACKEND'] = BACKEND
os.environ['CONF_THRESHOLD'] = str(CONF_THRESHOLD)
os.environ['IOU_THRESHOLD'] = str(IOU_THRESHOLD)
os.environ['DEVICE'] = DEVICE

# Import and run the FastAPI server
sys.path.insert(0, os.path.join(project_root, "inference"))

try:
    import uvicorn
    from yolo_api_server import app
    
    # Start the server
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
    
except KeyboardInterrupt:
    print("\n\nShutting down server...")
    sys.exit(0)
    
except Exception as e:
    print(f"\nERROR: Failed to start server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
