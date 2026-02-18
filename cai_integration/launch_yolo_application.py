#!/usr/bin/env python3
"""
Python launcher for YOLO API Application in CAI.
This wrapper executes the bash launcher script to start the application.
"""

import os
import sys
import subprocess

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
print(f"Project root: {project_root}")

# Path to the bash launcher script
launcher_script = os.path.join(project_root, "cai_integration", "launch_yolo_application.sh")

if not os.path.exists(launcher_script):
    print(f"ERROR: Launcher script not found: {launcher_script}")
    sys.exit(1)

# Make script executable
os.chmod(launcher_script, 0o755)

# Execute the bash launcher script
print(f"Launching YOLO API via: {launcher_script}")
print()

try:
    # Use exec to replace this process with the bash script
    # This ensures signals and process management work correctly
    os.execv("/bin/bash", ["/bin/bash", launcher_script])
except Exception as e:
    print(f"ERROR: Failed to launch application: {e}")
    sys.exit(1)
