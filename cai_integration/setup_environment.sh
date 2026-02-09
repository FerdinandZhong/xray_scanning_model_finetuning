#!/bin/bash
set -e

echo "=============================================================="
echo "Starting X-ray VQA environment setup"
echo "=============================================================="

# Sync latest code from git (if in git repo)
echo "Syncing latest code from repository..."
cd /home/cdsw
if [ -d ".git" ]; then
    if git pull; then
        echo "✓ Code synced successfully"
    else
        echo "⚠ Git pull failed (continuing with existing code)"
    fi
else
    echo "Not a git repository, skipping git sync"
fi

VENV_PATH="/home/cdsw/.venv"

# Check if we can skip setup (venv exists with transformers and torch installed)
check_existing_setup() {
    if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/python" ]; then
        # Check if required packages are installed
        if "$VENV_PATH/bin/python" -c "import transformers; import torch" 2>/dev/null; then
            return 0  # Setup exists
        fi
    fi
    return 1  # Setup needed
}

# Check FORCE_REINSTALL flag
if [ "${FORCE_REINSTALL:-false}" = "true" ]; then
    echo "FORCE_REINSTALL=true, performing full setup..."
    SKIP_SETUP=false
elif check_existing_setup; then
    echo "✓ Existing setup detected with required packages installed"
    SKIP_SETUP=true
else
    echo "No existing setup found, performing full setup..."
    SKIP_SETUP=false
fi

if [ "$SKIP_SETUP" = "true" ]; then
    echo "Skipping environment setup (use FORCE_REINSTALL=true to force)"

    # Just verify the installation
    echo "Verifying installation..."
    source "$VENV_PATH/bin/activate"
    python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}')"

    echo "=============================================================="
    echo "Environment already configured - setup skipped"
    echo "Virtual environment location: $VENV_PATH"
    echo "=============================================================="
else
    # Full setup path
    # Check if virtual environment exists
    if [ -d "$VENV_PATH" ]; then
        if [ "${FORCE_REINSTALL:-false}" = "true" ]; then
            echo "Removing existing virtual environment for reinstall..."
            rm -rf "$VENV_PATH"
            echo "Creating new virtual environment at $VENV_PATH"
            python3.10 -m venv "$VENV_PATH"
            echo "✓ Virtual environment created successfully"
        else
            echo "✓ Virtual environment already exists at $VENV_PATH"
            echo "  Reusing existing environment..."
        fi
    else
        echo "Creating new virtual environment at $VENV_PATH"
        python3.10 -m venv "$VENV_PATH"
        echo "✓ Virtual environment created successfully"
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"

    # Disable --user flag (conflicts with virtualenv)
    export PIP_USER=0

    # Try to upgrade pip (non-critical)
    echo "Attempting to upgrade pip..."
    if pip install --upgrade pip; then
        echo "✓ Pip upgraded successfully"
    else
        echo "⚠ Pip upgrade failed (continuing anyway)"
    fi

    # Install dependencies from requirements.txt
    echo "Installing dependencies from setup/requirements.txt..."
    if [ -f "setup/requirements.txt" ]; then
        pip install -r setup/requirements.txt
        echo "✓ All dependencies installed successfully"
    else
        echo "❌ Error: setup/requirements.txt not found"
        exit 1
    fi

    # Verify installation
    echo "Verifying installation..."
    python -c "import torch; import transformers; import peft; print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}, PEFT: {peft.__version__}')"

    echo "=============================================================="
    echo "Environment setup completed successfully"
    echo "Virtual environment location: $VENV_PATH"
    echo "=============================================================="
fi
