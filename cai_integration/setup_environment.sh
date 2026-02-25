#!/bin/bash
set -e

echo "=============================================================="
echo "Starting X-ray VQA environment setup (with uv - ultra-fast)"
echo "=============================================================="

# Sync latest code and Git LFS files from repository
echo "Syncing latest code from repository..."
cd /home/cdsw
if [ -d ".git" ]; then
    if git pull; then
        echo "✓ Code synced successfully"
    else
        echo "⚠ Git pull failed (continuing with existing code)"
    fi

    # Pull Git LFS objects (RAR files for STCray dataset)
    if command -v git-lfs &> /dev/null || git lfs version &> /dev/null 2>&1; then
        echo "Pulling Git LFS objects (STCray RAR files)..."
        if git lfs pull; then
            echo "✓ Git LFS objects pulled"
        else
            echo "⚠ Git LFS pull failed (RAR files may already be present)"
        fi
    else
        echo "⚠ git-lfs not installed — skipping LFS pull"
    fi
else
    echo "Not a git repository, skipping git sync"
fi

# Install unrar for STCray RAR extraction
echo "Checking unrar availability..."
if ! command -v unrar &> /dev/null; then
    echo "unrar not found, attempting install..."
    if command -v apt-get &> /dev/null; then
        apt-get install -y -q unrar 2>/dev/null || \
        apt-get install -y -q unrar-free 2>/dev/null || \
        echo "⚠ apt-get install unrar failed (may need sudo or alternative)"
    elif command -v yum &> /dev/null; then
        yum install -y -q unrar 2>/dev/null || \
        echo "⚠ yum install unrar failed"
    fi
    # Fall back to 7z if unrar still unavailable
    if ! command -v unrar &> /dev/null && ! command -v 7z &> /dev/null; then
        if command -v apt-get &> /dev/null; then
            apt-get install -y -q p7zip-full 2>/dev/null || true
        fi
    fi
fi
if command -v unrar &> /dev/null; then
    echo "✓ unrar available: $(unrar 2>&1 | head -1)"
elif command -v 7z &> /dev/null; then
    echo "✓ 7z available as RAR extraction fallback"
else
    echo "⚠ Neither unrar nor 7z found — STCray extraction will fail"
fi

VENV_PATH="/home/cdsw/.venv"

# Check if we can skip setup (venv exists with required packages installed)
check_existing_setup() {
    if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/python" ]; then
        # datasets is required by all pipeline jobs (download_stcray.py)
        if ! "$VENV_PATH/bin/python" -c "import datasets" 2>/dev/null; then
            echo "  (datasets package missing, reinstall needed)"
            return 1
        fi
        # Check for YOLO requirements (torch + ultralytics)
        if "$VENV_PATH/bin/python" -c "import torch; from ultralytics import YOLO" 2>/dev/null; then
            echo "  (YOLO environment detected)"
            return 0  # Setup exists
        fi
        # Check for VLM requirements (torch + transformers)
        if "$VENV_PATH/bin/python" -c "import transformers; import torch" 2>/dev/null; then
            echo "  (VLM environment detected)"
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
    
    # Try YOLO verification first, fall back to VLM
    if python -c "import torch; from ultralytics import YOLO" 2>/dev/null; then
        python -c "import torch; from ultralytics import YOLO; print(f'PyTorch: {torch.__version__}, Ultralytics: OK')"
    else
        python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}')"
    fi

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

    # Install uv if not already installed
    echo "Installing uv (ultra-fast Python package installer)..."
    if ! command -v uv &> /dev/null; then
        echo "uv not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        echo "✓ uv installed successfully"
    else
        echo "✓ uv already installed"
    fi

    # Verify uv is available
    if ! command -v uv &> /dev/null; then
        echo "⚠ uv installation failed, falling back to pip..."
        USE_UV=false
    else
        echo "✓ Using uv version: $(uv --version)"
        USE_UV=true
    fi

    # Install dependencies from requirements.txt
    # Auto-detect YOLO vs VLM based on available requirements files
    if [ -f "setup/requirements_yolo.txt" ]; then
        REQUIREMENTS_FILE="setup/requirements_yolo.txt"
        echo "Installing YOLO dependencies (lightweight, no DeepSpeed/vLLM)..."
    elif [ -f "setup/requirements.txt" ]; then
        REQUIREMENTS_FILE="setup/requirements.txt"
        echo "Installing full VLM dependencies..."
    else
        echo "❌ Error: No requirements file found"
        exit 1
    fi
    
    if [ "$USE_UV" = "true" ]; then
        echo "Using uv for ultra-fast installation (10-100x faster than pip)..."
        
        # Clear cache if lock errors occurred before
        if [ -d "$HOME/.cache/uv" ]; then
            echo "Clearing any stale uv locks..."
            find "$HOME/.cache/uv" -name "*.lock" -type f -delete 2>/dev/null || true
        fi
        
        # Install with uv, with retry on lock errors
        MAX_RETRIES=3
        RETRY_COUNT=0
        
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if uv pip install -r "$REQUIREMENTS_FILE"; then
                echo "✓ All dependencies installed successfully"
                break
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                    echo "⚠ Installation failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
                    # Clear cache and retry
                    rm -rf "$HOME/.cache/uv/sdists-v9" 2>/dev/null || true
                    sleep 5
                else
                    echo "⚠ uv installation failed after $MAX_RETRIES attempts"
                    echo "Falling back to pip..."
                    pip install --upgrade pip
                    pip install -r "$REQUIREMENTS_FILE"
                    echo "✓ All dependencies installed successfully (via pip fallback)"
                fi
            fi
        done
    else
        echo "Using pip..."
        pip install --upgrade pip
        pip install -r "$REQUIREMENTS_FILE"
        echo "✓ All dependencies installed successfully"
    fi

    # Verify installation
    echo "Verifying installation..."
    if [ "$REQUIREMENTS_FILE" = "setup/requirements_yolo.txt" ]; then
        # YOLO-specific verification
        python -c "import torch; from ultralytics import YOLO; import fastapi; print(f'PyTorch: {torch.__version__}, Ultralytics: OK, FastAPI: OK')"
    else
        # Full VLM verification
        python -c "import torch; import transformers; import peft; print(f'PyTorch: {torch.__version__}, Transformers: {transformers.__version__}, PEFT: {peft.__version__}')"
    fi

    echo "=============================================================="
    echo "Environment setup completed successfully!"
    if [ "$USE_UV" = "true" ]; then
        echo "✓ Installed using uv (10-100x faster than pip)"
    fi
    echo "Virtual environment location: $VENV_PATH"
    echo "=============================================================="
fi
