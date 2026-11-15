#!/usr/bin/env bash
# kaggle_ops/bootstrap.sh — Idempotent bootstrap for Kaggle container environment
set -euo pipefail

echo "=== Starting TEXTSCOUR Bootstrap ==="

# 1. Check or install Rust toolchain
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

if ! command -v cargo &>/dev/null; then
    echo "Cargo not detected. Installing minimal rustup toolchain..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    fi
fi

echo "Rust version: $(rustc --version 2>/dev/null || echo 'not found')"
echo "Cargo version: $(cargo --version 2>/dev/null || echo 'not found')"

# 2. Verify/install Python dependencies
echo "Verifying Python dependencies..."
python3 -c '
import sys, subprocess

required = [
    "maturin",
    "zstandard",
    "pyarrow",
    "pyyaml",
    "datasets",
    "huggingface_hub",
    "transformers",
    "xxhash",
]

missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

# Check pynvml if GPU available
try:
    __import__("pynvml")
except ImportError:
    # Attempt to install if missing
    missing.append("pynvml")

if missing:
    print(f"Installing missing packages: {missing}")
    cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location"] + missing
    subprocess.check_call(cmd)
else:
    print("All core Python packages are already installed.")
'

# 3. Locate code repository root and build scour_core native extension
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# If running on Kaggle, the code dataset might be mounted in /kaggle/input
if [ ! -f "$REPO_ROOT/scripts/build_core.sh" ]; then
    FOUND_BUILD="$(find /kaggle/input -name "build_core.sh" 2>/dev/null | head -n1 || true)"
    if [ -n "$FOUND_BUILD" ]; then
        REPO_ROOT="$(cd "$(dirname "$FOUND_BUILD")/.." && pwd)"
    elif [ -f "./scripts/build_core.sh" ]; then
        REPO_ROOT="."
    fi
fi

echo "Repository root located at: $REPO_ROOT"

# Kaggle dataset mount points under /kaggle/input are read-only filesystems.
# Cargo and maturin require writing target/ and Cargo.lock during build.
if [[ "$REPO_ROOT" == /kaggle/input* ]] || [ ! -w "$REPO_ROOT" ]; then
    BUILD_DIR="/tmp/textscour_build"
    echo "Source directory $REPO_ROOT is read-only. Copying to writable $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cp -r "$REPO_ROOT"/* "$BUILD_DIR"/
    REPO_ROOT="$BUILD_DIR"
    echo "Switched build root to writable: $REPO_ROOT"
fi

if [ -f "$REPO_ROOT/scripts/build_core.sh" ]; then
    bash "$REPO_ROOT/scripts/build_core.sh" --autotune
else
    echo "WARNING: scripts/build_core.sh not found at $REPO_ROOT/scripts/build_core.sh"
fi

# Ensure textscour Python package is installed and accessible
if [ -d "$REPO_ROOT/src/textscour" ]; then
    echo "Installing textscour Python package from $REPO_ROOT/src..."
    PIP_FLAGS="--no-deps"
    if pip install --help 2>&1 | grep -q -- "--break-system-packages"; then
        PIP_FLAGS="--no-deps --break-system-packages"
    fi
    pip install $PIP_FLAGS "$REPO_ROOT" || true
    python3 -c "
import sys, shutil, site
from pathlib import Path
src = Path('$REPO_ROOT') / 'src' / 'textscour'
if src.is_dir():
    for p in site.getsitepackages():
        if 'packages' in p:
            dst = Path(p) / 'textscour'
            try:
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                print(f'Installed textscour to {dst}')
                break
            except Exception as e:
                pass
" || true
fi

# 4. Print scour_core info and GPU info
python3 -c "
try:
    import scour_core
    print('scour_core version:', scour_core.version())
    print('scour_core cpu_features:', scour_core.cpu_features())
except Exception as e:
    print('scour_core notice:', e)
"

echo "=== GPU Status ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -L || echo "nvidia-smi command returned error"
else
    echo "No nvidia-smi present"
fi

# 5. Echo mandatory sentinel
echo "TEXTSCOUR_BOOTSTRAP_OK"
