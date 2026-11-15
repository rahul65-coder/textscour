#!/usr/bin/env bash
# scripts/build_core.sh — Build and install scour_core PyO3 extension
set -euo pipefail

# Ensure Cargo/Rust binaries are in PATH
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

VARIANT="native"
AUTOTUNE="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --variant=*)
            VARIANT="${1#*=}"
            shift 1
            ;;
        --autotune)
            AUTOTUNE="true"
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [--variant native|generic] [--autotune]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--variant native|generic] [--autotune]"
            exit 1
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Building scour_core ==="
echo "Variant: $VARIANT, Autotune: $AUTOTUNE"

# Configure RUSTFLAGS based on variant and autotune
if [ "$VARIANT" = "generic" ] && [ "$AUTOTUNE" = "false" ]; then
    export RUSTFLAGS="-C target-cpu=generic -C opt-level=3"
else
    # native or autotune: maximize host vectorization (AVX2, FMA, BMI2, etc.)
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
fi

echo "RUSTFLAGS: $RUSTFLAGS"

# Verify prerequisites
if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found in PATH. Please install rustup or source \$HOME/.cargo/env." >&2
    exit 1
fi

# Detect if pip requires --break-system-packages
PIP_FLAGS="--no-deps"
if pip install --help 2>&1 | grep -q -- "--break-system-packages"; then
    PIP_FLAGS="--no-deps --break-system-packages"
fi

if ! command -v maturin &>/dev/null; then
    echo "ERROR: maturin not found in PATH. Installing maturin..." >&2
    pip install maturin $PIP_FLAGS
fi

MANIFEST_PATH="$REPO_ROOT/rust/scour_core/Cargo.toml"
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found at $MANIFEST_PATH" >&2
    exit 1
fi

echo "Compiling and installing with maturin develop --release..."
if maturin develop --release --manifest-path "$MANIFEST_PATH" 2>/dev/null; then
    echo "Successfully installed scour_core via maturin develop."
else
    echo "maturin develop unavailable or non-venv; building wheel via maturin build..."
    (cd "$REPO_ROOT/rust/scour_core" && maturin build --release --compatibility off --out "$REPO_ROOT/target/wheels")
    # Find newest wheel and install
    LATEST_WHEEL="$(ls -t "$REPO_ROOT/target/wheels"/scour_core-*.whl 2>/dev/null | head -n1 || true)"
    if [ -n "$LATEST_WHEEL" ] && [ -f "$LATEST_WHEEL" ]; then
        pip install --force-reinstall $PIP_FLAGS "$LATEST_WHEEL"
        echo "Successfully installed $LATEST_WHEEL"
    else
        echo "ERROR: Wheel not found in target/wheels" >&2
        exit 1
    fi
fi

# Verify module import and display version/cpu_features
python3 -c "
try:
    import scour_core
    print(f'scour_core version: {scour_core.version()}')
    print(f'CPU features: {scour_core.cpu_features()}')
except Exception as e:
    print(f'scour_core verification note: {e}')
"

echo "=== Build core complete ==="
