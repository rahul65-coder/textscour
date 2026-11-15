#!/usr/bin/env bash
set -euo pipefail

# Ensure Cargo/Rust binaries are in PATH
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

echo "=== Python & Pip Versions ==="
python3 --version
pip --version

echo ""
echo "=== Rust & Maturin Versions ==="
rustc --version
cargo --version
maturin --version

echo ""
echo "=== Key Python Libraries ==="
python3 -c '
for pkg in ["torch", "datasets", "huggingface_hub", "transformers", "pyarrow", "zstandard", "yaml"]:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"{pkg}: {version}")
    except ImportError as e:
        print(f"{pkg}: NOT INSTALLED ({e})")
'

echo ""
echo "=== CPU Cores ==="
echo "Total CPU cores: $(nproc)"

echo ""
echo "=== Memory (RAM) ==="
free -h

echo ""
echo "=== Disk Free Space ==="
df -h .

echo ""
echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi || echo "nvidia-smi failed"
else
    echo "No GPU"
fi
