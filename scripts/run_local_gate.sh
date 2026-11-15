#!/usr/bin/env bash
# Phase 2 Local Verification Gate for TEXTSCOUR
set -euo pipefail

# Ensure Cargo/Rust and local bin in PATH
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

GATE_DIR="evidence/local_gate"
mkdir -p "$GATE_DIR/accuracy"
mkdir -p outputs/local_gate

echo "=== [1/11] Checking for stubs, placeholders, and forbidden tokens ==="
python scripts/check_no_stubs.py

echo "=== [2/11] Checking Rust formatting ==="
cargo fmt --check --manifest-path rust/scour_core/Cargo.toml

echo "=== [3/11] Running Rust unit and integration tests ==="
cargo test --manifest-path rust/scour_core/Cargo.toml --release

echo "=== [4/11] Building native core extension ==="
bash scripts/build_core.sh --variant generic

echo "=== [5/11] Verifying scour_core native extension ==="
python -c "import scour_core; print('scour_core version:', scour_core.version())"

echo "=== [6/11] Running environment doctor ==="
python -m textscour doctor

echo "=== [7/11] Running comprehensive Python test suite ==="
pytest -x -q --maxfail=1 tests/

echo "=== [8/11] Running accuracy benchmark ==="
python -m textscour.train.bench_accuracy --config configs/local_gate.yaml --out "$GATE_DIR/accuracy" --docs 3000

echo "=== [9/11] Running dry-run pipeline test ==="
python -m textscour run --config configs/local_gate.yaml --dry-run 5000 --clean-name gate_clean --garbage-name gate_garbage

echo "=== [10/11] Running full pipeline with conservation verification ==="
python -m textscour run --config configs/local_gate.yaml --fresh
python tests/verify_conservation.py outputs/local_gate

echo "=== [11/11] Running kill-resume resilience check ==="
python tests/kill_resume_check.py

echo "=================================================================="
echo "Phase 2 Gate: ALL VERIFICATIONS PASSED SUCCESSFULLY."
echo "=================================================================="
