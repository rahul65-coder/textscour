#!/usr/bin/env bash
# scripts/poll_all.sh — Poll all active Kaggle kernels across accounts until completion
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INTERVAL="${INTERVAL:-30}"
TIMEOUT="${TIMEOUT:-43200}"

echo "=== TEXTSCOUR Active Kernel Poll Helper ==="
echo "Polling interval: ${INTERVAL}s, timeout: ${TIMEOUT}s"

# If specific slugs are passed as CLI arguments, poll them sequentially or in background
if [ $# -gt 0 ]; then
    for slug in "$@"; do
        echo "Polling kernel slug: $slug..."
        python3 -m kaggle_ops.poll_kernel --account acc1 --slug "$slug" --interval "$INTERVAL" --timeout "$TIMEOUT" || true
    done
    exit 0
fi

# Otherwise inspect dispatch plan and active accounts
echo "Listing active kernels from configured accounts..."
python3 -c "
import sys
from kaggle_ops import accounts

for acc in accounts.ACCOUNTS.keys():
    try:
        user = accounts.whoami(acc)
        print(f'Checking kernels for account {acc} (user: {user})...')
        res = accounts.run(acc, ['kernels', 'list', '--mine', '--page-size', '10'], timeout=30.0)
        lines = [l for l in (res.stdout or '').splitlines() if l.strip()]
        for l in lines[:10]:
            print(f'  [{acc}] {l}')
    except Exception as e:
        print(f'  [{acc}] Notice: {e}')
"

echo "=== Done polling check ==="
