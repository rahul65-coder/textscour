# TEXTSCOUR

> **High-throughput, streaming web corpus cleaning engine for LLM pretraining and SFT.**  
> Hybrid architecture: Rust core (`scour_core` via PyO3 + Rayon) + Python multiprocessing runtime with AIMD backpressure + GPU-accelerated ambiguous chunk classification.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](rust/scour_core/Cargo.toml)
[![Model](https://img.shields.io/badge/HF%20Model-titanAId%2Ftextscour--classifier-yellow)](https://huggingface.co/titanAId/textscour-classifier)

---

## 1. Architecture & Pipeline Mechanics

TEXTSCOUR cleans massive, unstructured multilingual web corpora (English, Hindi, Hinglish) within strict RAM and disk constraints. It avoids loading entire corpora into memory by combining zero-copy streaming readers, native Rust chunk-level processing, and selective GPU inference.

```text
  Raw Input Streams (HF Datasets / Local JSONL / Parquet)
                           │
                           ▼
  [Process 1..R] Streaming Readers (bounded memory, batching)
                           │
                           ▼ (RawBatch queue, AIMD backpressure)
  [Process 1..W] Rust Pipeline Workers (scour_core via Rayon)
    ├── LangID: English, Hindi (Devanagari), Hinglish (Romanized Hindi)
    ├── PII Scrubbing: Phone (Indian/Intl), Aadhaar (Verhoeff), PAN, Email, IPv4, Cards (Luhn)
    ├── URL & Promo Policy: Taxonomy filtering, markdown anchor preservation, handle stripping
    ├── Chunker & Gopher Heuristics: Sentence splitting, symbol/stopword density, repetition
    └── MinHash LSH: 64-bit XXH3 exact + 64 MinHash permutations in 8 bands (Jaccard >= 0.80)
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   [Ambiguous Chunks <= 8%]       [Deterministic Verdicts >= 92%]
   GPU Classifier Server                 │
   (titanAId/textscour-classifier)       │
   Dual Tesla T4 (fp16 batched)          │
            │                             │
            └──────────────┬──────────────┘
                           │
                           ▼
  [Process 1..W] Chunk Rebuild & Scoring
    ├── Clean paragraph stitching (preserving document continuity)
    ├── 0–10 Quality scoring rubric (length retention, URL density, promo penalty)
    └── Closed-set garbage attribution (19 discrete reason codes)
                           │
                           ▼ (ResultBatch queue)
  Ordered Sharded Writer & Deduplication Index
    ├── Clean Shards:   {clean_name}-{src}-{shard_idx:05d}.jsonl (.zst / .parquet)
    ├── Garbage Shards: {garbage_name}-{src}-{shard_idx:05d}.jsonl
    ├── Quarantine:     quarantine.jsonl (corrupt / unparseable lines)
    └── State Backend:  Two-phase commit checkpointing (safe SIGKILL resume)
```

### The 3-Stage Cascade:
1. **Stage 1 (Deterministic Rust Core):**  
   Every raw document is chunked into logical units. Rule-based filters immediately pass high-confidence clean text or drop obvious junk (spam URLs, adult keywords, SEO footers, low-alpha noise). This resolves >92% of chunks at wire speed with zero GPU overhead.
2. **Stage 2 (GPU Ambiguous Classifier):**  
   Only borderline chunks (ambiguous promo density, borderline language confidence) are dispatched to the GPU inference server (`titanAId/textscour-classifier`, temperature-calibrated DistilBERT fp16).
3. **Stage 3 (Rebuild & Atomic Sharding):**  
   Clean chunks are stitched back into a coherent document. Documents meeting the clean quality threshold (score $\ge 6.0/10$) are written to clean shards. Dropped documents and noise fragments are routed to garbage shards with attribution tags.

### Standardized 5-Field Clean Record
Every clean record emitted by TEXTSCOUR follows this exact schema:
```json
{
  "id": "doc_00192847",
  "domain": "web",
  "text": "Clean stitched document text...",
  "length": 1420,
  "score": 8.75
}
```

---

## 2. Setup & Installation

### Requirements
- **OS:** Linux (Ubuntu 20.04+, Debian 11+, or RHEL/CentOS)
- **Python:** 3.10 to 3.12
- **Rust:** `rustc >= 1.75` and `cargo` (for building native `scour_core`)
- **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ / 12.0+ (Dual T4, V100, A10G, A100 supported). Automatically falls back to CPU if no GPU is detected.

### A. Install from Private GitHub Repository
```bash
# Set your GitHub personal access token with repo read access
export GITHUB_TOKEN="ghp_your_token_here"

# Install package directly from private repo
pip install "git+https://${GITHUB_TOKEN}@github.com/rahul65-coder/textscour.git"
```

### B. Install from Source Clone
```bash
git clone https://github.com/rahul65-coder/textscour.git
cd textscour

# Install Python dependencies and build native PyO3 Rust extension
pip install -e .
bash scripts/build_core.sh
```

### C. Public Model Auto-Download
The fine-tuned classifier model (`titanAId/textscour-classifier`) is hosted **publicly** on Hugging Face.
- No manual weight download or Kaggle dataset mounting is needed.
- On first execution, TEXTSCOUR automatically downloads and caches `model.safetensors`, tokenizer files, and `calibration.json`.
- You can pre-cache everything ahead of time with:
```bash
python -m textscour setup
```

---

## 3. CLI Quickstart & Recipes

### 1. Hardware & Environment Probe (`doctor`)
Inspects system cgroups, CPU flags (AVX2/AVX-512), RAM ceilings, disk capacity, CUDA devices, and Rust extension status:
```bash
python -m textscour doctor
```

### 2. Pre-cache Model & Verify Rust Core (`setup`)
```bash
python -m textscour setup
```

### 3. Dynamic Hardware Autotuner (`autotune`)
Searches worker process count, Rayon thread pools, batch sizes, and queue depths optimized for your exact core topology:
```bash
python -m textscour autotune --config configs/default.yaml --trials 15
```

### 4. Dry-Run Verification (`run --dry-run`)
Validates your input data sources, parsing schema, and cleaning filters on $N$ documents without modifying persistent checkpoints:
```bash
python -m textscour run --config configs/default.yaml --dry-run 5000
```

### 5. Production Pipeline Execution (`run`)
```bash
# Run with custom output shard names
python -m textscour run \
  --config configs/default.yaml \
  --clean-name fine_corpus \
  --garbage-name fine_trash

# Resume from previous run (automatically resumes from last uncommitted shard)
python -m textscour run --config configs/default.yaml --resume

# Start completely fresh (clears previous checkpoints and output files)
python -m textscour run --config configs/default.yaml --fresh
```

---

## 4. Multi-Source Configuration Guide

Configure data sources and pipeline thresholds in a single YAML file.

```yaml
# configs/pipeline.yaml

# 1. Input Sources: Stream Hugging Face datasets and local files simultaneously
sources:
  # Hugging Face streaming dataset
  - type: hf
    repo: "HuggingFaceTB/smollm-corpus"
    split: "train"
    text:
      mode: "field"             # Extract single text field
      field: "text"
    domain: "educational"
    weight: 1.0

  # Chat / Conversational dataset with role filtering
  - type: hf
    repo: "HuggingFaceH4/ultrachat_200k"
    split: "train_sft"
    text:
      mode: "messages"          # Extract conversation history
      role_filter: ["user", "assistant"]
    domain: "dialogue"

  # Local JSONL or Parquet files
  - type: jsonl
    paths:
      - "/data/crawls/*.jsonl"
    text:
      mode: "concat"            # Concatenate multiple fields
      fields: ["title", "body"]
      sep: "\n\n"
    domain: "web"

# 2. Output Sharding & Formats
output:
  dir: "outputs"
  clean_name: "clean"           # Base name for clean output shards
  garbage_name: "garbage"       # Base name for garbage/filtered shards
  format: "jsonl.zst"           # 'jsonl' | 'jsonl.zst' | 'parquet'
  shard_mb: 250                 # Shard rotation threshold in MB
  emit_fragments: true          # Save dropped noisy paragraphs as garbage records

# 3. Cascade & Classifier Settings
cascade:
  gpu_mode: "auto"              # 'auto' (use CUDA if present) | 'off'
  model: "titanAId/textscour-classifier" # Public Hugging Face model
  gpu_drop_p: 0.85              # Classifier threshold to drop chunk
  gpu_strip_p: 0.50             # Threshold to strip inline promo text
  promo_lo: 0.30                # Ambiguity lower bound (routed to GPU)
  promo_hi: 0.70                # Ambiguity upper bound (routed to GPU)
  url_policy: "smart"           # 'smart' (preserves anchors) | 'strict' | 'strip'
  min_doc_chars: 200            # Minimum clean document length
  clean_threshold: 6.0          # Minimum quality score (0-10) to emit to clean shards
  pii_mode: "redact"            # 'redact' (replace with [REDACTED_*]) | 'drop'
  allowed_langs:
    - "en"
    - "hi"
    - "hinglish"

# 4. Runtime & Concurrency Tuning
runtime:
  autotune: "auto"              # Use cached hardware profile or autotune
  readers: "auto"               # Number of reader processes
  workers: "auto"               # Number of parallel pipeline workers
  rust_threads: "auto"          # Rayon threads per worker
  gpu_inflight: 2               # Concurrent GPU batches in flight
  max_runtime_seconds: 41400    # Graceful exit before session cutoff (e.g., Kaggle 11.5h)
  disk_budget_gb: 16.0          # Disk safety ceiling
```

---

## 5. Kaggle Dual-T4 Production Notebook Recipe

Copy and paste this single cell into a Kaggle Notebook with **2x NVIDIA T4 GPUs** enabled:

```python
# =====================================================================
# TEXTSCOUR Kaggle Dual-T4 Setup & Pipeline Runner
# =====================================================================
import os, subprocess, sys

# 1. Authenticate with GitHub & Install TEXTSCOUR
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    gh_token = user_secrets.get_secret("GITHUB_TOKEN")
    hf_token = user_secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
except Exception:
    gh_token = os.environ.get("GITHUB_TOKEN", "")

repo_url = f"git+https://{gh_token}@github.com/rahul65-coder/textscour.git" if gh_token else "git+https://github.com/rahul65-coder/textscour.git"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", repo_url], check=True)

# 2. Compile native Rust extension if installing from git source
if not os.path.exists("/kaggle/working/scour_core_ready"):
    subprocess.run(["bash", "-c", "which maturin || pip install -q maturin"], check=True)
    subprocess.run(["python3", "-m", "textscour", "setup"], check=True)
    with open("/kaggle/working/scour_core_ready", "w") as f:
        f.write("ready")

# 3. Environment Diagnosis & GPU Detection
subprocess.run(["python3", "-m", "textscour", "doctor"])

# 4. Run High-Throughput Production Pipeline
# Uses titanAId/textscour-classifier on Dual T4 GPUs automatically
subprocess.run([
    "python3", "-m", "textscour", "run",
    "--config", "configs/kaggle_dual_t4.yaml",
    "--clean-name", "kaggle_clean",
    "--garbage-name", "kaggle_garbage"
], check=True)
```

---

## 6. Garbage Attribution Codes

Filtered documents and dropped fragments are recorded in garbage shards with exact reasons:

| Reason Code | Category | Trigger Condition |
| :--- | :--- | :--- |
| `url_spam` | Promo / URL | URL density exceeds safety threshold or domain blacklisted |
| `promo_phrase` | Promo / URL | High density of commercial, affiliate, or marketing keywords |
| `low_alpha` | Noise | Ratio of alphabetic characters to total characters $< 0.50$ |
| `short_doc` | Heuristic | Clean length $< 200$ characters |
| `duplicate_exact` | Dedup | Identical 64-bit XXH3 hash match |
| `duplicate_near` | Dedup | MinHash-LSH similarity match (Jaccard $\ge 0.80$) |
| `pii_drop` | Privacy | Unredactable or sensitive PII detected under `pii_mode: drop` |
| `lang_disallowed` | Language | Primary language not in `allowed_langs` (e.g. not en/hi/hinglish) |
| `gpu_garbage` | Classifier | Model assigned $P(\text{garbage}) \ge \text{gpu\_drop\_p}$ |

---

## 7. Testing & Verification

Run the complete test suite across both Rust and Python modules:

```bash
# Run Rust unit, property, and UTF-8 boundary regression tests
cargo test --manifest-path rust/scour_core/Cargo.toml --release

# Run Python integration and supervisor tests
pytest -v tests/
```

---

## 8. License

Apache License 2.0. See [LICENSE](LICENSE) for details.
