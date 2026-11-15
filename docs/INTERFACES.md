# TEXTSCOUR Interface Contract & Specifications (`docs/INTERFACES.md`)

This document defines the formal, strict interface contracts for TEXTSCOUR across the Rust core (`scour_core`), Python runtime (`textscour`), queue messages, configuration schemas, checkpoint schemas, output formats, and error catalogs.

---

## 1. PyO3 Native Extension Interface (`scour_core`)

### 1.1 `scour_core.Engine`
Top-level multithreaded processing engine backed by Rayon threadpool and compiled pattern automata.

```rust
// Rust Definition
#[pyclass]
pub struct Engine {
    // compiled automata and threadpool internals
}

#[pymethods]
impl Engine {
    /// Creates a new Engine instance from a Python configuration dictionary.
    /// Rejects invalid configurations or missing lexicon files with PyValueError.
    #[new]
    pub fn new(cfg: &Bound<'_, PyDict>) -> PyResult<Self>;

    /// Executes Stage 1 processing over a batch of document texts.
    /// Releases the GIL and runs parallel processing over Rayon worker pool.
    pub fn stage1(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Stage1Batch>;
}
```

```python
# Python Signature
class Engine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialize engine with config dict. Raises ValueError on schema or file errors."""
        ...

    def stage1(self, texts: list[str]) -> Stage1Batch:
        """Run Stage 1 cleaning on texts. Releases GIL. Returns Stage1Batch."""
        ...
```

---

### 1.2 `scour_core.Stage1Batch`
Intermediate batch holding per-document chunk spans, stripped candidate spans, and ambiguous chunks requiring GPU scoring or rules-only resolution.

```rust
// Rust Definition
#[pyclass]
pub struct Stage1Batch {
    // internal doc states, chunk spans, strip ranges, and ambiguity queues
}

#[pymethods]
impl Stage1Batch {
    /// Returns the total number of documents in this batch.
    pub fn n_docs(&self) -> usize;

    /// Returns all ambiguous chunks across the batch for classification.
    /// Tuple contains:
    ///   0: Vec<u32> - document index for each ambiguous chunk
    ///   1: Vec<u32> - chunk index within the document
    ///   2: Vec<String> - post-strip chunk text to be scored
    pub fn ambiguous(&self) -> (Vec<u32>, Vec<u32>, Vec<String>);

    /// Finalizes the batch into final DocResult objects.
    /// `probs`: classification probabilities aligned to the exact order returned by `ambiguous()`.
    /// `mode`: "gpu" (uses gpu_drop_p and gpu_strip_p thresholds) or "rules_only" (uses rule midpoint).
    pub fn finalize(&self, py: Python<'_>, probs: Vec<f32>, mode: &str) -> PyResult<Vec<DocResult>>;
}
```

```python
# Python Signature
class Stage1Batch:
    def n_docs(self) -> int:
        """Returns the number of documents in the batch."""
        ...

    def ambiguous(self) -> tuple[list[int], list[int], list[str]]:
        """Returns (doc_indices, chunk_indices, chunk_texts) for all ambiguous chunks."""
        ...

    def finalize(self, probs: list[float], mode: str) -> list[DocResult]:
        """Finalize documents into DocResult objects using probabilities and mode ('gpu' | 'rules_only')."""
        ...
```

---

### 1.3 `scour_core.DocResult`
Output container for a single processed document.

```rust
// Rust Definition
#[pyclass]
pub struct DocResult {
    #[pyo3(get)]
    pub clean: bool,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub length: usize,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub reasons: Vec<String>,
    #[pyo3(get)]
    pub fragments: Vec<(String, String)>, // (reason, fragment_text)
    #[pyo3(get)]
    pub exact: u64,
    #[pyo3(get)]
    pub bands: Vec<u64>,
    #[pyo3(get)]
    pub orig_len: usize,
    #[pyo3(get)]
    pub lang: String,
    #[pyo3(get)]
    pub domain: String,
}
```

```python
# Python Attributes
class DocResult:
    clean: bool               # True if document passes clean thresholds and is not duplicate
    text: str                 # Rebuilt text if clean; original or stripped text if garbage
    length: int               # Unicode character count of final text
    score: float              # Quality score in [0.0, 10.0] rounded to 1 decimal place
    reasons: list[str]        # Triggered reason codes (from closed set)
    fragments: list[tuple[str, str]] # Dropped/stripped fragments as (reason, text) pairs
    exact: int                # XXH3 64-bit exact hash of normalized text
    bands: list[int]          # 8 MinHash-LSH band hashes (64-bit unsigned integers)
    orig_len: int             # Original raw character length before processing
    lang: str                 # Detected language: 'en', 'hi', 'hinglish', or 'other'
    domain: str               # Classified domain (when domain_mode == 'topic', else empty)
```

---

### 1.4 `scour_core.DedupIndex`
Global exact and MinHash-LSH deduplication index supporting atomic disk persistence and merging.

```rust
// Rust Definition
#[pyclass]
pub struct DedupIndex {
    // exact: HashSet<u64>
    // bands: [HashSet<u64>; 8]
    // n: u64
}

#[pymethods]
impl DedupIndex {
    #[new]
    pub fn new() -> Self;

    /// Checks whether the exact hash or any MinHash band matches an existing entry.
    /// If duplicate: returns Some("duplicate_exact") or Some("duplicate_near") and does NOT insert.
    /// If unique: inserts hashes into the index and returns None.
    pub fn check_and_insert(&mut self, exact: u64, bands: Vec<u64>) -> Option<&'static str>;

    /// Returns the total number of unique documents indexed.
    pub fn len(&self) -> usize;

    /// Returns the estimated heap memory footprint in bytes.
    pub fn mem_bytes(&self) -> usize;

    /// Atomically persists index to disk using bincode + zstd compression.
    pub fn save(&self, path: &str) -> PyResult<()>;

    /// Loads index from disk.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self>;

    /// Merges entries from another DedupIndex into self.
    pub fn merge_from(&mut self, other: &DedupIndex);
}
```

```python
# Python Signature
class DedupIndex:
    def __init__(self) -> None:
        ...

    def check_and_insert(self, exact: int, bands: list[int]) -> str | None:
        """Returns 'duplicate_exact', 'duplicate_near', or None. Inserts if unique."""
        ...

    def __len__(self) -> int:
        ...

    def mem_bytes(self) -> int:
        """Returns heap memory usage in bytes."""
        ...

    def save(self, path: str) -> None:
        """Atomically saves dedup state to path using tempfile + rename."""
        ...

    @staticmethod
    def load(path: str) -> DedupIndex:
        """Loads dedup state from path."""
        ...

    def merge_from(self, other: DedupIndex) -> None:
        """Merges another DedupIndex into this index."""
        ...
```

---

### 1.5 Module Functions (`scour_core`)

```rust
// Rust Signatures
#[pyfunction]
pub fn version() -> String;

#[pyfunction]
pub fn cpu_features() -> Vec<String>;

#[pyfunction]
pub fn bench_stage1(texts: Vec<String>, cfg: &Bound<'_, PyDict>, seconds: f64) -> PyResult<f64>;
```

```python
# Python Signatures
def version() -> str:
    """Returns the compiled scour_core semantic version string."""
    ...

def cpu_features() -> list[str]:
    """Returns the active CPU target features (e.g. ['avx2', 'fma', 'bmi2'])."""
    ...

def bench_stage1(texts: list[str], cfg: dict[str, Any], seconds: float) -> float:
    """Runs stage1 in a tight loop for `seconds` duration. Returns docs/second."""
    ...
```

---

## 2. Configuration Schema & Types

### 2.1 Python Configuration Dataclasses (`textscour.config`)

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class TextSpec:
    mode: Literal['field', 'path', 'concat', 'template', 'messages', 'auto'] = 'auto'
    field: str | None = None
    fields: list[str] = field(default_factory=list)
    sep: str = "\n\n"
    template: str | None = None
    role_filter: list[str] | None = None

@dataclass
class SourceCfg:
    type: Literal['hf', 'jsonl', 'json', 'parquet']
    repo: str | None = None
    name: str | None = None
    split: str = "train"
    revision: str | None = None
    data_files: str | list[str] | None = None
    paths: list[str] = field(default_factory=list)
    text: TextSpec = field(default_factory=TextSpec)
    id_field: str | None = None
    domain: str | None = None
    weight: float = 1.0
    max_docs: int | None = None

@dataclass
class OutputCfg:
    dir: str = "outputs"
    clean_name: str = "clean"
    garbage_name: str = "garbage"
    format: Literal['jsonl', 'jsonl.zst', 'parquet'] = 'jsonl'
    shard_mb: int = 500
    emit_fragments: bool = False
    extended_columns: bool = False
    final_merge: bool = False

@dataclass
class UploadCfg:
    enabled: bool = False
    repo: str | None = None
    private: bool = True
    commit_every_shards: int = 4
    commit_every_seconds: int = 300
    delete_after_upload: bool = False
    state_backend: Literal['hf', 'local'] = 'local'

@dataclass
class CascadeCfg:
    gpu_mode: Literal['auto', 'off'] = 'auto'
    model: str | None = "distilbert-base-multilingual-cased"
    gpu_drop_p: float = 0.85
    gpu_strip_p: float = 0.50
    promo_lo: float = 0.30
    promo_hi: float = 0.70
    url_policy: Literal['smart', 'remove_all', 'allowlist_only', 'keep_all', 'blocklist'] = 'smart'
    allow_hosts: list[str] = field(default_factory=list)
    block_hosts: list[str] = field(default_factory=list)
    min_doc_chars: int = 200
    clean_threshold: float = 6.0
    pii_mode: Literal['redact', 'drop', 'off'] = 'redact'
    pii_drop_min_hits: int = 1
    allowed_langs: list[str] = field(default_factory=lambda: ["en", "hi", "hinglish"])
    lexicon_dir: str = "configs/lexicons"
    domain_mode: Literal['field', 'url_host', 'dataset', 'topic'] = 'dataset'
    keep_urls_in_code: bool = True
    chunk_target_chars: int = 800
    chunk_min_chars: int = 100
    chunk_max_chars: int = 2000
    max_doc_chars: int = 2000000

@dataclass
class RuntimeCfg:
    autotune: Literal['auto', 'cached', 'off'] = 'auto'
    readers: int | Literal['auto'] = 'auto'
    workers: int | Literal['auto'] = 'auto'
    rust_threads: int | Literal['auto'] = 'auto'
    batch_docs: int | Literal['auto'] = 'auto'
    queue_depth: int | Literal['auto'] = 'auto'
    gpu_batch_tokens: int | Literal['auto'] = 'auto'
    gpu_inflight: int = 2
    gpu_job_chunks: int = 128
    max_runtime_seconds: int | None = None
    disk_budget_gb: float = 16.0
    ram_budget_frac: float = 0.80
    max_error_rate: float = 0.01

@dataclass
class DedupCfg:
    mode: Literal['exact', 'minhash', 'off'] = 'minhash'
    shingle_k: int = 5
    max_mem_gb: float = 2.0

@dataclass
class Config:
    sources: list[SourceCfg]
    output: OutputCfg = field(default_factory=OutputCfg)
    upload: UploadCfg = field(default_factory=UploadCfg)
    cascade: CascadeCfg = field(default_factory=CascadeCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    dedup: DedupCfg = field(default_factory=DedupCfg)
```

### 2.2 Rust Engine Configuration Mapping (`scour_core::config::Config`)

The dictionary passed to `scour_core.Engine(cfg_dict)` contains:

| Key | Type | Default | Description |
|---|---|---|---|
| `chunk_target_chars` | `usize` | `800` | Target soft chunk character size |
| `chunk_min_chars` | `usize` | `100` | Minimum chunk size before merging into predecessor |
| `chunk_max_chars` | `usize` | `2000` | Hard cap on chunk size before splitting at whitespace |
| `url_policy` | `String` | `"smart"` | One of `"smart"`, `"remove_all"`, `"allowlist_only"`, `"keep_all"`, `"blocklist"` |
| `allow_hosts` | `Vec<String>` | (Built-in standard allowlist) | List of allowed domain suffixes |
| `block_hosts` | `Vec<String>` | `[]` | List of blocked domain suffixes |
| `promo_lo` | `f32` | `0.30` | Lower sentence promo score threshold (start of ambiguity) |
| `promo_hi` | `f32` | `0.70` | Upper sentence promo score threshold (definitive promo) |
| `gpu_drop_p` | `f32` | `0.85` | Ambiguous chunk classifier threshold to drop entire chunk |
| `gpu_strip_p` | `f32` | `0.50` | Ambiguous chunk classifier threshold to strip ambiguous sentences |
| `min_doc_chars` | `usize` | `200` | Minimum character length for clean document verdict |
| `clean_threshold` | `f32` | `6.0` | Minimum quality score in [0.0, 10.0] for clean verdict |
| `pii_mode` | `String` | `"redact"` | One of `"redact"`, `"drop"`, `"off"` |
| `pii_drop_min_hits` | `usize` | `1` | Minimum PII hits required to drop document when mode is "drop" |
| `allowed_langs` | `Vec<String>` | `["en", "hi", "hinglish"]` | Allowed language codes |
| `rust_threads` | `usize` | `0` (auto = num_cpus) | Rayon worker pool size |
| `lexicon_dir` | `String` | `"configs/lexicons"` | Path to TSV and text lexicon files |
| `domain_mode` | `String` | `"dataset"` | One of `"field"`, `"url_host"`, `"dataset"`, `"topic"` |
| `emit_fragments` | `bool` | `false` | Whether to populate fragment records in output |
| `normalize_unicode` | `bool` | `true` | Apply NFKC Unicode normalization |
| `dedup_shingle_k` | `usize` | `5` | Word shingle size k for MinHash computation |
| `keep_urls_in_code`| `bool` | `true` | Preserve URLs inside markdown code fences/backticks |
| `max_doc_chars` | `usize` | `2000000` | Maximum document chars processed in a single window |
| `gopher` | `Dict` | (Defaults below) | Quality heuristic thresholds |

#### Gopher Sub-Dictionary Schema:
- `min_alpha_ratio`: `f32` (default `0.50`)
- `max_symbol_ratio`: `f32` (default `0.10`)
- `max_dup_line_frac`: `f32` (default `0.30`)
- `max_top_2gram_cov`: `f32` (default `0.20`)
- `max_top_3gram_cov`: `f32` (default `0.18`)
- `max_top_4gram_cov`: `f32` (default `0.16`)
- `max_short_line_frac`: `f32` (default `0.70`)
- `min_avg_line_len`: `f32` (default `40.0`)
- `max_boilerplate_hits`: `u32` (default `2`)
- `min_stopword_ratio`: `f32` (default `0.06`)

---

## 3. Inter-Process Queue Message Contracts

Multiprocessing queues pass lightweight dataclasses or tuples across process boundaries (`multiprocessing.get_context("spawn")`).

### 3.1 `RawBatch` (Reader ──► Pipeline Workers)
```python
@dataclass
class RawBatch:
    source_key: str              # Unique identifier of the originating source
    batch_id: int                # Monotonically increasing batch sequence number per source
    shard_idx: int               # Current reader shard index
    docs: list[dict | str]       # Extracted document records or raw text strings
    ids: list[str]               # Resolved document IDs
    domains: list[str]           # Extracted or configured domain strings
    stream_state: dict | None    # Source checkpoint state (offsets, row groups, or skip counts)
```

### 3.2 `GpuJob` (Pipeline Workers ──► GPU Workers)
```python
@dataclass
class GpuJob:
    worker_id: int               # Originating pipeline worker ID
    batch_id: int                # Batch ID within the worker
    job_idx: int                 # Sub-batch job index for routing
    texts: list[str]             # List of ambiguous chunk texts (max `gpu_job_chunks`)
```

### 3.3 `GpuResult` (GPU Workers ──► Pipeline Workers)
```python
@dataclass
class GpuResult:
    worker_id: int               # Destination pipeline worker ID
    batch_id: int                # Batch ID
    job_idx: int                 # Sub-batch job index
    probs: list[float]           # Float32 probabilities corresponding to each input text
```

### 3.4 `ResultBatch` (Pipeline Workers ──► Ordered Writer)
```python
@dataclass
class ResultBatch:
    batch_id: int                # Batch sequence number (used by reorder buffer)
    source_key: str              # Source identifier
    records: list[DocResult]     # Final DocResult objects produced by finalize()
    stats: dict[str, Any]        # Stage metrics (batch duration, ambiguous count, dropped count)
    stream_state: dict | None    # Carried forward from RawBatch for checkpoint persistence
```

---

## 4. Checkpoint & State Payloads

### 4.1 Checkpoint File (`state/ckpt-{source_key}.json`)
Written via two-phase commit: write to `state/ckpt-{source_key}.json.tmp` -> `fsync` -> rename. When replacing, the previous file is copied to `.prev`.

```json
{
  "source": "source_key_identifier",
  "next_batch_id": 1042,
  "reader_states": {
    "0": {
      "stream_offset": 8492048,
      "records_in": 52100,
      "shard_idx": 0
    }
  },
  "shard_idx": {
    "clean": 5,
    "garbage": 3
  },
  "records_in": 52100,
  "records_out": {
    "clean": 41200,
    "garbage": 10850,
    "quarantine": 50
  },
  "dedup_file": "state/dedup-source_key_identifier.bin",
  "engine_version": "0.1.0",
  "config_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

Critical config keys hashed into `config_hash`: `[url_policy, promo_lo, promo_hi, min_doc_chars, clean_threshold, allowed_langs, pii_mode]`. If loaded checkpoint config hash differs from current config, refuse resume with `E-CKPT-001`.

---

## 5. Output Record Schemas

All shard outputs are partitioned by source and 0-padded index. Write to `.part` files, `fsync`, and atomic rename upon shard rotation (`shard_mb`).

### 5.1 Clean Record Schema (`{clean_name}-{src}-{idx:05d}.jsonl`)
Every clean record has **exactly** these 5 fields in JSON output:

```json
{
  "id": "doc_id_string",
  "domain": "source_domain_string",
  "text": "Cleaned, stripped, and rebuilt document text string.",
  "length": 1284,
  "score": 8.4
}
```

- `id`: Unique identifier (string).
- `domain`: Associated domain name or category (string).
- `text`: Unicode char-normalized, cleaned, promo-stripped document content.
- `length`: Unicode character length (`text.chars().count()`).
- `score`: Document quality score in `[0.0, 10.0]` rounded to one decimal place.

### 5.2 Garbage Record Schema (`{garbage_name}-{src}-{idx:05d}.jsonl`)

```json
{
  "id": "doc_id_string",
  "domain": "source_domain_string",
  "text": "Original raw document text before stripping.",
  "length": 450,
  "score": 3.2,
  "reasons": ["url_spam", "promo_phrase"],
  "kind": "doc"
}
```

When `emit_fragments: true` is configured, extracted garbage chunks or promo sentences are also emitted as separate fragment records:

```json
{
  "id": "doc_id_string#f0",
  "domain": "source_domain_string",
  "text": "Visit https://spam-link.example.com for 50% discount!",
  "length": 53,
  "score": 0.0,
  "reasons": ["promo_phrase", "cta_sentence"],
  "kind": "fragment"
}
```

### 5.3 Quarantine Record Schema (`quarantine.jsonl`)
Record-level failures that do not abort the pipeline are isolated here:

```json
{
  "source": "source_key",
  "batch_id": 142,
  "record_id": "doc_rec_99",
  "code": "E-SCHEMA-002",
  "message": "Expected string for text field but encountered type int"
}
```

---

## 6. Closed Reason-Code Catalog (A.5)

Every garbage record or fragment contains one or more reason codes strictly from this closed set:

| Reason Code | Category | Definition |
|---|---|---|
| `url_spam` | URL | Non-allowlisted bare URL, link farm, or spam shortener in sentence |
| `promo_phrase` | Promo | Matches marketing, promotional, or sales lexicon phrases |
| `promo_handle` | Promo | Social media handle spam (`@username`, Telegram, WhatsApp) |
| `promo_contact` | Promo | Contact spam (unsolicited phone numbers, WhatsApp links, emails) |
| `cta_sentence` | Promo | Call-to-action sentence (e.g., imperative buy/subscribe triggers) |
| `boilerplate` | Web Noise | Generic website boilerplate, footer, copyright, or legal disclaimer |
| `cookie_banner` | Web Noise | GDPR / cookie consent notice text |
| `nav_menu` | Web Noise | Navigation links, breadcrumb lists, short-line menus |
| `symbol_heavy` | Heuristic | Symbol density exceeds Gopher threshold (`symbol_ratio > 0.10`) |
| `repetition` | Heuristic | High duplicated line fraction or repetitive n-gram coverage |
| `too_short` | Quality | Document final character length < `min_doc_chars` (default 200) |
| `low_alpha` | Heuristic | Alphabetic character ratio < `0.50` or low stopword density |
| `gpu_garbage` | Model | Ambiguous chunk classified as garbage by GPU model (`p >= gpu_drop_p`) |
| `wrong_lang` | Language | Detected language is not in `allowed_langs` |
| `duplicate_exact`| Deduplication | XXH3 exact 64-bit normalized hash collision |
| `duplicate_near` | Deduplication | MinHash-LSH band match collision (Jaccard near-duplicate) |
| `pii_dropped` | Privacy | Document dropped due to PII violations under `pii_mode: "drop"` |
| `empty_after_clean`| Structure | Text became empty or whitespace-only after stripping promo spans |
| `low_score` | Quality | Final quality score below `clean_threshold` (default 6.0) |

---

## 7. Actionable Error Code Catalog (Section B.3)

All system exceptions inherit from `ScourError` and render formatted multi-line diagnostic reports.

```text
[<CODE>] stage=<stage> dataset=<dataset> shard=<shard> record_id=<record_id>
  Problem : <summary>
  Where   : <module.py>:<function_or_method>
  Cause   : <underlying exception and details (secrets redacted)>
  Fix     : <prescriptive, actionable remedy>
```

| Error Code | Stage | Description | Fix Hint |
|---|---|---|---|
| `E-CFG-001` | config | Config file missing or invalid YAML syntax | Verify config file path and check YAML syntax with a validator |
| `E-CFG-002` | config | Unknown configuration key encountered | Remove or fix unknown key; check `configs/default.yaml` for valid schema |
| `E-CFG-003` | config | Invalid enum value provided | Check allowed values in INTERFACES.md (e.g. url_policy, pii_mode) |
| `E-CFG-004` | config | Contradictory threshold configuration | Ensure `promo_lo < promo_hi` and `gpu_strip_p <= gpu_drop_p` |
| `E-CFG-005` | config | Invalid chunk sizing parameters | Ensure `chunk_min_chars <= chunk_target_chars <= chunk_max_chars` |
| `E-CFG-006` | config | Lexicon directory or required lexicon file not found | Check `lexicon_dir` path and verify all required TSV/TXT files exist |
| `E-CFG-007` | config | Identical clean and garbage output names | Specify distinct names for `clean_name` and `garbage_name` |
| `E-CFG-008` | config | Upload enabled without destination repository | Set `upload.repo` to a valid Hugging Face dataset identifier |
| `E-CFG-009` | config | Invalid resource budget (workers, threads, or memory) | Set positive values for workers and ensure budgets are within host limits |
| `E-SRC-001` | source | Dataset or local source file path not found | Verify Hugging Face dataset ID or local path existence |
| `E-SRC-002` | source | HF authentication failed (HTTP 401/403) | Set `HF_TOKEN` in env / Kaggle Secrets or accept dataset license terms |
| `E-SRC-003` | source | Network timeout or connection drop after retries | Check internet connectivity or Hugging Face service status; increase retries |
| `E-SRC-004` | source | Source yielded zero records | Verify dataset split, revision, and data_files filtering parameters |
| `E-SCHEMA-001`| schema | Document text field missing or extracted as None | Check `text.field` or `text.mode` configuration against document schema |
| `E-SCHEMA-002`| schema | Document text field is not a string | Verify extractor expression or inspect source schema types |
| `E-SCHEMA-003`| schema | Duplicate document ID collision within same source | Ensure `id_field` is unique or use auto-generated hash IDs |
| `E-RUST-001` | engine | Native extension `scour_core` import failure | Run `bash scripts/build_core.sh` to compile PyO3 module |
| `E-RUST-002` | engine | Engine rejected configuration dictionary | Verify dictionary parameters passed to `scour_core.Engine` |
| `E-RUST-003` | engine | Unhandled panic caught during document processing | Inspect document characters; report panic traceback in quarantine |
| `E-GPU-001` | gpu | CUDA requested but no compatible GPU detected | Set `cascade.gpu_mode: "off"` or run on a GPU-enabled machine |
| `E-GPU-002` | gpu | GPU Out-Of-Memory (OOM) at minimum batch size | Reduce `gpu_batch_tokens` or free competing GPU processes |
| `E-GPU-003` | gpu | Classifier model or tokenizer loading failure | Verify model path or HF model identifier and check disk cache |
| `E-DISK-001` | storage| Available disk space below safety budget | Increase `runtime.disk_budget_gb`, enable upload-and-delete, or prune disk |
| `E-OUT-001` | writer | Failed writing shard or compression stream error | Check disk write permissions, zstandard library, or disk quotas |
| `E-CKPT-001` | resume | Checkpoint file corrupt or config hash mismatch | Use `--fresh` to restart or verify config compatibility |
| `E-UP-001` | upload | Hugging Face upload failed after max retries | Check write permissions for token on destination repo and network |
| `E-UP-002` | upload | Hugging Face commit rate limit exceeded (HTTP 429) | Increase `upload.commit_every_shards` and `commit_every_seconds` |
| `E-WORKER-001`| runtime| Subprocess worker died or heartbeat timed out | Check worker logs for segfaults or OOM killer interventions |
| `E-RATE-001` | runtime| Record error rate exceeded `max_error_rate` | Inspect quarantine.jsonl; check source schema formatting consistency |

---

## 8. Lexicon & Data File Formats

All lexicon files reside in `configs/lexicons/`.

### 8.1 TSV Weighted Phrase Format
Used for:
- `promo_en.tsv` (≥ 150 entries)
- `promo_hi.tsv` (≥ 60 entries)
- `promo_hinglish.tsv` (≥ 80 entries)
- `boilerplate_en.tsv` (≥ 60 entries)
- `boilerplate_hi.tsv` (≥ 30 entries)
- `orphan_refs.tsv` (≥ 40 entries)
- `topics.tsv`

**Format Rules:**
- One entry per line: `<PHRASE>\t<WEIGHT>`
- `WEIGHT`: floating-point number in `(0.0, 1.0]`
- Lines starting with `#` are ignored as comments.
- Empty lines or whitespace-only lines are ignored.
- Matches are bounded by Unicode word boundaries.
- English is case-insensitive; Devanagari is exact.

```tsv
# Example promo_en.tsv
click here to buy	0.85
limited time offer	0.75
subscribe to our channel	0.80
follow us on twitter	0.60
```

### 8.2 Plain Wordlists
Used for:
- `hinglish_words.txt` (≥ 300 entries)
- `stopwords_en.txt` (≥ 150 entries)
- `stopwords_hi.txt` (≥ 100 entries)
- `tlds.txt` (≥ 250 entries)
- `allow_hosts_default.txt` (≥ 20 entries)

**Format Rules:**
- One entry per line (trimmed).
- Comments start with `#`.
- Empty lines ignored.

### 8.3 Synthetic Template Files (`configs/lexicons/synth_templates_*.tsv`)
Used for synthetic benchmark generation and training classifier.
- Format: `lang<TAB>kind<TAB>template`
- `synth_templates_train.tsv`: ≥ 150 templates
- `synth_templates_test.tsv`: ≥ 50 templates (held-out, distinct phrasing families)
- Slots: `{BRAND}`, `{HANDLE}`, `{URL}`, `{PHONE}`, `{CODE}`, `{PCT}`, `{PRICE}`, `{CHANNEL}`, `{APP}`

---

## 9. Calibration Schema (`calibration.json`)

Written by `calibrate.py` and read by `gpu_scorer.py`:

```json
{
  "temperature": 1.15,
  "gpu_drop_p": 0.85,
  "gpu_strip_p": 0.50,
  "precision_garbage": 0.982,
  "recall_garbage": 0.915,
  "model_name": "distilbert-base-multilingual-cased",
  "evaluated_at": "2026-11-12T08:00:00Z"
}
```

- `temperature`: Temperature scaling parameter `T` applied to classifier logits: $p = \sigma(\text{logit} / T)$.
- `gpu_drop_p`: Calibrated probability threshold above which an ambiguous chunk is discarded (`gpu_garbage`).
- `gpu_strip_p`: Calibrated probability threshold above which ambiguous sentences in a chunk are stripped.
- `precision_garbage`: Validation precision achieved at `gpu_drop_p` (target ≥ 0.98).
- `recall_garbage`: Validation recall achieved at `gpu_strip_p` (target ≥ 0.90).
