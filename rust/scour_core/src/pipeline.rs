use crate::chunker::{chunk, split_sentences, Span};
use crate::config::{Config, DomainMode, Lang, PiiMode};
use crate::lang::detect;
use crate::minhash::{bands, exact_hash};
use crate::pii::{redact, scan};
use crate::promo::{score_sentence, Compiled};
use crate::reasons::{Reason, ReasonSet};
use crate::rebuild::{rebuild, ChunkVerdict};
use crate::urls::{decide, find_urls, UrlAction, UrlSpan};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

#[pyclass(dict)]
#[derive(Debug, Clone)]
pub struct DocResult {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub clean: bool,
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub length: usize,
    #[pyo3(get, set)]
    pub score: f32,
    #[pyo3(get, set)]
    pub reasons: Vec<String>,
    #[pyo3(get, set)]
    pub fragments: Vec<(String, String)>,
    #[pyo3(get, set)]
    pub exact: u64,
    #[pyo3(get, set)]
    pub bands: Vec<u64>,
    #[pyo3(get, set)]
    pub orig_len: usize,
    #[pyo3(get, set)]
    pub lang: String,
    #[pyo3(get, set)]
    pub domain: String,
}

#[derive(Clone)]
struct DocState {
    orig_text: String,
    orig_len: usize,
    lang_label: Lang,
    lang_conf: f32,
    pii_dropped: bool,
    chunks: Vec<Span>,
    verdicts: Vec<ChunkVerdict>,
    ambiguous_chunk_indices: Vec<usize>,
    ambiguous_sentence_spans: Vec<Vec<Span>>,
}

#[pyclass]
pub struct Stage1Batch {
    doc_states: Vec<DocState>,
    compiled: Arc<Compiled>,
    ambiguous_doc_indices: Vec<u32>,
    ambiguous_chunk_indices: Vec<u32>,
    ambiguous_texts: Vec<String>,
}

#[pymethods]
impl Stage1Batch {
    pub fn n_docs(&self) -> usize {
        self.doc_states.len()
    }

    pub fn ambiguous(&self) -> (Vec<u32>, Vec<u32>, Vec<String>) {
        (
            self.ambiguous_doc_indices.clone(),
            self.ambiguous_chunk_indices.clone(),
            self.ambiguous_texts.clone(),
        )
    }

    pub fn finalize(
        &self,
        py: Python<'_>,
        probs: Vec<f32>,
        mode: &str,
    ) -> PyResult<Vec<DocResult>> {
        let compiled = self.compiled.clone();
        let cfg = compiled.cfg.clone();
        let doc_states = self.doc_states.clone();
        let mode_str = mode.to_string();

        py.allow_threads(move || {
            // Map probabilities back to ambiguous chunks
            let mut prob_map: hashbrown::HashMap<(usize, usize), f32> = hashbrown::HashMap::new();
            for (idx, &p) in probs.iter().enumerate() {
                if idx < self.ambiguous_doc_indices.len() {
                    let d_idx = self.ambiguous_doc_indices[idx] as usize;
                    let c_idx = self.ambiguous_chunk_indices[idx] as usize;
                    prob_map.insert((d_idx, c_idx), p);
                }
            }

            let mut results = Vec::with_capacity(doc_states.len());

            for (d_idx, mut doc) in doc_states.into_iter().enumerate() {
                // Apply classifier or rules_only resolution to ambiguous chunks
                for &c_idx in &doc.ambiguous_chunk_indices {
                    let verdict = &mut doc.verdicts[c_idx];
                    if mode_str == "gpu" {
                        let p = prob_map.get(&(d_idx, c_idx)).copied().unwrap_or(0.0);
                        verdict.g = p;
                        if p >= cfg.gpu_drop_p {
                            verdict.keep = false;
                            verdict.reasons.insert(Reason::GpuGarbage);
                        } else if p >= cfg.gpu_strip_p {
                            // Strip ambiguous sentences
                            if let Some(ambig_sents) = doc.ambiguous_sentence_spans.get(c_idx) {
                                for &s in ambig_sents {
                                    verdict.strip.push(s);
                                }
                            }
                        }
                    } else {
                        // mode == "rules_only": midpoint rule resolution
                        let midpoint = 0.625f32;
                        if verdict.g >= midpoint {
                            verdict.keep = false;
                            verdict.reasons.insert(Reason::GpuGarbage);
                        } else if verdict.g >= cfg.promo_lo {
                            // Strip ambiguous promo sentences (score >= promo_lo)
                            if let Some(ambig_sents) = doc.ambiguous_sentence_spans.get(c_idx) {
                                for &s in ambig_sents {
                                    verdict.strip.push(s);
                                }
                            }
                        }
                    }
                }

                // Rebuild document text
                let rebuilt = rebuild(&doc.orig_text, &doc.chunks, &doc.verdicts, &compiled, &cfg);

                let final_len = rebuilt.text.chars().count();
                let exact = exact_hash(&rebuilt.text);
                let band_hashes = bands(&rebuilt.text, cfg.dedup_shingle_k).to_vec();

                let mut reasons = ReasonSet::new();
                for v in &doc.verdicts {
                    reasons |= v.reasons;
                }

                if doc.pii_dropped {
                    reasons.insert(Reason::PiiDropped);
                }
                if rebuilt.text.trim().is_empty() {
                    reasons.insert(Reason::EmptyAfterClean);
                }
                if final_len < cfg.min_doc_chars {
                    reasons.insert(Reason::TooShort);
                }

                let lang_allowed = cfg.allowed_langs.contains(&doc.lang_label);
                if !lang_allowed {
                    reasons.insert(Reason::WrongLang);
                }

                // Compute quality score per Section A.4
                let remaining_urls = find_urls(&rebuilt.text, Some(&compiled.tlds)).len();
                let score = score_doc(
                    &doc.chunks,
                    &doc.verdicts,
                    rebuilt.promo_chars,
                    doc.orig_len,
                    remaining_urls,
                    final_len,
                    lang_allowed,
                    doc.lang_conf,
                    rebuilt.orphan_refs,
                );

                if score < cfg.clean_threshold {
                    reasons.insert(Reason::LowScore);
                }

                let clean = score >= cfg.clean_threshold
                    && final_len >= cfg.min_doc_chars
                    && lang_allowed
                    && !doc.pii_dropped
                    && !rebuilt.text.trim().is_empty();

                // Topic classification when domain_mode == Topic
                let domain = if cfg.domain_mode == DomainMode::Topic {
                    classify_topic(&rebuilt.text, &compiled.topics)
                } else {
                    String::new()
                };

                let fragments = rebuilt
                    .removed
                    .into_iter()
                    .map(|f| (f.reason.as_str().to_string(), f.text))
                    .collect();

                let out_text = if clean {
                    rebuilt.text
                } else {
                    // For garbage docs, keep original or stripped text
                    if rebuilt.text.is_empty() {
                        doc.orig_text
                    } else {
                        rebuilt.text
                    }
                };

                results.push(DocResult {
                    id: String::new(),
                    clean,
                    text: out_text,
                    length: final_len,
                    score,
                    reasons: reasons.to_vec(),
                    fragments,
                    exact,
                    bands: band_hashes,
                    orig_len: doc.orig_len,
                    lang: doc.lang_label.as_str().to_string(),
                    domain,
                });
            }

            Ok(results)
        })
    }
}

pub fn score_doc(
    chunks: &[Span],
    verdicts: &[ChunkVerdict],
    promo_chars_removed: usize,
    orig_chars: usize,
    remaining_urls: usize,
    final_len: usize,
    lang_allowed: bool,
    lang_conf: f32,
    orphan_refs: u32,
) -> f32 {
    let orig_chars = orig_chars.max(1);

    // q = \sum(chars_i * (1 - g_i)) / \sum chars_i
    let mut weighted_q_sum = 0.0f32;
    let mut total_chunk_chars = 0usize;

    for (&chunk, verdict) in chunks.iter().zip(verdicts.iter()) {
        let c_len = chunk.len();
        total_chunk_chars += c_len;
        let g_i = if !verdict.keep { 1.0 } else { verdict.g };
        weighted_q_sum += c_len as f32 * (1.0 - g_i);
    }

    let q = if total_chunk_chars > 0 {
        (weighted_q_sum / total_chunk_chars as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // p_term = 1 - min(1, promo_chars_removed/orig_chars / 0.25)
    let p_ratio = promo_chars_removed as f32 / orig_chars as f32;
    let p_term = (1.0 - (p_ratio / 0.25).min(1.0)).clamp(0.0, 1.0);

    // url_term = 1 - min(1, remaining_urls_per_1k_chars / 5)
    let urls_per_1k = (remaining_urls as f32 * 1000.0) / (final_len as f32).max(1.0);
    let url_term = (1.0 - (urls_per_1k / 5.0).min(1.0)).clamp(0.0, 1.0);

    // len_term = min(1, log10(1 + L/200) / log10(11))
    let l_f32 = final_len as f32;
    let len_term = ((1.0 + l_f32 / 200.0).log10() / 11.0f32.log10())
        .min(1.0)
        .max(0.0);

    // lang_term = lang confidence if language allowed else 0
    let lang_term = if lang_allowed {
        lang_conf.clamp(0.0, 1.0)
    } else {
        0.0
    };

    // score = clamp(10*(0.40q + 0.20p_term + 0.10url_term + 0.15len_term + 0.15lang_term) - min(1.0, 0.1*orphan_refs), 0, 10)
    let base_score =
        10.0 * (0.40 * q + 0.20 * p_term + 0.10 * url_term + 0.15 * len_term + 0.15 * lang_term);
    let penalty = (0.10 * orphan_refs as f32).min(1.0);
    let final_score = (base_score - penalty).clamp(0.0, 10.0);

    // Round to 1 decimal place
    (final_score * 10.0).round() / 10.0
}

fn classify_topic(text: &str, topics: &[(String, crate::promo::Lexicon)]) -> String {
    let mut best_topic = "general".to_string();
    let mut max_weight = 0.0f32;

    for (topic_name, lexicon) in topics {
        let hits = lexicon.find_matches(text);
        let weight_sum: f32 = hits.iter().map(|h| h.2).sum();
        if weight_sum > max_weight {
            max_weight = weight_sum;
            best_topic = topic_name.clone();
        }
    }

    best_topic
}

#[pyclass]
pub struct Engine {
    compiled: Arc<Compiled>,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl Engine {
    #[new]
    pub fn new(cfg: &Bound<'_, PyDict>) -> PyResult<Self> {
        let config = Config::from_pydict(cfg)?;
        config
            .validate_lexicons()
            .map_err(|e| PyValueError::new_err(e))?;

        let compiled = Compiled::build(&config).map_err(|e| PyValueError::new_err(e))?;

        let threads = if config.rust_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            config.rust_threads
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyValueError::new_err(format!("ThreadPool error: {}", e)))?;

        Ok(Self { compiled, pool })
    }

    pub fn stage1(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Stage1Batch> {
        let compiled = self.compiled.clone();

        py.allow_threads(|| {
            let doc_states: Vec<DocState> = self.pool.install(|| {
                texts
                    .par_iter()
                    .map(|raw_text| {
                        // Catch unwinds per document to ensure panic safety
                        let result =
                            std::panic::catch_unwind(|| process_single_doc(raw_text, &compiled));
                        match result {
                            Ok(state) => state,
                            Err(_) => DocState {
                                orig_text: raw_text.clone(),
                                orig_len: raw_text.chars().count(),
                                lang_label: Lang::Other,
                                lang_conf: 0.5,
                                pii_dropped: false,
                                chunks: vec![Span::new(0, raw_text.len())],
                                verdicts: vec![ChunkVerdict {
                                    keep: false,
                                    strip: SmallVec::new(),
                                    reasons: {
                                        let mut r = ReasonSet::new();
                                        r.insert(Reason::LowScore);
                                        r
                                    },
                                    g: 1.0,
                                }],
                                ambiguous_chunk_indices: Vec::new(),
                                ambiguous_sentence_spans: Vec::new(),
                            },
                        }
                    })
                    .collect()
            });

            // Gather all ambiguous chunks across the batch
            let mut ambiguous_doc_indices = Vec::new();
            let mut ambiguous_chunk_indices = Vec::new();
            let mut ambiguous_texts = Vec::new();

            for (d_idx, doc) in doc_states.iter().enumerate() {
                for &c_idx in &doc.ambiguous_chunk_indices {
                    ambiguous_doc_indices.push(d_idx as u32);
                    ambiguous_chunk_indices.push(c_idx as u32);
                    let chunk_span = doc.chunks[c_idx];
                    let chunk_text = chunk_span.slice(&doc.orig_text).to_string();
                    ambiguous_texts.push(chunk_text);
                }
            }

            Ok(Stage1Batch {
                doc_states,
                compiled,
                ambiguous_doc_indices,
                ambiguous_chunk_indices,
                ambiguous_texts,
            })
        })
    }
}

fn process_single_doc(raw_text: &str, compiled: &Compiled) -> DocState {
    let cfg = &compiled.cfg;
    let norm_text: String = if cfg.normalize_unicode {
        raw_text.nfkc().collect()
    } else {
        raw_text.to_string()
    };

    let orig_len = norm_text.chars().count();
    let lang_info = detect(&norm_text, compiled);

    // PII scan & redact
    let pii_hits = scan(&norm_text, compiled);
    let pii_dropped = cfg.pii_mode == PiiMode::Drop && pii_hits.len() >= cfg.pii_drop_min_hits;
    let active_text = if cfg.pii_mode == PiiMode::Redact {
        redact(&norm_text, &pii_hits)
    } else {
        norm_text.clone()
    };

    // URLs and Sentences
    let urls = find_urls(&active_text, Some(&compiled.tlds));
    let sents = split_sentences(&active_text);

    // Per-sentence promo score and strip decisions
    let mut sent_scores = Vec::with_capacity(sents.len());
    let mut sent_strips: Vec<SmallVec<[Span; 2]>> = vec![SmallVec::new(); sents.len()];

    for (s_idx, &sent) in sents.iter().enumerate() {
        let sent_text = sent.slice(&active_text);
        let urls_in_sent: Vec<UrlSpan> = urls
            .iter()
            .filter(|u| u.start >= sent.start && u.end <= sent.end)
            .cloned()
            .collect();

        let s_score = score_sentence(sent_text, &urls_in_sent, compiled);
        sent_scores.push(s_score.score);

        // Definitive promo sentence: drop entire sentence
        if s_score.score >= cfg.promo_hi {
            sent_strips[s_idx].push(sent);
        } else {
            // URL actions
            let sent_url_only = urls_in_sent.len() == 1
                && urls_in_sent[0].start == sent.start
                && urls_in_sent[0].end == sent.end;

            for u in &urls_in_sent {
                let action = decide(u, s_score.score, sent_url_only, cfg);
                match action {
                    UrlAction::DropSentence => {
                        sent_strips[s_idx].push(sent);
                        break;
                    }
                    UrlAction::DeleteToken => {
                        sent_strips[s_idx].push(Span::new(u.start, u.end));
                    }
                    UrlAction::ReplaceWithAnchor => {
                        if let Some((a_start, a_end)) = u.anchor {
                            // Strip prefix before anchor
                            if a_start > u.start {
                                sent_strips[s_idx].push(Span::new(u.start, a_start));
                            }
                            // Strip suffix after anchor
                            if u.end > a_end {
                                sent_strips[s_idx].push(Span::new(a_end, u.end));
                            }
                        } else {
                            sent_strips[s_idx].push(Span::new(u.start, u.end));
                        }
                    }
                    UrlAction::Keep => {}
                }
            }
        }
    }

    // Chunking
    let chunks = chunk(&active_text, &sents, cfg);
    let mut verdicts = Vec::with_capacity(chunks.len());
    let mut ambiguous_chunk_indices = Vec::new();
    let mut ambiguous_sentence_spans = Vec::new();

    for &chunk_span in &chunks {
        let chunk_text = chunk_span.slice(&active_text);
        let features = crate::features::compute(chunk_text, lang_info.label, compiled);
        let (g_rule, mut rule_reasons) = crate::features::rule_garbage(&features, &cfg.gopher);

        // Find sentences falling into this chunk
        let mut chunk_strips: SmallVec<[Span; 4]> = SmallVec::new();
        let mut max_sent_promo = 0.0f32;
        let mut ambig_sents = Vec::new();

        for (s_idx, &sent) in sents.iter().enumerate() {
            if sent.start >= chunk_span.start && sent.end <= chunk_span.end {
                let p = sent_scores[s_idx];
                max_sent_promo = max_sent_promo.max(p);
                if p >= cfg.promo_lo && p < cfg.promo_hi {
                    ambig_sents.push(sent);
                }
                for &strip in &sent_strips[s_idx] {
                    chunk_strips.push(strip);
                }
            }
        }

        // Rule verdict:
        // A chunk is dropped (keep: false) if rule features indicate garbage (g_rule >= 0.85)
        // or if stripped spans encompass the entire chunk.
        // If a chunk contains promo sentence(s) but also clean content, keep chunk and strip promo.
        let strip_chars: usize = chunk_strips.iter().map(|s| s.len()).sum();
        let chunk_all_stripped = strip_chars >= chunk_span.len() && chunk_span.len() > 0;
        let is_garbage = g_rule >= 0.85 || chunk_all_stripped;
        let is_ambiguous = !is_garbage
            && ((max_sent_promo >= cfg.promo_lo && max_sent_promo < cfg.promo_hi)
                || (g_rule >= 0.40 && g_rule < 0.85));

        if max_sent_promo >= cfg.promo_hi {
            rule_reasons.insert(Reason::PromoPhrase);
        }

        if is_garbage {
            verdicts.push(ChunkVerdict {
                keep: false,
                strip: chunk_strips,
                reasons: rule_reasons,
                g: g_rule.max(max_sent_promo),
            });
            ambiguous_sentence_spans.push(Vec::new());
        } else if is_ambiguous {
            let c_idx = verdicts.len();
            ambiguous_chunk_indices.push(c_idx);
            verdicts.push(ChunkVerdict {
                keep: true,
                strip: chunk_strips,
                reasons: rule_reasons,
                g: g_rule.max(max_sent_promo),
            });
            ambiguous_sentence_spans.push(ambig_sents);
        } else {
            verdicts.push(ChunkVerdict {
                keep: true,
                strip: chunk_strips,
                reasons: rule_reasons,
                g: g_rule.max(max_sent_promo),
            });
            ambiguous_sentence_spans.push(Vec::new());
        }
    }

    DocState {
        orig_text: active_text,
        orig_len,
        lang_label: lang_info.label,
        lang_conf: lang_info.conf,
        pii_dropped,
        chunks,
        verdicts,
        ambiguous_chunk_indices,
        ambiguous_sentence_spans,
    }
}

#[pyfunction]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pyfunction]
pub fn cpu_features() -> Vec<String> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            features.push("avx2".to_string());
        }
        if is_x86_feature_detected!("fma") {
            features.push("fma".to_string());
        }
        if is_x86_feature_detected!("bmi1") {
            features.push("bmi1".to_string());
        }
        if is_x86_feature_detected!("bmi2") {
            features.push("bmi2".to_string());
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("sse4.1".to_string());
        }
        if is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2".to_string());
        }
        if is_x86_feature_detected!("avx512f") {
            features.push("avx512f".to_string());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        features.push("neon".to_string());
    }
    if features.is_empty() {
        features.push("generic".to_string());
    }
    features
}

#[pyfunction]
pub fn bench_stage1(texts: Vec<String>, cfg: &Bound<'_, PyDict>, seconds: f64) -> PyResult<f64> {
    Python::with_gil(|py| {
        let engine = Engine::new(cfg)?;
        let start = Instant::now();
        let mut total_docs = 0usize;

        while start.elapsed().as_secs_f64() < seconds {
            let batch = engine.stage1(py, texts.clone())?;
            total_docs += batch.n_docs();
        }

        let elapsed = start.elapsed().as_secs_f64().max(0.001);
        Ok(total_docs as f64 / elapsed)
    })
}
