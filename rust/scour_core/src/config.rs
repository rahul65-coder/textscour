use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UrlPolicy {
    Smart,
    RemoveAll,
    AllowlistOnly,
    KeepAll,
    Blocklist,
}

impl UrlPolicy {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "smart" => Ok(UrlPolicy::Smart),
            "remove_all" => Ok(UrlPolicy::RemoveAll),
            "allowlist_only" => Ok(UrlPolicy::AllowlistOnly),
            "keep_all" => Ok(UrlPolicy::KeepAll),
            "blocklist" => Ok(UrlPolicy::Blocklist),
            _ => Err(format!("Unknown url_policy: '{}'", s)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            UrlPolicy::Smart => "smart",
            UrlPolicy::RemoveAll => "remove_all",
            UrlPolicy::AllowlistOnly => "allowlist_only",
            UrlPolicy::KeepAll => "keep_all",
            UrlPolicy::Blocklist => "blocklist",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiMode {
    Redact,
    Drop,
    Off,
}

impl PiiMode {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "redact" => Ok(PiiMode::Redact),
            "drop" => Ok(PiiMode::Drop),
            "off" => Ok(PiiMode::Off),
            _ => Err(format!("Unknown pii_mode: '{}'", s)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            PiiMode::Redact => "redact",
            PiiMode::Drop => "drop",
            PiiMode::Off => "off",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainMode {
    Field,
    UrlHost,
    Dataset,
    Topic,
}

impl DomainMode {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "field" => Ok(DomainMode::Field),
            "url_host" => Ok(DomainMode::UrlHost),
            "dataset" => Ok(DomainMode::Dataset),
            "topic" => Ok(DomainMode::Topic),
            _ => Err(format!("Unknown domain_mode: '{}'", s)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DomainMode::Field => "field",
            DomainMode::UrlHost => "url_host",
            DomainMode::Dataset => "dataset",
            DomainMode::Topic => "topic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lang {
    En,
    Hi,
    Hinglish,
    Other,
}

impl Lang {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "en" => Ok(Lang::En),
            "hi" => Ok(Lang::Hi),
            "hinglish" => Ok(Lang::Hinglish),
            "other" => Ok(Lang::Other),
            _ => Err(format!("Unknown lang: '{}'", s)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Hi => "hi",
            Lang::Hinglish => "hinglish",
            Lang::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GopherThresholds {
    pub min_alpha_ratio: f32,
    pub max_symbol_ratio: f32,
    pub max_dup_line_frac: f32,
    pub max_top_2gram_cov: f32,
    pub max_top_3gram_cov: f32,
    pub max_top_4gram_cov: f32,
    pub max_short_line_frac: f32,
    pub min_avg_line_len: f32,
    pub max_boilerplate_hits: u32,
    pub min_stopword_ratio: f32,
}

impl Default for GopherThresholds {
    fn default() -> Self {
        Self {
            min_alpha_ratio: 0.50,
            max_symbol_ratio: 0.10,
            max_dup_line_frac: 0.30,
            max_top_2gram_cov: 0.20,
            max_top_3gram_cov: 0.18,
            max_top_4gram_cov: 0.16,
            max_short_line_frac: 0.70,
            min_avg_line_len: 40.0,
            max_boilerplate_hits: 2,
            min_stopword_ratio: 0.06,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub chunk_target_chars: usize,
    pub chunk_min_chars: usize,
    pub chunk_max_chars: usize,
    pub url_policy: UrlPolicy,
    pub allow_hosts: Vec<String>,
    pub block_hosts: Vec<String>,
    pub promo_lo: f32,
    pub promo_hi: f32,
    pub gpu_drop_p: f32,
    pub gpu_strip_p: f32,
    pub min_doc_chars: usize,
    pub clean_threshold: f32,
    pub pii_mode: PiiMode,
    pub pii_drop_min_hits: usize,
    pub allowed_langs: Vec<Lang>,
    pub rust_threads: usize,
    pub lexicon_dir: String,
    pub gopher: GopherThresholds,
    pub domain_mode: DomainMode,
    pub emit_fragments: bool,
    pub normalize_unicode: bool,
    pub dedup_shingle_k: usize,
    pub keep_urls_in_code: bool,
    pub max_doc_chars: usize,
}

pub fn default_allow_hosts() -> Vec<String> {
    vec![
        "wikipedia.org".to_string(),
        "wikimedia.org".to_string(),
        "github.com".to_string(),
        "arxiv.org".to_string(),
        "doi.org".to_string(),
        "huggingface.co".to_string(),
        "stackoverflow.com".to_string(),
        "python.org".to_string(),
        "docs.python.org".to_string(),
        "kaggle.com".to_string(),
        "nih.gov".to_string(),
        "who.int".to_string(),
        "*.gov.in".to_string(),
        "*.edu".to_string(),
        "readthedocs.io".to_string(),
        "mozilla.org".to_string(),
        "w3.org".to_string(),
        "ietf.org".to_string(),
        "pypi.org".to_string(),
        "npmjs.com".to_string(),
    ]
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_target_chars: 800,
            chunk_min_chars: 100,
            chunk_max_chars: 2000,
            url_policy: UrlPolicy::Smart,
            allow_hosts: default_allow_hosts(),
            block_hosts: Vec::new(),
            promo_lo: 0.30,
            promo_hi: 0.70,
            gpu_drop_p: 0.85,
            gpu_strip_p: 0.50,
            min_doc_chars: 200,
            clean_threshold: 6.0,
            pii_mode: PiiMode::Redact,
            pii_drop_min_hits: 1,
            allowed_langs: vec![Lang::En, Lang::Hi, Lang::Hinglish],
            rust_threads: 0,
            lexicon_dir: "configs/lexicons".to_string(),
            gopher: GopherThresholds::default(),
            domain_mode: DomainMode::Dataset,
            emit_fragments: false,
            normalize_unicode: true,
            dedup_shingle_k: 5,
            keep_urls_in_code: true,
            max_doc_chars: 2_000_000,
        }
    }
}

impl Config {
    pub fn from_pydict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut cfg = Config::default();

        if let Some(v) = d.get_item("chunk_target_chars")? {
            cfg.chunk_target_chars = v.extract()?;
        }
        if let Some(v) = d.get_item("chunk_min_chars")? {
            cfg.chunk_min_chars = v.extract()?;
        }
        if let Some(v) = d.get_item("chunk_max_chars")? {
            cfg.chunk_max_chars = v.extract()?;
        }
        if let Some(v) = d.get_item("url_policy")? {
            let s: String = v.extract()?;
            cfg.url_policy = UrlPolicy::from_str(&s).map_err(|e| PyValueError::new_err(e))?;
        }
        if let Some(v) = d.get_item("allow_hosts")? {
            let hosts: Vec<String> = v.extract()?;
            if !hosts.is_empty() {
                cfg.allow_hosts = hosts;
            }
        }
        if let Some(v) = d.get_item("block_hosts")? {
            cfg.block_hosts = v.extract()?;
        }
        if let Some(v) = d.get_item("promo_lo")? {
            cfg.promo_lo = v.extract()?;
        }
        if let Some(v) = d.get_item("promo_hi")? {
            cfg.promo_hi = v.extract()?;
        }
        if let Some(v) = d.get_item("gpu_drop_p")? {
            cfg.gpu_drop_p = v.extract()?;
        }
        if let Some(v) = d.get_item("gpu_strip_p")? {
            cfg.gpu_strip_p = v.extract()?;
        }
        if let Some(v) = d.get_item("min_doc_chars")? {
            cfg.min_doc_chars = v.extract()?;
        }
        if let Some(v) = d.get_item("clean_threshold")? {
            cfg.clean_threshold = v.extract()?;
        }
        if let Some(v) = d.get_item("pii_mode")? {
            let s: String = v.extract()?;
            cfg.pii_mode = PiiMode::from_str(&s).map_err(|e| PyValueError::new_err(e))?;
        }
        if let Some(v) = d.get_item("pii_drop_min_hits")? {
            cfg.pii_drop_min_hits = v.extract()?;
        }
        if let Some(v) = d.get_item("allowed_langs")? {
            let langs: Vec<String> = v.extract()?;
            let mut parsed_langs = Vec::new();
            for l in langs {
                parsed_langs.push(Lang::from_str(&l).map_err(|e| PyValueError::new_err(e))?);
            }
            cfg.allowed_langs = parsed_langs;
        }
        if let Some(v) = d.get_item("rust_threads")? {
            cfg.rust_threads = v.extract()?;
        }
        if let Some(v) = d.get_item("lexicon_dir")? {
            cfg.lexicon_dir = v.extract()?;
        }
        if let Some(v) = d.get_item("domain_mode")? {
            let s: String = v.extract()?;
            cfg.domain_mode = DomainMode::from_str(&s).map_err(|e| PyValueError::new_err(e))?;
        }
        if let Some(v) = d.get_item("emit_fragments")? {
            cfg.emit_fragments = v.extract()?;
        }
        if let Some(v) = d.get_item("normalize_unicode")? {
            cfg.normalize_unicode = v.extract()?;
        }
        if let Some(v) = d.get_item("dedup_shingle_k")? {
            cfg.dedup_shingle_k = v.extract()?;
        }
        if let Some(v) = d.get_item("keep_urls_in_code")? {
            cfg.keep_urls_in_code = v.extract()?;
        }
        if let Some(v) = d.get_item("max_doc_chars")? {
            cfg.max_doc_chars = v.extract()?;
        }

        if let Some(v) = d.get_item("gopher")? {
            if let Ok(g_dict) = v.downcast::<PyDict>() {
                if let Some(gv) = g_dict.get_item("min_alpha_ratio")? {
                    cfg.gopher.min_alpha_ratio = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("max_symbol_ratio")? {
                    cfg.gopher.max_symbol_ratio = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("max_dup_line_frac")? {
                    cfg.gopher.max_dup_line_frac = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("max_top_2gram_cov")? {
                    cfg.gopher.max_top_2gram_cov = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("max_top_3gram_cov")? {
                    cfg.gopher.max_top_3gram_cov = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("max_top_4gram_cov")? {
                    cfg.gopher.max_top_4gram_cov = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("max_short_line_frac")? {
                    cfg.gopher.max_short_line_frac = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("min_avg_line_len")? {
                    cfg.gopher.min_avg_line_len = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("max_boilerplate_hits")? {
                    cfg.gopher.max_boilerplate_hits = gv.extract()?;
                }
                if let Some(gv) = g_dict.get_item("min_stopword_ratio")? {
                    cfg.gopher.min_stopword_ratio = gv.extract()?;
                }
            }
        }

        cfg.validate().map_err(|e| PyValueError::new_err(e))?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.promo_lo >= self.promo_hi {
            return Err(format!(
                "promo_lo ({}) must be strictly less than promo_hi ({})",
                self.promo_lo, self.promo_hi
            ));
        }
        if self.gpu_strip_p > self.gpu_drop_p {
            return Err(format!(
                "gpu_strip_p ({}) must be <= gpu_drop_p ({})",
                self.gpu_strip_p, self.gpu_drop_p
            ));
        }
        if self.chunk_min_chars > self.chunk_target_chars
            || self.chunk_target_chars > self.chunk_max_chars
        {
            return Err(format!(
                "Chunk sizes must satisfy chunk_min_chars ({}) <= chunk_target_chars ({}) <= chunk_max_chars ({})",
                self.chunk_min_chars, self.chunk_target_chars, self.chunk_max_chars
            ));
        }
        if self.min_doc_chars == 0 {
            return Err("min_doc_chars must be greater than 0".to_string());
        }
        if self.clean_threshold < 0.0 || self.clean_threshold > 10.0 {
            return Err(format!(
                "clean_threshold ({}) must be in [0.0, 10.0]",
                self.clean_threshold
            ));
        }
        if self.allowed_langs.is_empty() {
            return Err("allowed_langs must contain at least one language".to_string());
        }

        Ok(())
    }

    pub fn validate_lexicons(&self) -> Result<(), String> {
        if self.lexicon_dir.is_empty() || self.lexicon_dir == ":memory:" {
            return Ok(());
        }
        let dir = Path::new(&self.lexicon_dir);
        if !dir.exists() {
            return Err(format!(
                "Lexicon directory does not exist: {}",
                self.lexicon_dir
            ));
        }
        let required_files = [
            "promo_en.tsv",
            "promo_hi.tsv",
            "promo_hinglish.tsv",
            "boilerplate_en.tsv",
            "boilerplate_hi.tsv",
            "hinglish_words.txt",
            "tlds.txt",
            "stopwords_en.txt",
            "stopwords_hi.txt",
            "orphan_refs.tsv",
            "topics.tsv",
        ];
        for file in &required_files {
            let p = dir.join(file);
            if !p.exists() {
                return Err(format!("Lexicon file missing: {}", p.display()));
            }
        }
        Ok(())
    }
}
