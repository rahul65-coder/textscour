use crate::config::Config;
use crate::reasons::Reason;
use crate::urls::{host_matches, UrlKind, UrlSpan};
use aho_corasick::AhoCorasick;
use regex::Regex;
use smallvec::SmallVec;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct Lexicon {
    pub ac: AhoCorasick,
    pub weights: Vec<f32>,
    pub patterns: Vec<String>,
}

impl Lexicon {
    pub fn empty() -> Self {
        let ac = AhoCorasick::builder().build(&[""]).unwrap();
        Self {
            ac,
            weights: Vec::new(),
            patterns: Vec::new(),
        }
    }

    pub fn load_tsv<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| {
            format!(
                "Failed to open lexicon file {}: {}",
                path.as_ref().display(),
                e
            )
        })?;
        let reader = BufReader::new(file);

        let mut patterns = Vec::new();
        let mut weights = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("Error reading line {}: {}", line_idx + 1, e))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = trimmed.split('\t').collect();
            if parts.is_empty() {
                continue;
            }
            let phrase = parts[0].trim();
            if phrase.is_empty() {
                continue;
            }

            let weight = if parts.len() > 1 {
                parts[1].trim().parse::<f32>().unwrap_or(0.8)
            } else {
                0.8
            };

            let clamped_weight = weight.clamp(0.01, 1.0);
            patterns.push(phrase.to_string());
            weights.push(clamped_weight);
        }

        if patterns.is_empty() {
            return Ok(Self::empty());
        }

        let ac = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .map_err(|e| format!("Failed to build Aho-Corasick automaton: {}", e))?;

        Ok(Self {
            ac,
            weights,
            patterns,
        })
    }

    pub fn find_matches<'a>(&'a self, text: &'a str) -> Vec<(usize, usize, f32)> {
        if self.patterns.is_empty() {
            return Vec::new();
        }

        let mut hits = Vec::new();
        for m in self.ac.find_iter(text) {
            let start = m.start();
            let end = m.end();

            // Word-boundary check: left boundary
            if start > 0 {
                let prev = text[..start].chars().next_back().unwrap();
                if prev.is_alphanumeric() {
                    continue;
                }
            }

            // Word-boundary check: right boundary
            if end < text.len() {
                let next = text[end..].chars().next().unwrap();
                if next.is_alphanumeric() {
                    continue;
                }
            }

            let w = self.weights[m.pattern().as_usize()];
            hits.push((start, end, w));
        }

        hits
    }
}

pub struct Compiled {
    pub cfg: Config,
    pub promo_en: Lexicon,
    pub promo_hi: Lexicon,
    pub promo_hinglish: Lexicon,
    pub boilerplate_en: Lexicon,
    pub boilerplate_hi: Lexicon,
    pub orphan_refs: Lexicon,
    pub hinglish_words: hashbrown::HashSet<String>,
    pub tlds: Vec<String>,
    pub stopwords_en: hashbrown::HashSet<String>,
    pub stopwords_hi: hashbrown::HashSet<String>,
    pub topics: Vec<(String, Lexicon)>,
    pub allow_hosts: Vec<String>,
    pub block_hosts: Vec<String>,
    // Regexes
    pub price_discount_re: Regex,
    pub cta_start_re: Regex,
    pub promo_code_re: Regex,
    pub hashtag_run_re: Regex,
    pub phone_re: Regex,
}

impl Compiled {
    pub fn build(cfg: &Config) -> Result<Arc<Self>, String> {
        let mut dir = PathBuf::from(&cfg.lexicon_dir);
        if !dir.exists() {
            let parent = PathBuf::from("../../").join(&cfg.lexicon_dir);
            if parent.exists() {
                dir = parent;
            }
        }

        let promo_en = Lexicon::load_tsv(dir.join("promo_en.tsv"))?;
        let promo_hi = Lexicon::load_tsv(dir.join("promo_hi.tsv"))?;
        let promo_hinglish = Lexicon::load_tsv(dir.join("promo_hinglish.tsv"))?;
        let boilerplate_en = Lexicon::load_tsv(dir.join("boilerplate_en.tsv"))?;
        let boilerplate_hi = Lexicon::load_tsv(dir.join("boilerplate_hi.tsv"))?;
        let orphan_refs = Lexicon::load_tsv(dir.join("orphan_refs.tsv"))?;

        // Load hinglish_words.txt
        let mut hinglish_words = hashbrown::HashSet::new();
        if let Ok(file) = File::open(dir.join("hinglish_words.txt")) {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                let w = line.trim().to_ascii_lowercase();
                if !w.is_empty() && !w.starts_with('#') {
                    hinglish_words.insert(w);
                }
            }
        }

        // Load tlds.txt
        let mut tlds = Vec::new();
        if let Ok(file) = File::open(dir.join("tlds.txt")) {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                let t = line.trim().trim_start_matches('.').to_ascii_lowercase();
                if !t.is_empty() && !t.starts_with('#') {
                    tlds.push(t);
                }
            }
        }

        // Load stopwords_en.txt
        let mut stopwords_en = hashbrown::HashSet::new();
        if let Ok(file) = File::open(dir.join("stopwords_en.txt")) {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                let w = line.trim().to_ascii_lowercase();
                if !w.is_empty() && !w.starts_with('#') {
                    stopwords_en.insert(w);
                }
            }
        }

        // Load stopwords_hi.txt
        let mut stopwords_hi = hashbrown::HashSet::new();
        if let Ok(file) = File::open(dir.join("stopwords_hi.txt")) {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                let w = line.trim().to_string();
                if !w.is_empty() && !w.starts_with('#') {
                    stopwords_hi.insert(w);
                }
            }
        }

        // Load topics.tsv: format: topic<TAB>phrase<TAB>weight
        let mut topic_map: hashbrown::HashMap<String, (Vec<String>, Vec<f32>)> =
            hashbrown::HashMap::new();
        if let Ok(file) = File::open(dir.join("topics.tsv")) {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                let parts: Vec<&str> = trimmed.split('\t').collect();
                if parts.len() >= 2 {
                    let topic = parts[0].trim().to_ascii_lowercase();
                    let phrase = parts[1].trim().to_string();
                    let weight = if parts.len() > 2 {
                        parts[2].trim().parse::<f32>().unwrap_or(0.8)
                    } else {
                        0.8
                    };
                    let entry = topic_map
                        .entry(topic)
                        .or_insert_with(|| (Vec::new(), Vec::new()));
                    entry.0.push(phrase);
                    entry.1.push(weight);
                }
            }
        }

        let mut topics = Vec::new();
        for (topic, (patterns, weights)) in topic_map {
            if let Ok(ac) = AhoCorasick::builder()
                .ascii_case_insensitive(true)
                .build(&patterns)
            {
                topics.push((
                    topic,
                    Lexicon {
                        ac,
                        weights,
                        patterns,
                    },
                ));
            }
        }

        let price_discount_re = Regex::new(
            r"(?i)(?:₹|Rs\.?|\$|€|£)\s*\d+(?:,\d+)*(?:\.\d+)?|\b\d+%\s*(?:off|discount|percent|concession|cut|बचत|छूट|rupaye)?\b|\bflat\s+\d+%\b|\bstarting\s+(?:figures\s+sirf|rates|tariffs|at)\b|\bbargain\b|\bmarkdown\b|\bliquidation\b|\bभारी\s*कटौती\b|\bरियायत\b",
        )
        .unwrap();
        let cta_start_re = Regex::new(
            r"(?i)^(?:click|buy|shop|order|subscribe|follow|join|register|download|sign\s*up|learn\s*more|visit|unlock|tune|deploy|endorse|acquire|fetch|enroll|become|browse|dispatch|ping|forward|dial|examine|inspect|navigate|audit|get|install|launch|reserve\s+(?:your|a|now|today|seat|slot|table|ticket|hotel|room)|upgrade|hurry|don't\s*forget|hit\s*the|खरीदें|ऑर्डर|सब्सक्राइब|फॉलो|जुड़ें|मिलाएं|खोलें|देखें|मंगवाएं)\b",
        )
        .unwrap();
        let promo_code_re = Regex::new(
            r"(?i)\b(?:use|apply|enter|injecting|redeem|dalte|punch|प्रविष्ट)\s+(?:code|coupon|promo|token|voucher|secret|key|pass|कुंजी|सांकेतिक\s*कोड)\b|\b(?:promo|coupon|voucher)\s*(?:code|string|pass)?\b|\b(?:secret|token)\s+(?:code|string|pass|key)\b|\bसांकेतिक\s*कोड\b|\bगोपनीय\s*कुंजी\b",
        )
        .unwrap();
        let hashtag_run_re = Regex::new(r"(?:#[a-zA-Z0-9_]+\s*){3,}").unwrap();
        let phone_re = Regex::new(
            r"(?:\+91|0)?[6-9]\d{9}\b|\b1800[-.\s]?\d{3}[-.\s]?\d{4}\b|\b0\d{2,4}[-.\s]?\d{6,8}\b|\+\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        )
        .unwrap();

        let allow_hosts = cfg.allow_hosts.clone();
        let block_hosts = cfg.block_hosts.clone();

        Ok(Arc::new(Self {
            cfg: cfg.clone(),
            promo_en,
            promo_hi,
            promo_hinglish,
            boilerplate_en,
            boilerplate_hi,
            orphan_refs,
            hinglish_words,
            tlds,
            stopwords_en,
            stopwords_hi,
            topics,
            allow_hosts,
            block_hosts,
            price_discount_re,
            cta_start_re,
            promo_code_re,
            hashtag_run_re,
            phone_re,
        }))
    }
}

#[derive(Debug, Clone)]
pub struct SentenceScore {
    pub score: f32,
    pub hits: SmallVec<[(usize, usize, Reason); 8]>,
}

fn emoji_density(s: &str) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    let mut emoji_count = 0;
    let mut total_count = 0;
    for c in s.chars() {
        total_count += 1;
        // Emoji ranges
        if ('\u{1F300}'..='\u{1F9FF}').contains(&c)
            || ('\u{2600}'..='\u{26FF}').contains(&c)
            || ('\u{2700}'..='\u{27BF}').contains(&c)
        {
            emoji_count += 1;
        }
    }
    emoji_count as f32 / total_count as f32
}

fn allcaps_ratio(s: &str) -> (f32, usize) {
    let mut upper = 0;
    let mut letters = 0;
    for c in s.chars() {
        if c.is_alphabetic() {
            letters += 1;
            if c.is_uppercase() {
                upper += 1;
            }
        }
    }
    let ratio = if letters > 0 {
        upper as f32 / letters as f32
    } else {
        0.0
    };
    (ratio, letters)
}

pub fn score_sentence(s: &str, urls_in: &[UrlSpan], c: &Compiled) -> SentenceScore {
    let mut terms: SmallVec<[f32; 16]> = SmallVec::new();
    let mut hits: SmallVec<[(usize, usize, Reason); 8]> = SmallVec::new();

    // 1. Lexicon hits
    for (start, end, w) in c.promo_en.find_matches(s) {
        terms.push(w);
        hits.push((start, end, Reason::PromoPhrase));
    }
    for (start, end, w) in c.promo_hi.find_matches(s) {
        terms.push(w);
        hits.push((start, end, Reason::PromoPhrase));
    }
    for (start, end, w) in c.promo_hinglish.find_matches(s) {
        terms.push(w);
        hits.push((start, end, Reason::PromoPhrase));
    }
    for (start, end, w) in c.boilerplate_en.find_matches(s) {
        terms.push(w);
        hits.push((start, end, Reason::Boilerplate));
    }
    for (start, end, w) in c.boilerplate_hi.find_matches(s) {
        terms.push(w);
        hits.push((start, end, Reason::Boilerplate));
    }

    // 2. URL structural terms
    for u in urls_in {
        if u.kind == UrlKind::Shortener {
            terms.push(0.60);
            hits.push((u.start, u.end, Reason::UrlSpam));
        } else if !u.host.is_empty() && !host_matches(&u.host, &c.allow_hosts) {
            terms.push(0.35);
            hits.push((u.start, u.end, Reason::UrlSpam));
        }

        if u.kind == UrlKind::Handle {
            terms.push(0.30);
            hits.push((u.start, u.end, Reason::PromoHandle));
        }

        if u.kind == UrlKind::WhatsApp || u.kind == UrlKind::Telegram {
            terms.push(0.40);
            hits.push((u.start, u.end, Reason::PromoContact));
        }
    }

    // 3. Phone number
    if let Some(m) = c.phone_re.find(s) {
        terms.push(0.40);
        hits.push((m.start(), m.end(), Reason::PromoContact));
    }

    // 4. Price/discount pattern
    if let Some(m) = c.price_discount_re.find(s) {
        terms.push(0.35);
        hits.push((m.start(), m.end(), Reason::PromoPhrase));
    }

    // 5. CTA at sentence start
    let trimmed = s.trim_start();
    if let Some(m) = c.cta_start_re.find(trimmed) {
        let offset = s.len() - trimmed.len();
        terms.push(0.30);
        hits.push((offset + m.start(), offset + m.end(), Reason::CtaSentence));
    }

    // 6. Emoji density > 0.15
    if emoji_density(s) > 0.15 {
        terms.push(0.25);
        hits.push((0, s.len(), Reason::SymbolHeavy));
    }

    // 7. Hashtag run >= 3
    if let Some(m) = c.hashtag_run_re.find(s) {
        terms.push(0.40);
        hits.push((m.start(), m.end(), Reason::PromoPhrase));
    }

    // 8. ALLCAPS ratio > 0.60 with len >= 12
    let (upper_ratio, letter_count) = allcaps_ratio(s);
    if letter_count >= 12 && upper_ratio > 0.60 {
        terms.push(0.20);
        hits.push((0, s.len(), Reason::PromoPhrase));
    }

    // 9. Promo code pattern: "use code" / "promo code"
    if let Some(m) = c.promo_code_re.find(s) {
        terms.push(0.70);
        hits.push((m.start(), m.end(), Reason::PromoPhrase));
    }

    // Noisy-OR calculation: 1.0 - \prod (1.0 - w_i)
    let mut prod = 1.0f32;
    for &w in &terms {
        let clamped = w.clamp(0.0, 1.0);
        prod *= 1.0 - clamped;
    }
    let score = (1.0 - prod).clamp(0.0, 1.0);

    SentenceScore { score, hits }
}
