use crate::config::{GopherThresholds, Lang};
use crate::promo::Compiled;
use crate::reasons::{Reason, ReasonSet};
use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct ChunkFeatures {
    pub alpha_ratio: f32,
    pub digit_ratio: f32,
    pub symbol_ratio: f32,
    pub upper_ratio: f32,
    pub avg_word_len: f32,
    pub avg_line_len: f32,
    pub short_line_frac: f32,
    pub dup_line_frac: f32,
    pub top_ngram_cov: [f32; 3], // 2, 3, 4-gram
    pub link_density: f32,
    pub stopword_ratio: f32,
    pub ellipsis_line_frac: f32,
    pub bullet_line_frac: f32,
    pub boilerplate_hits: u32,
    pub n_chars: usize,
    pub n_words: usize,
}

pub fn compute(chunk: &str, lang: Lang, c: &Compiled) -> ChunkFeatures {
    let n_chars = chunk.chars().count();
    if n_chars == 0 {
        return ChunkFeatures {
            alpha_ratio: 0.0,
            digit_ratio: 0.0,
            symbol_ratio: 0.0,
            upper_ratio: 0.0,
            avg_word_len: 0.0,
            avg_line_len: 0.0,
            short_line_frac: 0.0,
            dup_line_frac: 0.0,
            top_ngram_cov: [0.0; 3],
            link_density: 0.0,
            stopword_ratio: 0.0,
            ellipsis_line_frac: 0.0,
            bullet_line_frac: 0.0,
            boilerplate_hits: 0,
            n_chars: 0,
            n_words: 0,
        };
    }

    let mut alpha_count = 0usize;
    let mut digit_count = 0usize;
    let mut symbol_count = 0usize;
    let mut upper_count = 0usize;

    for ch in chunk.chars() {
        if ch.is_alphabetic() {
            alpha_count += 1;
            if ch.is_uppercase() {
                upper_count += 1;
            }
        } else if ch.is_numeric() {
            digit_count += 1;
        } else if !ch.is_whitespace() {
            symbol_count += 1;
        }
    }

    let alpha_ratio = alpha_count as f32 / n_chars as f32;
    let digit_ratio = digit_count as f32 / n_chars as f32;
    let symbol_ratio = symbol_count as f32 / n_chars as f32;
    let upper_ratio = if alpha_count > 0 {
        upper_count as f32 / alpha_count as f32
    } else {
        0.0
    };

    // Words
    let words: Vec<&str> = chunk.split_whitespace().collect();
    let n_words = words.len();
    let total_word_chars: usize = words.iter().map(|w| w.chars().count()).sum();
    let avg_word_len = if n_words > 0 {
        total_word_chars as f32 / n_words as f32
    } else {
        0.0
    };

    // Lines
    let lines: Vec<&str> = chunk.lines().collect();
    let n_lines = lines.len().max(1);
    let mut short_lines = 0;
    let mut ellipsis_lines = 0;
    let mut bullet_lines = 0;
    let mut non_empty_lines = Vec::new();

    for &line in &lines {
        let trimmed = line.trim();
        let line_chars = trimmed.chars().count();
        if line_chars < 40 {
            short_lines += 1;
        }
        if trimmed.ends_with("...") || trimmed.ends_with('…') {
            ellipsis_lines += 1;
        }
        if trimmed.starts_with("* ")
            || trimmed.starts_with("- ")
            || trimmed.starts_with("+ ")
            || trimmed.starts_with("• ")
        {
            bullet_lines += 1;
        }
        if !trimmed.is_empty() {
            non_empty_lines.push(trimmed);
        }
    }

    let avg_line_len = if !lines.is_empty() {
        n_chars as f32 / n_lines as f32
    } else {
        0.0
    };
    let short_line_frac = short_lines as f32 / n_lines as f32;
    let ellipsis_line_frac = ellipsis_lines as f32 / n_lines as f32;
    let bullet_line_frac = bullet_lines as f32 / n_lines as f32;

    // Duplicate line fraction
    let dup_line_frac = if non_empty_lines.len() > 1 {
        let mut line_counts = HashMap::new();
        for &l in &non_empty_lines {
            *line_counts.entry(l).or_insert(0usize) += 1;
        }
        let dupes: usize = line_counts.values().filter(|&&cnt| cnt > 1).sum();
        dupes as f32 / non_empty_lines.len() as f32
    } else {
        0.0
    };

    // Top N-gram coverage (2, 3, 4-grams of words)
    let mut top_ngram_cov = [0.0f32; 3];
    for (idx, &n) in [2usize, 3, 4].iter().enumerate() {
        if n_words >= n {
            let mut gram_counts = HashMap::new();
            for window in words.windows(n) {
                let gram = window.join(" ");
                *gram_counts.entry(gram).or_insert(0usize) += 1;
            }
            if let Some(&max_cnt) = gram_counts.values().max() {
                if max_cnt > 1 {
                    top_ngram_cov[idx] = (max_cnt * n) as f32 / n_words as f32;
                }
            }
        }
    }

    // Stopwords ratio
    let stopword_ratio = if n_words > 0 {
        let mut stop_count = 0;
        let stopwords = match lang {
            Lang::Hi => &c.stopwords_hi,
            _ => &c.stopwords_en,
        };
        for w in &words {
            let clean_w = w
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_ascii_lowercase();
            if stopwords.contains(&clean_w)
                || (lang == Lang::Hinglish && c.hinglish_words.contains(&clean_w))
            {
                stop_count += 1;
            }
        }
        stop_count as f32 / n_words as f32
    } else {
        0.0
    };

    // Boilerplate hits
    let mut boilerplate_hits = 0u32;
    boilerplate_hits += c.boilerplate_en.find_matches(chunk).len() as u32;
    boilerplate_hits += c.boilerplate_hi.find_matches(chunk).len() as u32;

    // Link density
    let urls = crate::urls::find_urls(chunk, Some(&c.tlds));
    let url_chars: usize = urls.iter().map(|u| u.end - u.start).sum();
    let link_density = if n_chars > 0 {
        url_chars as f32 / n_chars as f32
    } else {
        0.0
    };

    ChunkFeatures {
        alpha_ratio,
        digit_ratio,
        symbol_ratio,
        upper_ratio,
        avg_word_len,
        avg_line_len,
        short_line_frac,
        dup_line_frac,
        top_ngram_cov,
        link_density,
        stopword_ratio,
        ellipsis_line_frac,
        bullet_line_frac,
        boilerplate_hits,
        n_chars,
        n_words,
    }
}

pub fn rule_garbage(f: &ChunkFeatures, g: &GopherThresholds) -> (f32, ReasonSet) {
    let mut reasons = ReasonSet::new();
    let mut max_margin = 0.0f32;

    // 1. alpha_ratio < min_alpha_ratio (0.50)
    if f.alpha_ratio < g.min_alpha_ratio {
        reasons.insert(Reason::LowAlpha);
        let margin = (g.min_alpha_ratio - f.alpha_ratio) / g.min_alpha_ratio;
        max_margin = max_margin.max(margin);
    }

    // 2. symbol_ratio > max_symbol_ratio (0.10)
    if f.symbol_ratio > g.max_symbol_ratio {
        reasons.insert(Reason::SymbolHeavy);
        let margin = (f.symbol_ratio - g.max_symbol_ratio) / (1.0 - g.max_symbol_ratio);
        max_margin = max_margin.max(margin);
    }

    // 3. dup_line_frac > max_dup_line_frac (0.30)
    if f.dup_line_frac > g.max_dup_line_frac {
        reasons.insert(Reason::Repetition);
        let margin = (f.dup_line_frac - g.max_dup_line_frac) / (1.0 - g.max_dup_line_frac);
        max_margin = max_margin.max(margin);
    }

    // 4. top n-gram coverage
    if f.top_ngram_cov[0] > g.max_top_2gram_cov {
        reasons.insert(Reason::Repetition);
        let margin = (f.top_ngram_cov[0] - g.max_top_2gram_cov) / (1.0 - g.max_top_2gram_cov);
        max_margin = max_margin.max(margin);
    }
    if f.top_ngram_cov[1] > g.max_top_3gram_cov {
        reasons.insert(Reason::Repetition);
        let margin = (f.top_ngram_cov[1] - g.max_top_3gram_cov) / (1.0 - g.max_top_3gram_cov);
        max_margin = max_margin.max(margin);
    }
    if f.top_ngram_cov[2] > g.max_top_4gram_cov {
        reasons.insert(Reason::Repetition);
        let margin = (f.top_ngram_cov[2] - g.max_top_4gram_cov) / (1.0 - g.max_top_4gram_cov);
        max_margin = max_margin.max(margin);
    }

    // 5. short_line_frac > 0.70 && avg_line_len < 40.0
    if f.short_line_frac > g.max_short_line_frac && f.avg_line_len < g.min_avg_line_len {
        reasons.insert(Reason::NavMenu);
        let margin = (f.short_line_frac - g.max_short_line_frac) / (1.0 - g.max_short_line_frac);
        max_margin = max_margin.max(margin);
    }

    // 6. boilerplate_hits >= max_boilerplate_hits (2)
    if f.boilerplate_hits >= g.max_boilerplate_hits {
        reasons.insert(Reason::Boilerplate);
        let margin = (f.boilerplate_hits as f32 / 4.0).min(1.0);
        max_margin = max_margin.max(margin);
    }

    // 7. stopword_ratio < min_stopword_ratio (0.06)
    if f.stopword_ratio < g.min_stopword_ratio {
        reasons.insert(Reason::LowAlpha);
        let margin = (g.min_stopword_ratio - f.stopword_ratio) / g.min_stopword_ratio;
        max_margin = max_margin.max(margin);
    }

    (max_margin.clamp(0.0, 1.0), reasons)
}
