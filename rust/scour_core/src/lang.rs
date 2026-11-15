use crate::config::Lang;
use crate::promo::Compiled;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangInfo {
    pub label: Lang,
    pub conf: f32,
    pub devanagari_frac: f32,
    pub latin_frac: f32,
}

pub fn detect(text: &str, c: &Compiled) -> LangInfo {
    // Sample first 4 000 chars for speed
    let sample: String = text.chars().take(4000).collect();

    let mut devanagari_count = 0usize;
    let mut latin_count = 0usize;
    let mut total_letters = 0usize;

    for ch in sample.chars() {
        if ch.is_alphabetic() {
            total_letters += 1;
            if ('\u{0900}'..='\u{097F}').contains(&ch) {
                devanagari_count += 1;
            } else if ch.is_ascii_alphabetic() || ('\u{00C0}'..='\u{024F}').contains(&ch) {
                latin_count += 1;
            }
        }
    }

    if total_letters == 0 {
        return LangInfo {
            label: Lang::Other,
            conf: 0.5,
            devanagari_frac: 0.0,
            latin_frac: 0.0,
        };
    }

    let devanagari_frac = devanagari_count as f32 / total_letters as f32;
    let latin_frac = latin_count as f32 / total_letters as f32;

    // Rule 1: Devanagari >= 0.5 -> Hi
    if devanagari_frac >= 0.50 {
        let conf = devanagari_frac.clamp(0.5, 1.0);
        return LangInfo {
            label: Lang::Hi,
            conf,
            devanagari_frac,
            latin_frac,
        };
    }

    // Rule 2: Mixed script (Devanagari 0.1..0.5 with Latin) -> Hinglish
    if devanagari_frac >= 0.10 && latin_frac >= 0.20 {
        let conf = (devanagari_frac + latin_frac).clamp(0.5, 1.0);
        return LangInfo {
            label: Lang::Hinglish,
            conf,
            devanagari_frac,
            latin_frac,
        };
    }

    // Rule 3: Latin script >= 0.80 -> check English stopwords vs Hinglish wordlist
    if latin_frac >= 0.80 {
        let words: Vec<&str> = sample.split_whitespace().collect();
        let n_words = words.len().max(1);

        let mut en_stop_count = 0usize;
        let mut hing_count = 0usize;

        for w in &words {
            let clean: String = w
                .chars()
                .filter(|c| c.is_alphabetic())
                .flat_map(|c| c.to_lowercase())
                .collect();

            if clean.is_empty() {
                continue;
            }

            if c.stopwords_en.contains(&clean) {
                en_stop_count += 1;
            } else if c.hinglish_words.contains(&clean) {
                hing_count += 1;
            }
        }

        let en_stop = en_stop_count as f32 / n_words as f32;
        let hing_ratio = hing_count as f32 / n_words as f32;

        if hing_ratio >= 0.06 && hing_ratio >= 0.5 * en_stop {
            let conf = (0.5 + hing_ratio * 3.0).clamp(0.5, 1.0);
            return LangInfo {
                label: Lang::Hinglish,
                conf,
                devanagari_frac,
                latin_frac,
            };
        } else if en_stop >= 0.08 {
            let conf = (0.5 + en_stop * 2.5).clamp(0.5, 1.0);
            return LangInfo {
                label: Lang::En,
                conf,
                devanagari_frac,
                latin_frac,
            };
        } else {
            return LangInfo {
                label: Lang::Other,
                conf: 0.5,
                devanagari_frac,
                latin_frac,
            };
        }
    }

    LangInfo {
        label: Lang::Other,
        conf: 0.5,
        devanagari_frac,
        latin_frac,
    }
}
