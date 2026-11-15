use crate::chunker::Span;
use crate::promo::Compiled;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiKind {
    Email,
    Phone,
    Aadhaar,
    Pan,
    Ip,
    Key,
    Card,
}

impl PiiKind {
    pub fn token(&self) -> &'static str {
        match self {
            PiiKind::Email => "<EMAIL>",
            PiiKind::Phone => "<PHONE>",
            PiiKind::Aadhaar => "<AADHAAR>",
            PiiKind::Pan => "<PAN>",
            PiiKind::Ip => "<IP>",
            PiiKind::Key => "<KEY>",
            PiiKind::Card => "<CARD>",
        }
    }
}

// Verhoeff tables
const D_TABLE: [[u8; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

const P_TABLE: [[u8; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

const INV_TABLE: [u8; 10] = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];

pub fn compute_verhoeff_check_digit(num_str: &str) -> u8 {
    let mut c = 0usize;
    let digits: Vec<u32> = num_str.chars().filter_map(|ch| ch.to_digit(10)).collect();
    for (i, &d) in digits.iter().rev().enumerate() {
        let p_val = P_TABLE[(i + 1) % 8][d as usize] as usize;
        c = D_TABLE[c][p_val] as usize;
    }
    INV_TABLE[c]
}

pub fn validate_verhoeff(num_str: &str) -> bool {
    let mut c = 0usize;
    let digits: Vec<u32> = num_str.chars().filter_map(|ch| ch.to_digit(10)).collect();
    if digits.is_empty() {
        return false;
    }
    for (i, &d) in digits.iter().rev().enumerate() {
        let p_val = P_TABLE[i % 8][d as usize] as usize;
        c = D_TABLE[c][p_val] as usize;
    }
    c == 0
}

pub fn validate_luhn(num_str: &str) -> bool {
    let digits: Vec<u32> = num_str.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let mut sum = 0;
    let mut alt = false;
    for &d in digits.iter().rev() {
        if alt {
            let mut doubled = d * 2;
            if doubled > 9 {
                doubled -= 9;
            }
            sum += doubled;
        } else {
            sum += d;
        }
        alt = !alt;
    }
    sum % 10 == 0
}

static EMAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b").unwrap()
});

static PHONE_IN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:(?:\+91|0)[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b").unwrap());

static PHONE_INTL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\+\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b").unwrap()
});

static AADHAAR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b").unwrap());

static PAN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b").unwrap());

static IPV4_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap());

static KEY_PATTERNS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]+)\b").unwrap()
});

static CARD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:\d{4}[\s-]?){3}\d{1,7}\b|\b\d{13,19}\b").unwrap());

pub fn scan(text: &str, _c: &Compiled) -> Vec<(Span, PiiKind)> {
    let mut hits = Vec::new();

    // 1. Email
    for m in EMAIL_RE.find_iter(text) {
        hits.push((Span::new(m.start(), m.end()), PiiKind::Email));
    }

    // 2. PAN (check before Aadhaar or others to avoid clash)
    for m in PAN_RE.find_iter(text) {
        hits.push((Span::new(m.start(), m.end()), PiiKind::Pan));
    }

    // 3. Secrets / API Keys
    for m in KEY_PATTERNS_RE.find_iter(text) {
        hits.push((Span::new(m.start(), m.end()), PiiKind::Key));
    }

    // 4. Aadhaar (12 digits, grouped 4-4-4, Verhoeff validated)
    for m in AADHAAR_RE.find_iter(text) {
        let raw = m.as_str();
        let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == 12 && validate_verhoeff(&digits) {
            hits.push((Span::new(m.start(), m.end()), PiiKind::Aadhaar));
        }
    }

    // 5. Credit/Debit Card (13-19 digits, Luhn validated)
    for m in CARD_RE.find_iter(text) {
        let raw = m.as_str();
        let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 13 && digits.len() <= 19 && validate_luhn(&digits) {
            hits.push((Span::new(m.start(), m.end()), PiiKind::Card));
        }
    }

    // 6. Indian & International Phone numbers
    for m in PHONE_IN_RE.find_iter(text) {
        let raw = m.as_str();
        let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 10 && digits.len() <= 12 {
            hits.push((Span::new(m.start(), m.end()), PiiKind::Phone));
        }
    }
    for m in PHONE_INTL_RE.find_iter(text) {
        let raw = m.as_str();
        let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 8 && digits.len() <= 15 {
            hits.push((Span::new(m.start(), m.end()), PiiKind::Phone));
        }
    }

    // 7. IPv4 addresses (octets <= 255, excluding version numbers)
    for m in IPV4_RE.find_iter(text) {
        let raw = m.as_str();
        let parts: Vec<&str> = raw.split('.').collect();
        if parts.len() == 4 {
            let valid_octets = parts.iter().all(|p| {
                if let Ok(num) = p.parse::<u16>() {
                    num <= 255
                } else {
                    false
                }
            });

            if valid_octets {
                // Context check to reject version strings: "version 1.2.3.4", "v1.2.3.4"
                let start = m.start();
                let mut ctx_start = start.saturating_sub(10);
                while ctx_start > 0 && !text.is_char_boundary(ctx_start) {
                    ctx_start -= 1;
                }
                let before = &text[ctx_start..start];
                let before_lower = before.to_ascii_lowercase();
                if before_lower.ends_with("version ")
                    || before_lower.ends_with("v")
                    || before_lower.ends_with("release ")
                {
                    continue;
                }
                hits.push((Span::new(m.start(), m.end()), PiiKind::Ip));
            }
        }
    }

    // Sort hits by start position
    hits.sort_by_key(|h| h.0.start);

    // Dedup overlaps: keep earlier / more specific hit
    let mut deduped: Vec<(Span, PiiKind)> = Vec::with_capacity(hits.len());
    for hit in hits {
        if let Some(last) = deduped.last_mut() {
            if hit.0.start < last.0.end {
                // Overlap: prefer Card/Aadhaar/Key over Phone
                if matches!(hit.1, PiiKind::Card | PiiKind::Aadhaar | PiiKind::Key)
                    && matches!(last.1, PiiKind::Phone)
                {
                    *last = hit;
                }
                continue;
            }
        }
        deduped.push(hit);
    }

    deduped
}

pub fn redact(text: &str, hits: &[(Span, PiiKind)]) -> String {
    if hits.is_empty() {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut cur = 0;

    for (span, kind) in hits {
        if span.start >= cur && span.end <= text.len() {
            out.push_str(&text[cur..span.start]);
            out.push_str(kind.token());
            cur = span.end;
        }
    }

    if cur < text.len() {
        out.push_str(&text[cur..]);
    }

    out
}
