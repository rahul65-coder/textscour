use crate::config::Config;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[inline]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    #[inline]
    pub fn slice<'a>(&self, text: &'a str) -> &'a str {
        let mut start = self.start.min(text.len());
        let mut end = self.end.min(text.len());
        if start > end {
            start = end;
        }
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        &text[start..end]
    }
}

pub static ABBREVIATIONS: &[&str] = &[
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "rs.",
    "no.", "st.", "inc.", "ltd.", "co.", "corp.", "univ.", "jan.", "feb.", "mar.", "apr.", "jun.",
    "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.", "approx.", "govt.", "dept.", "ph.d.",
    "m.d.", "b.a.", "b.s.", "m.a.", "m.s.",
];

fn is_abbreviation(text: &str, dot_idx: usize) -> bool {
    if dot_idx >= text.len() || !text.is_char_boundary(dot_idx) {
        return false;
    }
    let before_dot = &text[..dot_idx];
    let word_start = before_dot
        .rfind(|c: char| c.is_whitespace() || c == '(' || c == '[' || c == '"' || c == '\'')
        .map(|idx| {
            let ch = before_dot[idx..].chars().next().unwrap();
            idx + ch.len_utf8()
        })
        .unwrap_or(0);

    let mut safe_word_start = word_start;
    while safe_word_start < dot_idx && !text.is_char_boundary(safe_word_start) {
        safe_word_start += 1;
    }
    if safe_word_start > dot_idx {
        safe_word_start = dot_idx;
    }

    let word_with_dot = format!("{}.", &text[safe_word_start..dot_idx].to_ascii_lowercase());
    if ABBREVIATIONS.contains(&word_with_dot.as_str()) {
        return true;
    }

    // Single capital letter initial (e.g. "J.")
    let word = &text[safe_word_start..dot_idx];
    if word.chars().count() == 1 {
        let ch = word.chars().next().unwrap();
        if ch.is_alphabetic() {
            return true;
        }
    }

    // Number with decimal point: e.g. "3.14"
    if let Some(prev) = before_dot.chars().next_back() {
        if prev.is_ascii_digit() {
            let next_slice_start = dot_idx + 1;
            if next_slice_start < text.len() && text.is_char_boundary(next_slice_start) {
                if let Some(next) = text[next_slice_start..].chars().next() {
                    if next.is_ascii_digit() {
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn is_bullet_start(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("* ")
        || trimmed.starts_with("- ")
        || trimmed.starts_with("+ ")
        || trimmed.starts_with("• ")
        || trimmed.starts_with("– ")
        || trimmed.starts_with("— ")
    {
        return true;
    }

    // Check "1. " or "1) "
    if let Some(first) = trimmed.chars().next() {
        if first.is_ascii_digit() {
            let rest = &trimmed[first.len_utf8()..];
            let after_digits = rest.trim_start_matches(|c: char| c.is_ascii_digit());
            if after_digits.starts_with(". ") || after_digits.starts_with(") ") {
                return true;
            }
        }
    }

    false
}

pub fn split_sentences(text: &str) -> Vec<Span> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut spans = Vec::new();
    let mut sent_start = 0;
    let bytes = text.as_bytes();
    let len = text.len();

    let mut i = 0;
    while i < len {
        // Check paragraph break: "\n\n"
        if bytes[i] == b'\n' {
            if i + 1 < len && bytes[i + 1] == b'\n' {
                // Skip following newlines
                let mut end = i + 2;
                while end < len && bytes[end] == b'\n' {
                    end += 1;
                }
                if end > sent_start {
                    spans.push(Span::new(sent_start, end));
                    sent_start = end;
                    i = end;
                    continue;
                }
            }

            // Check if next line is a bullet
            let next_line_start = i + 1;
            if next_line_start < len && next_line_start > sent_start {
                let rest_line = &text[next_line_start..];
                if is_bullet_start(rest_line) {
                    spans.push(Span::new(sent_start, next_line_start));
                    sent_start = next_line_start;
                    i = next_line_start;
                    continue;
                }
            }
        }

        // Check sentence terminator characters:
        // . ! ? । (U+0964, e0 a5 a4) ॥ (U+0965, e0 a5 a5) … (U+2026, e2 80 a6)
        let ch = text[i..].chars().next().unwrap();
        let ch_len = ch.len_utf8();

        let is_term = match ch {
            '।' | '॥' => true,
            '!' | '?' | '…' => {
                if i + ch_len >= len {
                    true
                } else {
                    let next = text[i + ch_len..].chars().next().unwrap();
                    next.is_whitespace()
                        || matches!(next, '"' | '\'' | ')' | ']' | '}' | '”' | '’' | '»')
                }
            }
            '.' => {
                if is_abbreviation(text, i) {
                    false
                } else if i + 1 < len {
                    let next = text[i + 1..].chars().next().unwrap();
                    if next.is_alphanumeric() {
                        false
                    } else {
                        next.is_whitespace()
                            || matches!(next, '"' | '\'' | ')' | ']' | '}' | '”' | '’' | '»')
                    }
                } else {
                    true
                }
            }
            _ => false,
        };

        if is_term {
            let mut end = i + ch_len;
            // Absorb any following matching terminators, quotes, parens, brackets, whitespace
            while end < len {
                let next_ch = text[end..].chars().next().unwrap();
                if matches!(
                    next_ch,
                    '!' | '?'
                        | '.'
                        | '।'
                        | '॥'
                        | '…'
                        | '"'
                        | '\''
                        | ')'
                        | ']'
                        | '}'
                        | '”'
                        | '’'
                        | '»'
                ) {
                    end += next_ch.len_utf8();
                } else {
                    break;
                }
            }

            let has_following_ws_or_eof = if end >= len {
                true
            } else {
                let next_ch = text[end..].chars().next().unwrap();
                next_ch.is_whitespace()
            };

            if !has_following_ws_or_eof && ch == '.' {
                i += ch_len;
                continue;
            }

            // Consume trailing single spaces or single newline that separates sentences
            while end < len {
                let next_ch = text[end..].chars().next().unwrap();
                if next_ch == ' ' || next_ch == '\t' {
                    end += next_ch.len_utf8();
                } else if next_ch == '\n' && end + 1 < len && bytes[end + 1] != b'\n' {
                    end += 1;
                    break;
                } else {
                    break;
                }
            }

            if end > sent_start {
                spans.push(Span::new(sent_start, end));
                sent_start = end;
                i = end;
                continue;
            }
        }

        i += ch_len;
    }

    if sent_start < len {
        spans.push(Span::new(sent_start, len));
    }

    // Fallback if empty
    if spans.is_empty() {
        spans.push(Span::new(0, len));
    }

    spans
}

pub fn chunk(text: &str, sents: &[Span], cfg: &Config) -> Vec<Span> {
    if text.is_empty() {
        return Vec::new();
    }
    if sents.is_empty() {
        return vec![Span::new(0, text.len())];
    }

    // Step 1: Flatten sentences and subdivide any sentence that exceeds chunk_max_chars
    let mut units: Vec<Span> = Vec::new();
    for &s in sents {
        if s.len() <= cfg.chunk_max_chars {
            units.push(s);
        } else {
            // Subdivide s into slices <= chunk_max_chars
            let mut cur_start = s.start;
            while cur_start < s.end {
                let remaining = s.end - cur_start;
                if remaining <= cfg.chunk_max_chars {
                    units.push(Span::new(cur_start, s.end));
                    break;
                }

                // Look for nearest whitespace before chunk_max_chars
                let mut max_boundary = (cur_start + cfg.chunk_max_chars).min(s.end);
                while !text.is_char_boundary(max_boundary) && max_boundary > cur_start {
                    max_boundary -= 1;
                }
                let sub = &text[cur_start..max_boundary];

                let split_offset = sub.rfind(|c: char| c.is_whitespace()).map(|idx| {
                    // include the whitespace in this slice
                    let ws_char = sub[idx..].chars().next().unwrap();
                    idx + ws_char.len_utf8()
                });

                let split_pt = match split_offset {
                    Some(off) if off > 0 => cur_start + off,
                    _ => {
                        // No whitespace found: split at char boundary before max_boundary
                        let mut pt = max_boundary;
                        while !text.is_char_boundary(pt) && pt > cur_start {
                            pt -= 1;
                        }
                        if pt == cur_start {
                            // If single char is larger than max_boundary, advance to next char boundary
                            pt = max_boundary;
                            while !text.is_char_boundary(pt) && pt < s.end {
                                pt += 1;
                            }
                        }
                        pt
                    }
                };

                units.push(Span::new(cur_start, split_pt));
                cur_start = split_pt;
            }
        }
    }

    if units.is_empty() {
        return vec![Span::new(0, text.len())];
    }

    // Step 2: Greedy packing up to chunk_target_chars
    let mut chunks: Vec<Span> = Vec::new();
    let mut cur_chunk = units[0];

    for &unit in &units[1..] {
        let combined_len = unit.end - cur_chunk.start;
        if combined_len <= cfg.chunk_target_chars {
            cur_chunk.end = unit.end;
        } else if combined_len <= cfg.chunk_max_chars && cur_chunk.len() < cfg.chunk_min_chars {
            // Keep packing if current chunk is smaller than min
            cur_chunk.end = unit.end;
        } else {
            chunks.push(cur_chunk);
            cur_chunk = unit;
        }
    }
    chunks.push(cur_chunk);

    // Step 3: Merge tail < chunk_min_chars into previous chunk
    if chunks.len() > 1 {
        let last = *chunks.last().unwrap();
        if last.len() < cfg.chunk_min_chars {
            chunks.pop();
            let prev = chunks.last_mut().unwrap();
            prev.end = last.end;
        }
    }

    // Invariant check: ensure exact coverage [0, text.len())
    if let Some(first) = chunks.first_mut() {
        first.start = 0;
    }
    if let Some(last) = chunks.last_mut() {
        last.end = text.len();
    }

    chunks
}
