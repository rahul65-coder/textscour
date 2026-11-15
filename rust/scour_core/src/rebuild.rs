use crate::chunker::Span;
use crate::config::Config;
use crate::promo::Compiled;
use crate::reasons::{Reason, ReasonSet};
use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub struct ChunkVerdict {
    pub keep: bool,
    pub strip: SmallVec<[Span; 4]>,
    pub reasons: ReasonSet,
    pub g: f32,
}

#[derive(Debug, Clone)]
pub struct Fragment {
    pub reason: Reason,
    pub text: String,
    pub start: usize,
}

#[derive(Debug, Clone)]
pub struct Rebuilt {
    pub text: String,
    pub removed: Vec<Fragment>,
    pub orphan_refs: u32,
    pub promo_chars: usize,
}

fn tidy_chunk_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for line in s.lines() {
        let trimmed_line = line.trim();
        // Remove empty bullets/markers
        if trimmed_line == "*"
            || trimmed_line == "-"
            || trimmed_line == "+"
            || trimmed_line == "•"
            || trimmed_line == "–"
            || trimmed_line == "—"
        {
            continue;
        }

        // Collapse multiple spaces within line
        let mut clean_line = String::with_capacity(line.len());
        let mut in_ws = false;
        for c in line.chars() {
            if c == ' ' || c == '\t' {
                if !in_ws {
                    clean_line.push(' ');
                    in_ws = true;
                }
            } else {
                clean_line.push(c);
                in_ws = false;
            }
        }

        // Fix punctuation spacing: " ," -> ", ", ",," -> ","
        let fixed = clean_line
            .replace(" ,", ", ")
            .replace(" .", ". ")
            .replace(" !", "! ")
            .replace(" ?", "? ")
            .replace(",,", ",")
            .replace("..", ".");

        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&fixed);
    }
    out
}

pub fn rebuild(
    text: &str,
    chunks: &[Span],
    verdicts: &[ChunkVerdict],
    c: &Compiled,
    _cfg: &Config,
) -> Rebuilt {
    if text.is_empty() || chunks.is_empty() {
        return Rebuilt {
            text: String::new(),
            removed: Vec::new(),
            orphan_refs: 0,
            promo_chars: 0,
        };
    }

    // Identity invariant: if all chunks kept and no strips, return byte-identical input
    let has_modifications = verdicts.iter().any(|v| !v.keep || !v.strip.is_empty());
    if !has_modifications {
        return Rebuilt {
            text: text.to_string(),
            removed: Vec::new(),
            orphan_refs: 0,
            promo_chars: 0,
        };
    }

    let mut removed = Vec::new();
    let mut promo_chars = 0usize;
    let mut orphan_refs = 0u32;

    let mut kept_pieces: Vec<(usize, String)> = Vec::new();
    let mut prev_dropped = false;

    for (i, (&chunk, verdict)) in chunks.iter().zip(verdicts.iter()).enumerate() {
        if !verdict.keep {
            promo_chars += chunk.len();
            let primary_reason = verdict.reasons.iter().next().unwrap_or(Reason::PromoPhrase);
            removed.push(Fragment {
                reason: primary_reason,
                text: chunk.slice(text).to_string(),
                start: chunk.start,
            });
            prev_dropped = true;
            continue;
        }

        // Kept chunk: check if preceding chunk was dropped and this chunk starts with orphan ref
        if prev_dropped {
            let chunk_raw = chunk.slice(text).trim_start();
            for (start, _end, _w) in c.orphan_refs.find_matches(chunk_raw) {
                if start == 0 {
                    orphan_refs += 1;
                    break;
                }
            }
        }
        prev_dropped = false;

        // Handle strip spans inside kept chunk
        if verdict.strip.is_empty() {
            let piece = chunk.slice(text).to_string();
            kept_pieces.push((i, piece));
        } else {
            // Cut out strip spans
            let mut sorted_strips: Vec<Span> = verdict.strip.to_vec();
            sorted_strips.sort_by_key(|s| s.start);

            let mut piece = String::with_capacity(chunk.len());
            let mut cur = chunk.start;

            for strip in sorted_strips {
                promo_chars += strip.len();
                removed.push(Fragment {
                    reason: Reason::PromoPhrase,
                    text: strip.slice(text).to_string(),
                    start: strip.start,
                });

                if strip.start > cur && strip.start <= chunk.end {
                    piece.push_str(Span::new(cur, strip.start).slice(text));
                }
                cur = cur.max(strip.end);
            }

            if cur < chunk.end {
                piece.push_str(Span::new(cur, chunk.end).slice(text));
            }

            let tidied = tidy_chunk_text(&piece);
            if !tidied.trim().is_empty() {
                kept_pieces.push((i, tidied));
            }
        }
    }

    if kept_pieces.is_empty() {
        return Rebuilt {
            text: String::new(),
            removed,
            orphan_refs,
            promo_chars,
        };
    }

    // Assemble final text
    let mut final_text = String::with_capacity(text.len());
    for (idx, (chunk_idx, piece)) in kept_pieces.iter().enumerate() {
        if idx > 0 {
            let prev_chunk_idx = kept_pieces[idx - 1].0;
            if *chunk_idx == prev_chunk_idx + 1 {
                // Adjacent chunks: preserve original boundary or space
                final_text.push('\n');
            } else {
                // Across a dropped chunk join with "\n\n"
                final_text.push_str("\n\n");
            }
        }
        final_text.push_str(piece);
    }

    // Normalize multiple newlines (> 2 consecutive newlines -> 2)
    let mut normalized = String::with_capacity(final_text.len());
    let mut newline_run = 0;
    for c in final_text.chars() {
        if c == '\n' {
            newline_run += 1;
            if newline_run <= 2 {
                normalized.push(c);
            }
        } else {
            newline_run = 0;
            normalized.push(c);
        }
    }

    let trimmed = normalized.trim().to_string();

    Rebuilt {
        text: trimmed,
        removed,
        orphan_refs,
        promo_chars,
    }
}
