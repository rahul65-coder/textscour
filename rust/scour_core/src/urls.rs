use crate::config::{Config, UrlPolicy};
use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UrlKind {
    Http,
    Www,
    BareDomain,
    Shortener,
    Markdown,
    HtmlAnchor,
    Handle,
    Telegram,
    WhatsApp,
    Email,
    Obfuscated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlSpan {
    pub start: usize,
    pub end: usize,
    pub host: String,
    pub kind: UrlKind,
    pub anchor: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrlAction {
    Keep,
    DeleteToken,
    ReplaceWithAnchor,
    DropSentence,
}

pub static SHORTENERS: &[&str] = &[
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "cutt.ly",
    "rb.gy",
    "is.gd",
    "ow.ly",
    "shorturl.at",
    "linktr.ee",
];

static PROTOCOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:https?|ftp|hxxps?)://[^\s<>]+").expect("valid url protocol regex")
});

static WWW_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\bwww\.[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z0-9][-a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=%]+",
    )
    .expect("valid www regex")
});

static TELEGRAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:t\.me|telegram\.me)/[a-zA-Z0-9_]{3,}[^\s<>]*")
        .expect("valid telegram regex")
});

static WHATSAPP_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:wa\.me/|api\.whatsapp\.com/send\?phone=)[0-9+\[\]\s-]{6,}[^\s<>]*")
        .expect("valid whatsapp regex")
});

static MD_LINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").expect("valid markdown link regex"));

static HTML_A_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)<a\s+[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>"#)
        .expect("valid html link regex")
});

static HANDLE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|[\s(,;:\[])@([a-zA-Z0-9_]{3,30})\b").expect("valid handle regex")
});

static OBFUSCATED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b[a-zA-Z0-9][-a-zA-Z0-9]*(?:\[\.\]|\(dot\))[a-zA-Z]{2,}(?:[/.][^\s<>]*)?")
        .expect("valid obfuscated regex")
});

pub fn host_of(url: &str) -> Option<String> {
    let mut s = url.trim();
    if s.is_empty() {
        return None;
    }

    // Strip wrapping markdown/html if any
    if s.starts_with('[') && s.contains("](") {
        if let Some(pos) = s.find("](") {
            let rest = &s[pos + 2..];
            if let Some(end) = rest.find(')') {
                s = &rest[..end];
            }
        }
    }

    let unobs = s.replace("[.]", ".").replace("(dot)", ".");
    let mut cur = unobs.as_str();

    if let Some(idx) = cur.find("://") {
        cur = &cur[idx + 3..];
    } else if cur.to_ascii_lowercase().starts_with("hxxp://") {
        cur = &cur[7..];
    } else if cur.to_ascii_lowercase().starts_with("hxxps://") {
        cur = &cur[8..];
    }

    if let Some(idx) = cur.find('/') {
        cur = &cur[..idx];
    }
    if let Some(idx) = cur.find('?') {
        cur = &cur[..idx];
    }
    if let Some(idx) = cur.find('#') {
        cur = &cur[..idx];
    }

    if let Some(idx) = cur.rfind('@') {
        cur = &cur[idx + 1..];
    }

    if let Some(idx) = cur.find(':') {
        cur = &cur[..idx];
    }

    let mut host = cur.trim().to_ascii_lowercase();
    while host.starts_with('.') {
        host.remove(0);
    }
    while host.ends_with('.') {
        host.pop();
    }

    if host.starts_with("www.") {
        host = host[4..].to_string();
    }

    if host.is_empty() || !host.contains('.') {
        return None;
    }

    Some(host)
}

pub fn host_matches(host: &str, patterns: &[String]) -> bool {
    let h = host.to_ascii_lowercase();
    for p in patterns {
        let mut clean = p.trim().to_ascii_lowercase();
        if clean.starts_with("*.") {
            clean = clean[2..].to_string();
        } else if clean.starts_with('*') {
            clean = clean[1..].to_string();
        }
        if clean.is_empty() {
            continue;
        }

        if h == clean || h.ends_with(&format!(".{}", clean)) {
            return true;
        }
    }
    false
}

fn trim_trailing_punct(s: &str) -> usize {
    let mut trim_len = 0;
    let chars: Vec<char> = s.chars().collect();
    let mut i = chars.len();

    let mut open_paren = 0;
    let mut close_paren = 0;
    for &c in &chars {
        if c == '(' {
            open_paren += 1;
        } else if c == ')' {
            close_paren += 1;
        }
    }

    while i > 0 {
        let c = chars[i - 1];
        if c == ')' && open_paren >= close_paren {
            break;
        }
        if matches!(
            c,
            '.' | ',' | ';' | ':' | '!' | '?' | ']' | '}' | '\'' | '"' | '>'
        ) {
            trim_len += c.len_utf8();
            i -= 1;
        } else {
            break;
        }
    }
    trim_len
}

pub fn find_urls(text: &str, known_tlds: Option<&[String]>) -> Vec<UrlSpan> {
    let mut spans: Vec<UrlSpan> = Vec::new();

    // 1. Markdown links [text](url)
    for cap in MD_LINK_RE.captures_iter(text) {
        if let Some(m) = cap.get(0) {
            let anchor_match = cap.get(1).unwrap();
            let url_match = cap.get(2).unwrap();
            let host = host_of(url_match.as_str()).unwrap_or_default();
            let kind = if SHORTENERS
                .iter()
                .any(|&s| host == s || host.ends_with(&format!(".{}", s)))
            {
                UrlKind::Shortener
            } else {
                UrlKind::Markdown
            };
            spans.push(UrlSpan {
                start: m.start(),
                end: m.end(),
                host,
                kind,
                anchor: Some((anchor_match.start(), anchor_match.end())),
            });
        }
    }

    // 2. HTML anchor links <a href="...">text</a>
    for cap in HTML_A_RE.captures_iter(text) {
        if let Some(m) = cap.get(0) {
            let url_str = cap.get(1).map(|c| c.as_str()).unwrap_or_default();
            let anchor_match = cap.get(2).unwrap();
            let host = host_of(url_str).unwrap_or_default();
            let kind = if SHORTENERS
                .iter()
                .any(|&s| host == s || host.ends_with(&format!(".{}", s)))
            {
                UrlKind::Shortener
            } else {
                UrlKind::HtmlAnchor
            };
            spans.push(UrlSpan {
                start: m.start(),
                end: m.end(),
                host,
                kind,
                anchor: Some((anchor_match.start(), anchor_match.end())),
            });
        }
    }

    // Helper to check overlap with already found markdown / html spans
    let overlaps = |start: usize, end: usize, existing: &[UrlSpan]| -> bool {
        existing.iter().any(|s| !(end <= s.start || start >= s.end))
    };

    // 3. Protocol URLs (http://, https://, hxxp://)
    for m in PROTOCOL_RE.find_iter(text) {
        let start = m.start();
        let mut end = m.end();
        let raw = &text[start..end];
        let trim_len = trim_trailing_punct(raw);
        end -= trim_len;
        if end <= start || overlaps(start, end, &spans) {
            continue;
        }
        let url_str = &text[start..end];
        let host = host_of(url_str).unwrap_or_default();
        let is_obf = url_str.to_ascii_lowercase().starts_with("hxxp");
        let kind = if is_obf {
            UrlKind::Obfuscated
        } else if SHORTENERS
            .iter()
            .any(|&s| host == s || host.ends_with(&format!(".{}", s)))
        {
            UrlKind::Shortener
        } else {
            UrlKind::Http
        };
        spans.push(UrlSpan {
            start,
            end,
            host,
            kind,
            anchor: None,
        });
    }

    // 4. Telegram links
    for m in TELEGRAM_RE.find_iter(text) {
        let start = m.start();
        let mut end = m.end();
        let trim_len = trim_trailing_punct(&text[start..end]);
        end -= trim_len;
        if end > start && !overlaps(start, end, &spans) {
            let host = "t.me".to_string();
            spans.push(UrlSpan {
                start,
                end,
                host,
                kind: UrlKind::Telegram,
                anchor: None,
            });
        }
    }

    // 5. WhatsApp links
    for m in WHATSAPP_RE.find_iter(text) {
        let start = m.start();
        let mut end = m.end();
        let trim_len = trim_trailing_punct(&text[start..end]);
        end -= trim_len;
        if end > start && !overlaps(start, end, &spans) {
            let host = "wa.me".to_string();
            spans.push(UrlSpan {
                start,
                end,
                host,
                kind: UrlKind::WhatsApp,
                anchor: None,
            });
        }
    }

    // 6. WWW URLs
    for m in WWW_RE.find_iter(text) {
        let start = m.start();
        let mut end = m.end();
        let trim_len = trim_trailing_punct(&text[start..end]);
        end -= trim_len;
        if end > start && !overlaps(start, end, &spans) {
            let url_str = &text[start..end];
            let host = host_of(url_str).unwrap_or_default();
            let kind = if SHORTENERS
                .iter()
                .any(|&s| host == s || host.ends_with(&format!(".{}", s)))
            {
                UrlKind::Shortener
            } else {
                UrlKind::Www
            };
            spans.push(UrlSpan {
                start,
                end,
                host,
                kind,
                anchor: None,
            });
        }
    }

    // 7. Obfuscated URLs (e.g. evil[.]com)
    for m in OBFUSCATED_RE.find_iter(text) {
        let start = m.start();
        let mut end = m.end();
        let trim_len = trim_trailing_punct(&text[start..end]);
        end -= trim_len;
        if end > start && !overlaps(start, end, &spans) {
            let url_str = &text[start..end];
            let host = host_of(url_str).unwrap_or_default();
            spans.push(UrlSpan {
                start,
                end,
                host,
                kind: UrlKind::Obfuscated,
                anchor: None,
            });
        }
    }

    // 8. Handles (@handle)
    for cap in HANDLE_RE.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            let start = m.start() - 1; // include '@'
            let end = m.end();
            if !overlaps(start, end, &spans) {
                // Check it's not email (e.g. user@host)
                if start > 0 {
                    let prev = text[..start].chars().next_back().unwrap();
                    if prev.is_alphanumeric() {
                        continue;
                    }
                }
                spans.push(UrlSpan {
                    start,
                    end,
                    host: String::new(),
                    kind: UrlKind::Handle,
                    anchor: None,
                });
            }
        }
    }

    // 9. Bare domain URLs if known_tlds is available
    if let Some(tlds) = known_tlds {
        // Find potential domain candidates
        static BARE_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"\b([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>]*)?)")
                .expect("bare domain re")
        });
        for m in BARE_RE.find_iter(text) {
            let start = m.start();
            let mut end = m.end();
            let trim_len = trim_trailing_punct(&text[start..end]);
            end -= trim_len;
            if end <= start || overlaps(start, end, &spans) {
                continue;
            }
            let candidate = &text[start..end];
            if let Some(host) = host_of(candidate) {
                // Reject false positives: e.g, i.e, file.txt, version 3.11.4
                if host == "e.g"
                    || host == "i.e"
                    || host.starts_with("eg.")
                    || host.starts_with("ie.")
                {
                    continue;
                }
                let tld = host.split('.').last().unwrap_or("").to_ascii_lowercase();
                if tlds.iter().any(|t| t == &tld) {
                    let kind = if SHORTENERS
                        .iter()
                        .any(|&s| host == s || host.ends_with(&format!(".{}", s)))
                    {
                        UrlKind::Shortener
                    } else {
                        UrlKind::BareDomain
                    };
                    spans.push(UrlSpan {
                        start,
                        end,
                        host,
                        kind,
                        anchor: None,
                    });
                }
            }
        }
    }

    // Sort spans by start offset
    spans.sort_by_key(|s| s.start);

    // Dedup any remaining overlapping spans (keep earlier/longer)
    let mut deduped: Vec<UrlSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        if let Some(last) = deduped.last_mut() {
            if span.start < last.end {
                // Overlap: if current span is strictly within last, skip; if extending, ignore
                continue;
            }
        }
        deduped.push(span);
    }

    deduped
}

pub fn decide(
    span: &UrlSpan,
    sentence_promo: f32,
    sentence_url_only: bool,
    cfg: &Config,
) -> UrlAction {
    match cfg.url_policy {
        UrlPolicy::KeepAll => UrlAction::Keep,
        UrlPolicy::RemoveAll => {
            if span.anchor.is_some() {
                UrlAction::ReplaceWithAnchor
            } else if sentence_url_only || sentence_promo >= cfg.promo_hi {
                UrlAction::DropSentence
            } else {
                UrlAction::DeleteToken
            }
        }
        UrlPolicy::AllowlistOnly => {
            if !span.host.is_empty() && host_matches(&span.host, &cfg.allow_hosts) {
                UrlAction::Keep
            } else if span.anchor.is_some() {
                UrlAction::ReplaceWithAnchor
            } else if sentence_url_only || sentence_promo >= cfg.promo_hi {
                UrlAction::DropSentence
            } else {
                UrlAction::DeleteToken
            }
        }
        UrlPolicy::Blocklist => {
            let blocked = !span.host.is_empty() && host_matches(&span.host, &cfg.block_hosts);
            if blocked {
                if span.anchor.is_some() {
                    UrlAction::ReplaceWithAnchor
                } else if sentence_url_only || sentence_promo >= cfg.promo_hi {
                    UrlAction::DropSentence
                } else {
                    UrlAction::DeleteToken
                }
            } else {
                UrlAction::Keep
            }
        }
        UrlPolicy::Smart => {
            let allowed = !span.host.is_empty()
                && host_matches(&span.host, &cfg.allow_hosts)
                && !host_matches(&span.host, &cfg.block_hosts);

            if allowed && sentence_promo < cfg.promo_lo {
                UrlAction::Keep
            } else if span.anchor.is_some() {
                UrlAction::ReplaceWithAnchor
            } else if sentence_url_only || sentence_promo >= cfg.promo_hi {
                UrlAction::DropSentence
            } else {
                UrlAction::DeleteToken
            }
        }
    }
}
