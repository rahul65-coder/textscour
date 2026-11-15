use scour_core::config::{Config, UrlPolicy};
use scour_core::urls::{decide, find_urls, host_matches, host_of, UrlAction, UrlKind, UrlSpan};

#[test]
fn test_host_of() {
    assert_eq!(
        host_of("https://en.wikipedia.org/wiki/Rust"),
        Some("en.wikipedia.org".to_string())
    );
    assert_eq!(
        host_of("http://www.google.com/search?q=test"),
        Some("google.com".to_string())
    );
    assert_eq!(
        host_of("www.github.com/rust-lang"),
        Some("github.com".to_string())
    );
    assert_eq!(
        host_of("hxxps://malicious[.]com/bad"),
        Some("malicious.com".to_string())
    );
    assert_eq!(
        host_of("https://user:pass@sub.domain.co.uk:8080/foo"),
        Some("sub.domain.co.uk".to_string())
    );
    assert_eq!(host_of("https://t.me/mychannel"), Some("t.me".to_string()));
    assert_eq!(host_of("not a url"), None);
    assert_eq!(host_of(""), None);
}

#[test]
fn test_host_matches() {
    let patterns = vec![
        "wikipedia.org".to_string(),
        "*.gov.in".to_string(),
        "*.edu".to_string(),
        "github.com".to_string(),
    ];

    assert!(host_matches("wikipedia.org", &patterns));
    assert!(host_matches("en.wikipedia.org", &patterns));
    assert!(host_matches("sub.en.wikipedia.org", &patterns));
    assert!(host_matches("uidai.gov.in", &patterns));
    assert!(host_matches("my.portal.gov.in", &patterns));
    assert!(host_matches("stanford.edu", &patterns));
    assert!(host_matches("github.com", &patterns));
    assert!(!host_matches("fake-wikipedia.org", &patterns));
    assert!(!host_matches("evilgov.in", &patterns));
    assert!(!host_matches("google.com", &patterns));
}

#[test]
fn test_find_urls() {
    let text = "Check out https://github.com/repo and www.wikipedia.org or [Python Docs](https://docs.python.org/3). Also contact @support on t.me/rustlang.";
    let tlds = vec!["com".to_string(), "org".to_string()];
    let urls = find_urls(text, Some(&tlds));
    assert!(!urls.is_empty());

    let md_span = urls.iter().find(|u| u.kind == UrlKind::Markdown);
    assert!(md_span.is_some());
    assert_eq!(md_span.unwrap().host, "docs.python.org");

    let tme_span = urls.iter().find(|u| u.kind == UrlKind::Telegram);
    assert!(tme_span.is_some());

    let handle_span = urls.iter().find(|u| u.kind == UrlKind::Handle);
    assert!(handle_span.is_some());
}

#[test]
fn test_decide_truth_table_40_rows() {
    let mut cfg = Config::default();
    cfg.allow_hosts = vec!["wikipedia.org".to_string(), "github.com".to_string()];
    cfg.block_hosts = vec!["bad.com".to_string(), "spam.org".to_string()];
    cfg.promo_lo = 0.30;
    cfg.promo_hi = 0.70;

    struct Case {
        policy: UrlPolicy,
        host: &'static str,
        kind: UrlKind,
        anchor: bool,
        sentence_promo: f32,
        sentence_url_only: bool,
        expected: UrlAction,
    }

    let cases = vec![
        // KeepAll policy (1..4)
        Case {
            policy: UrlPolicy::KeepAll,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::KeepAll,
            host: "spam.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.9,
            sentence_url_only: true,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::KeepAll,
            host: "unknown.com",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.5,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::KeepAll,
            host: "",
            kind: UrlKind::Handle,
            anchor: false,
            sentence_promo: 0.2,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        // RemoveAll policy (5..12)
        Case {
            policy: UrlPolicy::RemoveAll,
            host: "wikipedia.org",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::ReplaceWithAnchor,
        },
        Case {
            policy: UrlPolicy::RemoveAll,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: true,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::RemoveAll,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.85,
            sentence_url_only: false,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::RemoveAll,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        Case {
            policy: UrlPolicy::RemoveAll,
            host: "bad.com",
            kind: UrlKind::HtmlAnchor,
            anchor: true,
            sentence_promo: 0.8,
            sentence_url_only: false,
            expected: UrlAction::ReplaceWithAnchor,
        },
        Case {
            policy: UrlPolicy::RemoveAll,
            host: "bad.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.8,
            sentence_url_only: true,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::RemoveAll,
            host: "unknown.com",
            kind: UrlKind::BareDomain,
            anchor: false,
            sentence_promo: 0.2,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        Case {
            policy: UrlPolicy::RemoveAll,
            host: "",
            kind: UrlKind::Handle,
            anchor: false,
            sentence_promo: 0.85,
            sentence_url_only: false,
            expected: UrlAction::DropSentence,
        },
        // AllowlistOnly policy (13..20)
        Case {
            policy: UrlPolicy::AllowlistOnly,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::AllowlistOnly,
            host: "en.wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.6,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::AllowlistOnly,
            host: "github.com",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.8,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::AllowlistOnly,
            host: "unknown.com",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::ReplaceWithAnchor,
        },
        Case {
            policy: UrlPolicy::AllowlistOnly,
            host: "unknown.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: true,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::AllowlistOnly,
            host: "unknown.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.75,
            sentence_url_only: false,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::AllowlistOnly,
            host: "unknown.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        Case {
            policy: UrlPolicy::AllowlistOnly,
            host: "bad.com",
            kind: UrlKind::BareDomain,
            anchor: false,
            sentence_promo: 0.5,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        // Blocklist policy (21..28)
        Case {
            policy: UrlPolicy::Blocklist,
            host: "bad.com",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::ReplaceWithAnchor,
        },
        Case {
            policy: UrlPolicy::Blocklist,
            host: "bad.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: true,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::Blocklist,
            host: "spam.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.8,
            sentence_url_only: false,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::Blocklist,
            host: "bad.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.2,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        Case {
            policy: UrlPolicy::Blocklist,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.9,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::Blocklist,
            host: "unknown.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::Blocklist,
            host: "google.com",
            kind: UrlKind::BareDomain,
            anchor: false,
            sentence_promo: 0.4,
            sentence_url_only: true,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::Blocklist,
            host: "sub.spam.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.2,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        // Smart policy (29..44)
        Case {
            policy: UrlPolicy::Smart,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "en.wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.25,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "github.com",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::Keep,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.35,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        }, // promo >= promo_lo!
        Case {
            policy: UrlPolicy::Smart,
            host: "wikipedia.org",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.45,
            sentence_url_only: true,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "wikipedia.org",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.5,
            sentence_url_only: false,
            expected: UrlAction::ReplaceWithAnchor,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "bad.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "bad.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: true,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "bad.com",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::ReplaceWithAnchor,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "unknown.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "unknown.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.35,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "unknown.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.8,
            sentence_url_only: false,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "unknown.com",
            kind: UrlKind::Http,
            anchor: false,
            sentence_promo: 0.05,
            sentence_url_only: true,
            expected: UrlAction::DropSentence,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "unknown.com",
            kind: UrlKind::Markdown,
            anchor: true,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::ReplaceWithAnchor,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "unknown.com",
            kind: UrlKind::HtmlAnchor,
            anchor: true,
            sentence_promo: 0.9,
            sentence_url_only: false,
            expected: UrlAction::ReplaceWithAnchor,
        },
        Case {
            policy: UrlPolicy::Smart,
            host: "",
            kind: UrlKind::Handle,
            anchor: false,
            sentence_promo: 0.1,
            sentence_url_only: false,
            expected: UrlAction::DeleteToken,
        },
    ];

    assert!(
        cases.len() >= 40,
        "Must test at least 40 cases, got {}",
        cases.len()
    );

    for (i, c) in cases.iter().enumerate() {
        cfg.url_policy = c.policy;
        let span = UrlSpan {
            start: 0,
            end: 10,
            host: c.host.to_string(),
            kind: c.kind,
            anchor: if c.anchor { Some((0, 5)) } else { None },
        };
        let act = decide(&span, c.sentence_promo, c.sentence_url_only, &cfg);
        assert_eq!(
            act,
            c.expected,
            "Case {} failed: policy={:?} host={} promo={} url_only={} expected={:?} got={:?}",
            i + 1,
            c.policy,
            c.host,
            c.sentence_promo,
            c.sentence_url_only,
            c.expected,
            act
        );
    }
}
