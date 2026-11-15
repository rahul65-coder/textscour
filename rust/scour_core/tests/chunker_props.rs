use rand::{thread_rng, Rng};
use scour_core::chunker::{chunk, split_sentences};
use scour_core::config::Config;

#[test]
fn test_chunker_basic() {
    let text = "Hello world. This is the first sentence. And here is another one!\n\nNew paragraph starts here. Does it work? Yes it does।";
    let cfg = Config::default();
    let sents = split_sentences(text);
    assert!(!sents.is_empty());

    // Check sentence coverage
    assert_eq!(sents.first().unwrap().start, 0);
    assert_eq!(sents.last().unwrap().end, text.len());
    for i in 0..sents.len() - 1 {
        assert_eq!(sents[i].end, sents[i + 1].start);
    }

    let chunks = chunk(text, &sents, &cfg);
    assert!(!chunks.is_empty());
    assert_eq!(chunks.first().unwrap().start, 0);
    assert_eq!(chunks.last().unwrap().end, text.len());
    for i in 0..chunks.len() - 1 {
        assert_eq!(chunks[i].end, chunks[i + 1].start);
    }

    let reconstructed: String = chunks.iter().map(|c| &text[c.start..c.end]).collect();
    assert_eq!(reconstructed, text);
}

#[test]
fn test_chunker_property_invariants_10k() {
    let mut rng = thread_rng();
    let sample_chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?।॥…\n\t\r\n'\"()[]{}@#%&*+-=/\\:;नमस्ते दुनिया 你好世界 🌍🚀"
        .chars()
        .collect();

    let mut cfg = Config::default();
    cfg.chunk_target_chars = 300;
    cfg.chunk_min_chars = 50;
    cfg.chunk_max_chars = 600;

    for iter in 0..10_000 {
        // Generate random string
        let len = rng.gen_range(0..2500);
        let s: String = (0..len)
            .map(|_| sample_chars[rng.gen_range(0..sample_chars.len())])
            .collect();

        let sents = split_sentences(&s);
        let chunks = chunk(&s, &sents, &cfg);

        if s.is_empty() {
            assert!(chunks.is_empty(), "Empty input should produce empty chunks");
            continue;
        }

        // Invariant 1: First span starts at 0
        assert_eq!(
            chunks.first().unwrap().start,
            0,
            "Iter {}: first span must start at 0",
            iter
        );

        // Invariant 2: Last span ends at text.len()
        assert_eq!(
            chunks.last().unwrap().end,
            s.len(),
            "Iter {}: last span must end at s.len()",
            iter
        );

        // Invariant 3: Spans are contiguous and non-overlapping
        for i in 0..chunks.len() - 1 {
            assert_eq!(
                chunks[i].end,
                chunks[i + 1].start,
                "Iter {}: spans {} and {} must be contiguous",
                iter,
                i,
                i + 1
            );
        }

        // Invariant 4: Concatenation of chunk slices == original
        let mut reconstructed = String::with_capacity(s.len());
        for c in &chunks {
            // Invariant 5: No slice splits a UTF-8 char (is_char_boundary)
            assert!(
                s.is_char_boundary(c.start),
                "Iter {}: start is not char boundary",
                iter
            );
            assert!(
                s.is_char_boundary(c.end),
                "Iter {}: end is not char boundary",
                iter
            );
            reconstructed.push_str(&s[c.start..c.end]);
        }
        assert_eq!(
            reconstructed, s,
            "Iter {}: reconstructed text must match original",
            iter
        );
    }
}

#[test]
fn test_chunker_hindi_and_unicode_boundary_no_panic() {
    let cfg = Config::default();

    // Specific cases that panicked during K4 real HF run:
    // multi-byte Devanagari (\u{94d}, ा, स), arrow (→), and next-line (\u{85}) around periods/abbreviations
    let problematic_samples = [
        "यहाँ कुछ शब्द हैं\u{94d} e.g. उदाहरण के लिए। फिर दूसरा वाक्य।",
        "विशेष परिस्थिति में →\u{85}i.e. अर्थात ऐसा होता है।",
        "क\u{94d}ष. कुछ और वाक्य।",
        "शब्द\u{85}. अगला वाक्य।",
        "→. टेस्ट वाक्य।",
        "अ\u{94d}ा\u{85}→.end",
        "प\u{94d}र\u{947}मचन्द (३१ जुलाई, १८८० - ८ अक्‍तूबर, १९३६) हिन्दी और उर्दू के महानतम लेखकों में से एक हैं।",
        "123.456.789 → कुछ\u{94d} टेक्स्ट।",
    ];

    for s in &problematic_samples {
        let sents = split_sentences(s);
        let chunks = chunk(s, &sents, &cfg);
        assert!(!chunks.is_empty());
        let reconstructed: String = chunks.iter().map(|c| &s[c.start..c.end]).collect();
        assert_eq!(&reconstructed, s);
    }
}
