use scour_core::chunker::{chunk, split_sentences};
use scour_core::config::Config;
use scour_core::promo::Compiled;
use scour_core::reasons::{Reason, ReasonSet};
use scour_core::rebuild::{rebuild, ChunkVerdict};
use smallvec::SmallVec;

#[test]
fn test_rebuild_identity_invariant() {
    let mut cfg = Config::default();
    cfg.chunk_target_chars = 50;
    cfg.chunk_min_chars = 10;
    let compiled = Compiled::build(&cfg).expect("compiled should build");

    let text = "First sentence of paragraph. Second sentence of paragraph.\n\nAnother paragraph begins here with some details.";
    let sents = split_sentences(text);
    let chunks = chunk(text, &sents, &cfg);

    let verdicts: Vec<ChunkVerdict> = chunks
        .iter()
        .map(|_| ChunkVerdict {
            keep: true,
            strip: SmallVec::new(),
            reasons: ReasonSet::new(),
            g: 0.0,
        })
        .collect();

    let res = rebuild(text, &chunks, &verdicts, &compiled, &cfg);
    assert_eq!(
        res.text, text,
        "Identity invariant violated: text should be byte-identical"
    );
    assert_eq!(res.promo_chars, 0);
    assert_eq!(res.orphan_refs, 0);
    assert!(res.removed.is_empty());
}

#[test]
fn test_rebuild_drop_and_strip() {
    let mut cfg = Config::default();
    cfg.chunk_target_chars = 50;
    cfg.chunk_min_chars = 10;
    let compiled = Compiled::build(&cfg).expect("compiled should build");

    let text = "Keep this valid chunk.\n\nBuy now flat 50% discount call 9999999999.\n\nAs mentioned above this is conclusion.";
    let sents = split_sentences(text);
    let chunks = chunk(text, &sents, &cfg);

    assert!(chunks.len() >= 2);

    let mut verdicts: Vec<ChunkVerdict> = Vec::new();
    // Keep first chunk
    verdicts.push(ChunkVerdict {
        keep: true,
        strip: SmallVec::new(),
        reasons: ReasonSet::new(),
        g: 0.0,
    });
    // Drop second chunk (promo)
    let mut reasons = ReasonSet::new();
    reasons.insert(Reason::PromoPhrase);
    verdicts.push(ChunkVerdict {
        keep: false,
        strip: SmallVec::new(),
        reasons,
        g: 0.9,
    });
    // If there's a third chunk, keep it
    if chunks.len() > 2 {
        verdicts.push(ChunkVerdict {
            keep: true,
            strip: SmallVec::new(),
            reasons: ReasonSet::new(),
            g: 0.0,
        });
    }

    let res = rebuild(text, &chunks, &verdicts, &compiled, &cfg);
    assert!(!res.removed.is_empty());
    assert!(res.promo_chars > 0);
    assert!(!res.text.contains("Buy now flat 50%"));
    assert!(res.text.contains("Keep this valid chunk."));
}
