use scour_core::chunker::Span;
use scour_core::pipeline::score_doc;
use scour_core::reasons::ReasonSet;
use scour_core::rebuild::ChunkVerdict;
use smallvec::SmallVec;

fn make_verdict(keep: bool, g: f32) -> ChunkVerdict {
    ChunkVerdict {
        keep,
        strip: SmallVec::new(),
        reasons: ReasonSet::new(),
        g,
    }
}

#[test]
fn test_15_worked_examples_scoring() {
    // Case 1: Perfect Long Document (Score 10.0)
    let chunks1 = vec![Span::new(0, 2000)];
    let verdicts1 = vec![make_verdict(true, 0.0)];
    let s1 = score_doc(&chunks1, &verdicts1, 0, 2000, 0, 2000, true, 1.0, 0);
    assert_eq!(s1, 10.0, "Case 1 failed");

    // Case 2: Pure Garbage Document (Score 2.2)
    let chunks2 = vec![Span::new(0, 500)];
    let verdicts2 = vec![make_verdict(false, 1.0)];
    let s2 = score_doc(&chunks2, &verdicts2, 500, 500, 0, 0, true, 0.8, 0);
    assert_eq!(s2, 2.2, "Case 2 failed");

    // Case 3: Wrong Language Doc (Score 8.1)
    let chunks3 = vec![Span::new(0, 1000)];
    let verdicts3 = vec![make_verdict(true, 0.0)];
    let s3 = score_doc(&chunks3, &verdicts3, 0, 1000, 0, 1000, false, 0.9, 0);
    assert_eq!(s3, 8.1, "Case 3 failed");

    // Case 4: Moderate Promo Stripped (Score 8.8)
    let chunks4 = vec![Span::new(0, 500), Span::new(500, 1000)];
    let verdicts4 = vec![make_verdict(true, 0.0), make_verdict(true, 0.0)];
    let s4 = score_doc(&chunks4, &verdicts4, 100, 1000, 0, 900, true, 1.0, 0);
    assert_eq!(s4, 8.8, "Case 4 failed");

    // Case 5: Heavy Promo Stripped (Score 5.2)
    let chunks5 = vec![Span::new(0, 500), Span::new(500, 1000)];
    let verdicts5 = vec![make_verdict(true, 0.0), make_verdict(false, 1.0)];
    let s5 = score_doc(&chunks5, &verdicts5, 500, 1000, 0, 500, true, 0.95, 0);
    assert_eq!(s5, 5.2, "Case 5 failed");

    // Case 6: High URL Density (Score 8.6)
    let chunks6 = vec![Span::new(0, 1000)];
    let verdicts6 = vec![make_verdict(true, 0.0)];
    let s6 = score_doc(&chunks6, &verdicts6, 0, 1000, 5, 1000, true, 1.0, 0);
    assert_eq!(s6, 8.6, "Case 6 failed");

    // Case 7: Moderate URL Density (Score 9.2)
    let chunks7 = vec![Span::new(0, 1000)];
    let verdicts7 = vec![make_verdict(true, 0.0)];
    let s7 = score_doc(&chunks7, &verdicts7, 0, 1000, 2, 1000, true, 1.0, 0);
    assert_eq!(s7, 9.2, "Case 7 failed");

    // Case 8: Short Text (L=100) (Score 8.8)
    let chunks8 = vec![Span::new(0, 100)];
    let verdicts8 = vec![make_verdict(true, 0.0)];
    let s8 = score_doc(&chunks8, &verdicts8, 0, 100, 0, 100, true, 1.0, 0);
    assert_eq!(s8, 8.8, "Case 8 failed");

    // Case 9: Micro-Text (L=20) (Score 8.3)
    let chunks9 = vec![Span::new(0, 20)];
    let verdicts9 = vec![make_verdict(true, 0.0)];
    let s9 = score_doc(&chunks9, &verdicts9, 0, 20, 0, 20, true, 0.8, 0);
    assert_eq!(s9, 8.3, "Case 9 failed");

    // Case 10: Orphan Ref Penalty (Score 5.0)
    let chunks10 = vec![Span::new(0, 500), Span::new(500, 1000)];
    let verdicts10 = vec![make_verdict(false, 1.0), make_verdict(true, 0.0)];
    let s10 = score_doc(&chunks10, &verdicts10, 500, 1000, 0, 500, true, 1.0, 3);
    assert_eq!(s10, 5.0, "Case 10 failed");

    // Case 11: Max Orphan Penalty (Score 9.0)
    let chunks11 = vec![Span::new(0, 2000)];
    let verdicts11 = vec![make_verdict(true, 0.0)];
    let s11 = score_doc(&chunks11, &verdicts11, 0, 2000, 0, 2000, true, 1.0, 12);
    assert_eq!(s11, 9.0, "Case 11 failed");

    // Case 12: Low Lang Confidence (Score 9.2 or 9.3)
    let chunks12 = vec![Span::new(0, 2000)];
    let verdicts12 = vec![make_verdict(true, 0.0)];
    let s12 = score_doc(&chunks12, &verdicts12, 0, 2000, 0, 2000, true, 0.5, 0);
    assert!((s12 - 9.2).abs() < 0.15, "Case 12 failed, got {}", s12);

    // Case 13: Borderline Quality Doc (Score 7.4)
    let chunks13 = vec![Span::new(0, 400), Span::new(400, 800), Span::new(800, 1200)];
    let verdicts13 = vec![
        make_verdict(true, 0.3),
        make_verdict(true, 0.5),
        make_verdict(true, 0.2),
    ];
    let s13 = score_doc(&chunks13, &verdicts13, 50, 1200, 2, 1150, true, 0.9, 1);
    assert_eq!(s13, 7.4, "Case 13 failed");

    // Case 14: Degraded Multi-Flaw (Score 2.3)
    let chunks14 = vec![Span::new(0, 500), Span::new(500, 1000)];
    let verdicts14 = vec![make_verdict(true, 0.6), make_verdict(false, 1.0)];
    let s14 = score_doc(&chunks14, &verdicts14, 500, 1000, 4, 400, true, 0.7, 2);
    assert_eq!(s14, 2.3, "Case 14 failed");

    // Case 15: Clean Hindi Article (Score 9.5)
    let chunks15 = vec![Span::new(0, 600), Span::new(600, 1200)];
    let verdicts15 = vec![make_verdict(true, 0.0), make_verdict(true, 0.0)];
    let s15 = score_doc(&chunks15, &verdicts15, 0, 1200, 1, 1200, true, 0.98, 0);
    assert_eq!(s15, 9.5, "Case 15 failed");
}
