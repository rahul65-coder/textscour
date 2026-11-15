use scour_core::config::Config;
use scour_core::promo::{score_sentence, Compiled};
use scour_core::reasons::Reason;
use scour_core::urls::find_urls;

#[test]
fn test_promo_lexicon_and_scoring() {
    let cfg = Config::default();
    let compiled = Compiled::build(&cfg).expect("compiled should build");

    // 1. Clean factual sentence
    let clean_sent =
        "The Python programming language was created by Guido van Rossum and released in 1991.";
    let urls = find_urls(clean_sent, Some(&compiled.tlds));
    let score = score_sentence(clean_sent, &urls, &compiled);
    assert!(
        score.score < 0.20,
        "Clean sentence should have low promo score, got {}",
        score.score
    );
    assert!(score.hits.is_empty());

    // 2. Clear promo sentence with CTA and discount
    let promo_sent = "Click here to buy now and get flat 50% off on your first order!";
    let urls_p = find_urls(promo_sent, Some(&compiled.tlds));
    let score_p = score_sentence(promo_sent, &urls_p, &compiled);
    assert!(
        score_p.score >= 0.70,
        "Promo sentence should have high score, got {}",
        score_p.score
    );
    assert!(score_p
        .hits
        .iter()
        .any(|h| h.2 == Reason::PromoPhrase || h.2 == Reason::CtaSentence));

    // 3. Shortener URL promo
    let shortener_sent = "Check out our exclusive deal at https://bit.ly/deal123 today.";
    let urls_s = find_urls(shortener_sent, Some(&compiled.tlds));
    let score_s = score_sentence(shortener_sent, &urls_s, &compiled);
    assert!(
        score_s.score >= 0.60,
        "Shortener URL should trigger high score, got {}",
        score_s.score
    );
    assert!(score_s.hits.iter().any(|h| h.2 == Reason::UrlSpam));

    // 4. Promo code
    let code_sent = "Use coupon code SAVE20 at checkout for instant savings.";
    let urls_c = find_urls(code_sent, Some(&compiled.tlds));
    let score_c = score_sentence(code_sent, &urls_c, &compiled);
    assert!(
        score_c.score >= 0.70,
        "Coupon code should trigger >= 0.70, got {}",
        score_c.score
    );

    // 5. Devanagari Hindi promo
    let hi_promo = "ऑफर केवल सीमित समय के लिए उपलब्ध है। अभी खरीदें।";
    let urls_h = find_urls(hi_promo, Some(&compiled.tlds));
    let score_h = score_sentence(hi_promo, &urls_h, &compiled);
    assert!(
        score_h.score >= 0.50,
        "Hindi promo should trigger promo score, got {}",
        score_h.score
    );
}
