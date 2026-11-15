use scour_core::config::Config;
use scour_core::pii::{
    compute_verhoeff_check_digit, redact, scan, validate_luhn, validate_verhoeff, PiiKind,
};
use scour_core::promo::Compiled;

#[test]
fn test_luhn_algorithm() {
    // Valid standard test card numbers (Visa/Mastercard test numbers)
    assert!(validate_luhn("4532015112830366"));
    assert!(validate_luhn("4532-0151-1283-0366"));
    // Invalid card number
    assert!(!validate_luhn("4532015112830367"));
}

#[test]
fn test_verhoeff_algorithm() {
    // Known valid Verhoeff numbers
    assert!(validate_verhoeff("2363"));
    assert_eq!(compute_verhoeff_check_digit("236"), 3);

    // Synthesized 12-digit Aadhaar
    let base = "98765432109";
    let check = compute_verhoeff_check_digit(base);
    let valid_aadhaar = format!("{}{}", base, check);
    assert_eq!(valid_aadhaar.len(), 12);
    assert!(validate_verhoeff(&valid_aadhaar));
}

#[test]
fn test_pii_scan_and_redact() {
    let cfg = Config::default();
    let compiled = Compiled::build(&cfg).expect("compiled should build");

    let text = "Contact me at user@example.com or call +91 9876543210. My PAN is ABCDE1234F and secret is AKIA1234567890ABCDEF. Server IP is 192.168.1.10.";
    let hits = scan(text, &compiled);

    assert!(hits.iter().any(|h| h.1 == PiiKind::Email));
    assert!(hits.iter().any(|h| h.1 == PiiKind::Phone));
    assert!(hits.iter().any(|h| h.1 == PiiKind::Pan));
    assert!(hits.iter().any(|h| h.1 == PiiKind::Key));
    assert!(hits.iter().any(|h| h.1 == PiiKind::Ip));

    let redacted = redact(text, &hits);
    assert!(redacted.contains("<EMAIL>"));
    assert!(redacted.contains("<PHONE>"));
    assert!(redacted.contains("<PAN>"));
    assert!(redacted.contains("<KEY>"));
    assert!(redacted.contains("<IP>"));
    assert!(!redacted.contains("user@example.com"));
    assert!(!redacted.contains("ABCDE1234F"));
    assert!(!redacted.contains("AKIA1234567890ABCDEF"));
}

#[test]
fn test_pii_hindi_and_unicode_char_boundary_no_panic() {
    let cfg = Config::default();
    let compiled = Compiled::build(&cfg).expect("compiled should build");

    // Case with Hindi virama/matra/consonants preceding IP: \u{94d}, ा, स, →, \u{85}
    let texts = [
        "यह एक परीक्षण है\u{94d} 192.168.1.1 कुछ और",
        "नमस्ते→\u{85}10.0.0.1",
        "हिन्दी भाषा में IP 172.16.0.1 उपलब्ध है",
        "यहाँ एक arrow है →192.168.0.1",
        "क\u{94d}ष\u{94d}त्र\u{85}192.168.1.100",
        "स\u{94d}थापना v1.2.3.4 और 192.168.1.1",
        "ा\u{94d}→\u{85}192.168.1.1",
    ];

    for text in &texts {
        let hits = scan(text, &compiled);
        let _ = redact(text, &hits);
    }
}
