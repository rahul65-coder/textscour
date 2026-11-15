use scour_core::minhash::{bands, exact_hash, normalize_for_dedup, DedupIndex};

#[test]
fn test_normalization_and_exact_hash() {
    let t1 = "Hello, World! This is a TEST 123.";
    let t2 = "hello world this is a test 123";
    let norm1 = normalize_for_dedup(t1);
    let norm2 = normalize_for_dedup(t2);
    assert_eq!(norm1, norm2);
    assert_eq!(exact_hash(t1), exact_hash(t2));
}

#[test]
fn test_minhash_bands() {
    let text1 = "The quick brown fox jumps over the lazy dog in the sunny meadow with green grass and beautiful flowers blooming everywhere across the open hills and winding rivers under the bright blue morning sky.";
    let text2 = "The quick brown fox jumps over the lazy dog in the sunny meadow with green grass and beautiful flowers blooming everywhere across the open hills and winding rivers under the bright blue morning sky today.";
    let text3 = "Artificial intelligence and quantum computing are revolutionizing modern scientific research.";

    let b1 = bands(text1, 5);
    let b2 = bands(text2, 5);
    let b3 = bands(text3, 5);

    // Near duplicate (t1 vs t2) should share at least one band
    let shared_1_2 = b1.iter().zip(b2.iter()).filter(|(x, y)| x == y).count();
    assert!(
        shared_1_2 >= 1,
        "Near-duplicates should match at least 1 band"
    );

    // Dissimilar text (t1 vs t3) should share zero bands
    let shared_1_3 = b1.iter().zip(b3.iter()).filter(|(x, y)| x == y).count();
    assert_eq!(shared_1_3, 0, "Dissimilar texts should match 0 bands");
}

#[test]
fn test_dedup_index_crud_and_persistence() {
    let mut index = DedupIndex::new();
    assert_eq!(index.len(), 0);

    let doc1 = "First unique document about machine learning pipelines and models.";
    let doc2 = "First unique document about machine learning pipelines and models."; // exact dup
    let doc3 = "First unique document about machine learning pipelines and models for training."; // near dup
    let doc4 = "Completely distinct topic regarding culinary recipes and Italian cooking pasta.";

    let h1 = exact_hash(doc1);
    let b1 = bands(doc1, 5).to_vec();
    assert_eq!(index.check_and_insert(h1, b1), None);
    assert_eq!(index.len(), 1);

    // Exact dup check
    let h2 = exact_hash(doc2);
    let b2 = bands(doc2, 5).to_vec();
    assert_eq!(index.check_and_insert(h2, b2), Some("duplicate_exact"));
    assert_eq!(index.len(), 1);

    // Near dup check
    let h3 = exact_hash(doc3);
    let b3 = bands(doc3, 5).to_vec();
    assert_eq!(index.check_and_insert(h3, b3), Some("duplicate_near"));
    assert_eq!(index.len(), 1);

    // Distinct document
    let h4 = exact_hash(doc4);
    let b4 = bands(doc4, 5).to_vec();
    assert_eq!(index.check_and_insert(h4, b4), None);
    assert_eq!(index.len(), 2);

    // Save and load round-trip
    let tmp_path = format!("/tmp/test_dedup_index_{}.bin", std::process::id());
    index.save(&tmp_path).expect("save should succeed");

    let loaded = DedupIndex::load(&tmp_path).expect("load should succeed");
    assert_eq!(loaded.len(), 2);

    let _ = std::fs::remove_file(tmp_path);
}
