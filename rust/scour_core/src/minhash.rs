use hashbrown::HashSet;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use unicode_normalization::UnicodeNormalization;
use xxhash_rust::xxh3::xxh3_64;

// Mersenne prime 2^61 - 1
const MERSENNE_PRIME_61: u128 = (1u128 << 61) - 1;

// 64 fixed deterministic (a_i, b_i) constants
const A_COEFFS: [u64; 64] = [
    0x2f64_9273_1e3b_5c81,
    0x1b4c_8a39_7d2e_6f15,
    0x3d7e_1a9b_5c2f_4e63,
    0x5a2c_7e1b_3d9f_8b05,
    0x6f1e_2b4c_8a3d_7e95,
    0x4e6b_1a3c_5d7f_2e89,
    0x7c3d_5e1b_2a8f_9e05,
    0x1e8a_3d7c_5b2f_6e91,
    0x3f5c_7a1e_2d8b_4e09,
    0x5d2e_4b8a_1c7f_3e95,
    0x2b8f_1e3c_5d7a_6e01,
    0x6e09_3a5c_7b1d_2f85,
    0x4a7e_2d1b_5c8f_3e95,
    0x7d1b_5e3a_2c8f_6e05,
    0x1c5d_7a2e_3b8f_4e91,
    0x3b8a_1e5c_7d2f_6e05,
    0x5e2c_4d8a_1b7f_3e91,
    0x2d7f_1a3c_5e8b_6e05,
    0x6a05_3d7e_1b2f_5c89,
    0x4f8b_2e1c_5a7d_3e05,
    0x7b2f_5c1e_3a8d_6e91,
    0x1a9e_3d5b_7c2f_4e05,
    0x3d8b_1e7a_5c2f_6e95,
    0x5c1e_2a8f_3d7b_4e05,
    0x2e95_4b1c_7a3d_6f05,
    0x6f3d_1e5a_2b8c_7e91,
    0x4b7a_2c1e_5d8f_3e05,
    0x7e05_3b8a_1d5c_6f91,
    0x1d2f_5a7c_3e8b_4e05,
    0x3a8d_1c5e_7b2f_6e91,
    0x5f1a_2e8b_3c7d_4e05,
    0x2c7e_4b1a_5d8f_6e95,
    0x6e91_3d5b_1a8f_7c05,
    0x4d1b_2a7e_5c8f_3e91,
    0x7a3d_5e1b_2c8f_6e05,
    0x1f8b_3c5d_7a2e_4e91,
    0x3e05_1a8d_5c7b_6f95,
    0x5b2f_4e1c_7d8a_3e05,
    0x2a7d_1e5b_3c8f_6e91,
    0x6c1e_3a5d_7b2f_4e05,
    0x4e95_2b7a_1c8f_5d05,
    0x7d8a_3e1c_5b2f_6e95,
    0x1b5c_7a2d_3e8f_4e05,
    0x3c8f_1e7b_5a2d_6e91,
    0x5a1e_2d8b_4c7f_3e05,
    0x2f91_4a3c_6e8b_5d05,
    0x6b05_1d7e_3a8f_5c91,
    0x4c7f_2e1a_5b8d_3e05,
    0x7e95_3c1d_5a8f_2b05,
    0x1a8f_3e5c_7b2d_4e91,
    0x3d2e_1b7a_5c8f_6e05,
    0x5c7b_2e1d_4a8f_3e95,
    0x2b05_4d8a_1e7c_6f91,
    0x6f91_3a5c_1d8e_7b05,
    0x4a3d_2e7b_5c1f_8e95,
    0x7b8f_1e3a_5d2c_6e05,
    0x1e5c_3d7a_6b2f_4e91,
    0x3f1a_2b8d_5e7c_6e05,
    0x5d8f_4c1e_2a7b_3e95,
    0x2a3c_5e8f_1b7d_6e05,
    0x6e7b_1a5d_3c8f_4e91,
    0x4b2d_3e1a_5c8f_7e05,
    0x7c8f_2b5e_1a3d_6e95,
    0x1d5a_3c7b_5e8f_2e05,
];

const B_COEFFS: [u64; 64] = [
    0x1357_9bdf_0246_8ace,
    0x0246_8ace_1357_9bdf,
    0x2468_ace0_3579_bdf1,
    0x468a_ce02_579b_df13,
    0x68ac_e024_79bd_f135,
    0x8ace_0246_9bdf_1357,
    0xace0_2468_bdf1_3579,
    0xce02_468a_df13_579b,
    0xe024_68ac_f135_79bd,
    0x0357_9bdf_2468_ace0,
    0x1468_ace0_3579_bdf1,
    0x2579_bdf1_468a_ce02,
    0x368a_ce02_579b_df13,
    0x479b_df13_68ac_e024,
    0x58ac_e024_79bd_f135,
    0x69bd_f135_8ace_0246,
    0x7ace_0246_9bdf_1357,
    0x8bdf_1357_ace0_2468,
    0x9ce0_2468_bdf1_3579,
    0xad02_468a_df13_579b,
    0xbdf1_3579_e024_68ac,
    0xce02_468a_f135_79bd,
    0xdf13_579b_0246_8ace,
    0xf024_68ac_1357_9bdf,
    0x0135_79bd_f024_68ac,
    0x1246_8ace_0357_9bdf,
    0x2357_9bdf_1468_ace0,
    0x3468_ace0_2579_bdf1,
    0x4579_bdf1_368a_ce02,
    0x568a_ce02_479b_df13,
    0x679b_df13_58ac_e024,
    0x78ac_e024_69bd_f135,
    0x89bd_f135_7ace_0246,
    0x9ace_0246_8bdf_1357,
    0xabdf_1357_9ce0_2468,
    0xbce0_2468_ad02_468a,
    0xcd02_468a_bdf1_3579,
    0xdf13_579b_ce02_468a,
    0xf024_68ac_df13_579b,
    0x0246_8ace_f024_68ac,
    0x1357_9bdf_0135_79bd,
    0x2468_ace0_1246_8ace,
    0x3579_bdf1_2357_9bdf,
    0x468a_ce02_3468_ace0,
    0x579b_df13_4579_bdf1,
    0x68ac_e024_568a_ce02,
    0x79bd_f135_679b_df13,
    0x8ace_0246_78ac_e024,
    0x9bdf_1357_89bd_f135,
    0xace0_2468_9ace_0246,
    0xbdf1_3579_abdf_1357,
    0xce02_468a_bce0_2468,
    0xdf13_579b_cd02_468a,
    0xf024_68ac_df13_579b,
    0x0135_79bd_f024_68ac,
    0x1246_8ace_0135_79bd,
    0x2357_9bdf_1246_8ace,
    0x3468_ace0_2357_9bdf,
    0x4579_bdf1_3468_ace0,
    0x568a_ce02_4579_bdf1,
    0x679b_df13_568a_ce02,
    0x78ac_e024_679b_df13,
    0x89bd_f135_78ac_e024,
    0x9ace_0246_89bd_f135,
];

pub fn normalize_for_dedup(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let mut clean = String::with_capacity(nfkc.len());
    let mut in_ws = false;

    for ch in nfkc.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                clean.push(lower);
            }
            in_ws = false;
        } else if ch.is_whitespace() {
            if !in_ws {
                clean.push(' ');
                in_ws = true;
            }
        }
    }

    clean.trim().to_string()
}

pub fn exact_hash(text: &str) -> u64 {
    let norm = normalize_for_dedup(text);
    xxh3_64(norm.as_bytes())
}

pub fn bands(text: &str, k: usize) -> [u64; 8] {
    let norm = normalize_for_dedup(text);
    let words: Vec<&str> = norm.split_whitespace().collect();

    let mut shingles: Vec<String> = Vec::new();
    if words.is_empty() {
        shingles.push(String::new());
    } else if words.len() < k {
        shingles.push(words.join(" "));
    } else {
        for window in words.windows(k) {
            shingles.push(window.join(" "));
        }
    }

    // Compute 64-dim signature
    let mut signature = [u64::MAX; 64];
    for shingle in &shingles {
        let x = xxh3_64(shingle.as_bytes()) as u128;
        for i in 0..64 {
            let a = A_COEFFS[i] as u128;
            let b = B_COEFFS[i] as u128;
            let h = ((a.wrapping_mul(x).wrapping_add(b)) % MERSENNE_PRIME_61) as u64;
            if h < signature[i] {
                signature[i] = h;
            }
        }
    }

    // 8 bands x 8 rows
    let mut band_hashes = [0u64; 8];
    for b in 0..8 {
        let start = b * 8;
        let mut bytes = [0u8; 64];
        for r in 0..8 {
            let row_bytes = signature[start + r].to_le_bytes();
            bytes[r * 8..(r + 1) * 8].copy_from_slice(&row_bytes);
        }
        band_hashes[b] = xxh3_64(&bytes);
    }

    band_hashes
}

#[derive(Serialize, Deserialize)]
struct DedupIndexData {
    exact: HashSet<u64>,
    bands: [HashSet<u64>; 8],
    n: u64,
}

#[pyclass]
pub struct DedupIndex {
    exact: HashSet<u64>,
    bands: [HashSet<u64>; 8],
    n: u64,
}

#[pymethods]
impl DedupIndex {
    #[new]
    pub fn new() -> Self {
        Self {
            exact: HashSet::new(),
            bands: Default::default(),
            n: 0,
        }
    }

    pub fn check_and_insert(&mut self, exact: u64, bands: Vec<u64>) -> Option<&'static str> {
        if self.exact.contains(&exact) {
            return Some("duplicate_exact");
        }

        if bands.len() == 8 {
            for (i, &band_hash) in bands.iter().enumerate() {
                if self.bands[i].contains(&band_hash) {
                    return Some("duplicate_near");
                }
            }
        }

        // Unique: insert into index
        self.exact.insert(exact);
        if bands.len() == 8 {
            for (i, &band_hash) in bands.iter().enumerate() {
                self.bands[i].insert(band_hash);
            }
        }
        self.n += 1;
        None
    }

    pub fn len(&self) -> usize {
        self.n as usize
    }

    pub fn __len__(&self) -> usize {
        self.n as usize
    }

    pub fn mem_bytes(&self) -> usize {
        let exact_mem = self.exact.capacity() * 8;
        let bands_mem: usize = self.bands.iter().map(|b| b.capacity() * 8).sum();
        exact_mem + bands_mem + std::mem::size_of::<Self>()
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let data = DedupIndexData {
            exact: self.exact.clone(),
            bands: self.bands.clone(),
            n: self.n,
        };
        let encoded = bincode::serialize(&data)
            .map_err(|e| PyIOError::new_err(format!("Serialization error: {}", e)))?;
        let compressed = zstd::encode_all(&encoded[..], 3)
            .map_err(|e| PyIOError::new_err(format!("Compression error: {}", e)))?;

        let tmp_path = format!("{}.tmp.{}", path, std::process::id());
        let mut file = File::create(&tmp_path)
            .map_err(|e| PyIOError::new_err(format!("File create error: {}", e)))?;
        file.write_all(&compressed)
            .map_err(|e| PyIOError::new_err(format!("File write error: {}", e)))?;
        file.sync_all()
            .map_err(|e| PyIOError::new_err(format!("Fsync error: {}", e)))?;
        drop(file);

        std::fs::rename(&tmp_path, path)
            .map_err(|e| PyIOError::new_err(format!("File rename error: {}", e)))?;

        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file =
            File::open(path).map_err(|e| PyIOError::new_err(format!("Open error: {}", e)))?;
        let decompressed = zstd::decode_all(file)
            .map_err(|e| PyIOError::new_err(format!("Decompress error: {}", e)))?;
        let data: DedupIndexData = bincode::deserialize(&decompressed)
            .map_err(|e| PyIOError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            exact: data.exact,
            bands: data.bands,
            n: data.n,
        })
    }

    pub fn merge_from(&mut self, other: &DedupIndex) {
        for &e in &other.exact {
            self.exact.insert(e);
        }
        for i in 0..8 {
            for &b in &other.bands[i] {
                self.bands[i].insert(b);
            }
        }
        self.n = self.exact.len() as u64;
    }
}
