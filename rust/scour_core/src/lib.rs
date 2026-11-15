pub mod chunker;
pub mod config;
pub mod features;
pub mod lang;
pub mod minhash;
pub mod pii;
pub mod pipeline;
pub mod promo;
pub mod reasons;
pub mod rebuild;
pub mod urls;

use minhash::DedupIndex;
use pipeline::{bench_stage1, cpu_features, version, DocResult, Engine, Stage1Batch};
use pyo3::prelude::*;

#[pymodule]
fn scour_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<Stage1Batch>()?;
    m.add_class::<DocResult>()?;
    m.add_class::<DedupIndex>()?;

    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_features, m)?)?;
    m.add_function(wrap_pyfunction!(bench_stage1, m)?)?;

    Ok(())
}
