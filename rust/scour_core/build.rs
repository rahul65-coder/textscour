fn main() {
    let in_maturin = std::env::var("PYO3_PYTHON").is_ok();
    if !in_maturin {
        println!("cargo:rustc-link-lib=python3.13");
    }
}
