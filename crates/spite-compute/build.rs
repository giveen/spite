//! Compiles `core/quant.c` - the single dequantization implementation shared by
//! the Rust host and the C/CUDA kernels (vendored from ggml, see core/THIRD_PARTY.md).

fn main() {
    let core = std::path::Path::new("../../core");
    println!("cargo:rerun-if-changed={}", core.join("quant.c").display());
    println!("cargo:rerun-if-changed={}", core.join("quant.h").display());
    println!(
        "cargo:rerun-if-changed={}",
        core.join("ggml-common.h").display()
    );
    println!("cargo:rerun-if-changed={}", core.join("abi.h").display());
    cc::Build::new()
        .file(core.join("quant.c"))
        .include(core)
        .std("c11")
        .opt_level(2)
        .warnings(true)
        .compile("spite_quant");
}
