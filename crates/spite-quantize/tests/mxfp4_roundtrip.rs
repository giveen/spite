//! End-to-end MXFP4 (PXA "PXQ4") quantization: quantize the tiny qwen35 fixture
//! and check the file reloads with the right types and that a quantized weight
//! approximates the source within the e2m1 grid bound.

use std::path::Path;

use spite_quantize::{QuantType, QuantizeConfig, quantize_model};

fn deq(m: &spite_loader::GgufModel, name: &str, n: usize, out: &mut [f32]) {
    let t = m.tensor(name);
    let nbytes = n / t.kind.block_elements() as usize * t.kind.block_bytes() as usize;
    // SAFETY: t.data points at nbytes of the loader's mmap.
    let src = unsafe { std::slice::from_raw_parts(t.data as *const u8, nbytes) };
    spite_compute::dequant::dequant_to_f32(src, t.kind, n, out).expect("dequant");
}

#[test]
fn mxfp4_quantize_reload_roundtrip() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spite-models/tests/data/tiny-qwen35-f16.gguf");
    if !src.exists() {
        eprintln!("fixture missing, skipping");
        return;
    }
    let dst = std::env::temp_dir().join(format!("spite-mxfp4-{}.gguf", std::process::id()));

    let cfg = QuantizeConfig {
        target: QuantType::MXFP4,
        ..Default::default()
    };
    quantize_model(&src, &dst, &cfg).expect("quantize_model");

    let a = spite_loader::GgufModel::open(&src).expect("open src");
    let b = spite_loader::GgufModel::open(&dst).expect("open dst");
    assert_eq!(a.arch(), b.arch());
    assert_eq!(a.n_tensors(), b.n_tensors());

    // A 2-D weight is now MXFP4 and approximates the source.
    let name = "blk.0.ffn_gate.weight";
    let ta = a.tensor(name);
    let n: usize = ta.ne.iter().map(|&x| x.max(1) as usize).product();
    let tb = b.tensor(name);
    assert_eq!(tb.kind, spite_abi::SpiteType::Mxfp4, "weight not quantized");
    assert_eq!(ta.ne, tb.ne);

    let mut fa = vec![0f32; n];
    let mut fb = vec![0f32; n];
    deq(&a, name, n, &mut fa);
    deq(&b, name, n, &mut fb);
    let amax = fa.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let err = fa
        .iter()
        .zip(&fb)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(err <= amax / 3.0, "MXFP4 err {err} vs amax {amax}");

    // 1-D parameters stay F32 (the ops read them directly).
    assert_eq!(
        b.tensor("blk.0.ssm_a").kind,
        spite_abi::SpiteType::F32,
        "1-D parameter was quantized"
    );

    std::fs::remove_file(&dst).ok();
}
