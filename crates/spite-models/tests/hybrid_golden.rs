//! Qwen3.5 hybrid decoder against llama.cpp ground truth.
//!
//! `data/tiny-qwen35-f16.gguf` is a 36 KB randomly initialised `qwen35` model
//! (GDN + gated full-attention layers, F16 weights). The golden logits were
//! produced by llama.cpp (`tools/verify/llama_logits.c`, commit a25c9865f,
//! tokens 1..=7 fed one per decode call) and committed next to it.
//!
//! The test builds the generic C reference kernel with `cc`, resolves the
//! hybrid ops through the real dispatch table on the CPU backend, and requires
//! every logit to match llama.cpp. That pins the whole chain at once: ABI v7
//! op semantics, the generic kernels, tensor naming/layout handling, state
//! carry between tokens and `reset_cache`.

use std::path::{Path, PathBuf};
use std::process::Command;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;
use spite_models::hybrid::{HybridDecoder, is_hybrid};
use spite_models::{ModelArch, ModelConfig};

const TOKENS: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
/// llama.cpp computes with the same F16 weights but its own summation order.
const TOL: f32 = 5e-5;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Compile the generic kernel into `<tmp>/generic/generic/libkernel_generic.so`.
fn build_generic_kernel() -> PathBuf {
    let root = repo_root();
    let kdir = std::env::temp_dir().join(format!("spite-hybrid-golden-{}", std::process::id()));
    let out_dir = kdir.join("generic").join("generic");
    std::fs::create_dir_all(&out_dir).expect("create kernel dir");
    let mut cmd = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()));
    cmd.args(["-std=c11", "-O2", "-fPIC", "-shared"])
        .arg(format!("-I{}", root.display()))
        .arg(root.join("core/quant.c"));
    for f in ["kernel.c", "ops.c", "linear_attn.c"] {
        cmd.arg(root.join("kernels/generic/generic").join(f));
    }
    cmd.arg("-lm")
        .arg("-o")
        .arg(out_dir.join("libkernel_generic.so"));
    let status = cmd.status().expect("run C compiler");
    assert!(status.success(), "generic kernel failed to compile");
    kdir
}

fn golden() -> Vec<f32> {
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/tiny-qwen35-f16.llamacpp-logits.f32"),
    )
    .expect("golden logits");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run(model: &dyn ModelArch, vocab: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for (i, t) in TOKENS.iter().enumerate() {
        let mut logits = vec![0f32; vocab];
        let ctx = SpiteCtx {
            n_ctx: 0,
            n_batch: 1,
            n_threads: 1,
            pos: i as i32,
            n_heads: 0,
            n_kv_heads: 0,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: std::ptr::null_mut(),
            scratchpad_bytes: 0,
        };
        model.forward(&[*t], &mut logits, &ctx).expect("forward");
        out.extend(logits);
    }
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn hybrid_cpu_matches_llamacpp_logits() {
    let gguf = GgufModel::open(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny-qwen35-f16.gguf"),
    )
    .expect("open tiny gguf");
    let cfg = ModelConfig::from(spite_loader::config::ModelHyperparams::from_gguf(&gguf));
    assert!(is_hybrid(&cfg), "fixture must be a dense hybrid model");
    let vocab = cfg.vocab_size;
    let kdir = build_generic_kernel();
    let (table, backend) = HybridDecoder::resolve_table(&cfg.arch, "generic", &kdir, false)
        .expect("generic kernels provide every hybrid op");
    let (model, _) =
        HybridDecoder::load(cfg, &gguf, table, backend, 16).expect("load hybrid decoder");

    let want = golden();
    assert_eq!(want.len(), TOKENS.len() * vocab);
    let got = run(&model, vocab);
    let d = max_diff(&want, &got);
    assert!(
        d <= TOL,
        "logits differ from llama.cpp by {d:e} (tol {TOL:e})"
    );

    // A fresh sequence must start from zeroed recurrent state: same logits again.
    model.reset_cache();
    let again = run(&model, vocab);
    assert_eq!(got, again, "reset_cache did not restore the initial state");
    let _ = std::fs::remove_dir_all(kdir);
}
