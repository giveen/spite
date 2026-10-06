//! Dump per-position logits of a hybrid (Qwen3.5-style) GGUF to a raw f32 file,
//! for comparison against llama.cpp (`tools/verify/llama_logits.c`).
//!
//! cargo run --release -p spite-models --example dump_logits -- \
//!     MODEL.gguf OUT.f32 {cpu|gpu} KERNELS_DIR GPU_ARCH TOK0 TOK1 ...

use std::path::Path;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;
use spite_models::hybrid::HybridDecoder;
use spite_models::{ModelArch, ModelConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        return Err("usage: dump_logits MODEL OUT {cpu|gpu} KERNELS_DIR GPU_ARCH TOK...".into());
    }
    let gguf = GgufModel::open(Path::new(&a[1]))?;
    let cfg = ModelConfig::from(spite_loader::config::ModelHyperparams::from_gguf(&gguf));
    let tokens: Vec<u32> = a[6..].iter().map(|t| t.parse()).collect::<Result<_, _>>()?;
    let (table, backend) =
        HybridDecoder::resolve_table(&cfg.arch, &a[5], Path::new(&a[4]), a[3] == "gpu")
            .ok_or("no kernels provide every hybrid op")?;
    let vocab = cfg.vocab_size;
    let (model, r) = HybridDecoder::load(cfg, &gguf, table, backend, tokens.len() + 8)?;
    eprintln!("backend {:?}, kv {:?}", r.backend, r.kv_kind);
    let mut out = Vec::with_capacity(tokens.len() * vocab * 4);
    let ctx = SpiteCtx {
        n_ctx: 0,
        n_batch: 1,
        n_threads: 1,
        pos: 0,
        n_heads: 0,
        n_kv_heads: 0,
        gpu_stream: std::ptr::null_mut(),
        scratchpad: std::ptr::null_mut(),
        scratchpad_bytes: 0,
    };
    for (i, t) in tokens.iter().enumerate() {
        let mut logits = vec![0f32; vocab];
        let c = SpiteCtx {
            pos: i as i32,
            ..ctx
        };
        model.forward(&[*t], &mut logits, &c)?;
        for v in &logits {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(&a[2], out)?;
    Ok(())
}
