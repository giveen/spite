//! Real-model Variable Bit Rate (VBR) KV-cache verification.
//!
//! Loads the Qwen3 checkpoint named by $SPITE_TEST_MODELS (defaulting to the
//! Q8_0 file below when the variable is unset), enables VBR at f16, and runs a
//! short forward pass. The context window is set small so the degradation
//! thresholds fire inside the prompt, and the test asserts the full-attention
//! layers actually stepped down the ladder while producing finite logits.
//!
//! This is the end-to-end rung for the VBR feature on real quantized weights:
//! the in-crate unit test exercises the ladder on synthetic tensors, this one
//! exercises quantize → degrade → dequantize → attention on a real model.
//!
//! Run: cargo test --package spite-models --test vbr_kv_cache

use spite_abi::SpiteCtx;
use spite_kvcache::{KvQuant, KvQuantConfig};
use spite_loader::GgufModel;
use spite_models::qwen::Qwen3_5;
use spite_models::{ModelArch, ModelConfig};

/// Resolve the checkpoint to test, or `None` when it is not on this machine.
fn model_path() -> Option<std::path::PathBuf> {
    let default = "/mnt/storage/models/qwen3/Qwen3-8B-Q8_0.gguf";
    let spec = std::env::var("SPITE_TEST_MODELS").unwrap_or_else(|_| default.into());
    let p = std::path::PathBuf::from(spec);
    p.is_file().then_some(p)
}

#[test]
fn vbr_degrades_real_model_kv_cache() {
    let Some(path) = model_path() else {
        println!(
            "skipping: no GGUF checkpoint available \
             (set SPITE_TEST_MODELS to a Qwen3 .gguf file)"
        );
        return;
    };

    let gguf = GgufModel::open(&path).unwrap();
    let hp = spite_loader::config::ModelHyperparams::from_gguf(&gguf);
    let mut cfg = ModelConfig::from(hp);
    if cfg.vocab_size == 0 {
        cfg.vocab_size = gguf.vocab_size();
    }
    // Small window => thresholds at 2, 4 and 6 tokens; the 8-token prompt
    // crosses all of them so the cache reaches the q4 floor.
    cfg.max_seq_len = 8;

    let mut model = Qwen3_5::new(cfg.clone());
    model.load_weights(&gguf).unwrap();
    model.set_kv_quant(KvQuantConfig {
        key: KvQuant::F16,
        val: KvQuant::F16,
    });

    let ctx = SpiteCtx {
        n_ctx: cfg.max_seq_len as i32,
        n_batch: 1,
        n_threads: 1,
        pos: 0,
        n_heads: cfg.n_heads as i32,
        n_kv_heads: cfg.n_kv_heads as i32,
        gpu_stream: std::ptr::null_mut(),
        scratchpad: std::ptr::null_mut(),
        scratchpad_bytes: 0,
    };

    let tokens: Vec<u32> = (0..8).map(|i| (i % cfg.vocab_size) as u32).collect();
    let mut logits = vec![0f32; tokens.len() * cfg.vocab_size];
    model.forward(&tokens, &mut logits, &ctx).unwrap();

    assert!(
        logits.iter().all(|x| x.is_finite()),
        "VBR forward produced non-finite logits"
    );

    let tiers = model.kv_key_tiers();
    assert_eq!(tiers.len(), cfg.n_layers, "one tier per attention layer");
    assert!(
        tiers.iter().all(|&q| q == KvQuant::Q4),
        "expected every layer at the q4 floor after 8 tokens, got {tiers:?}"
    );
}
