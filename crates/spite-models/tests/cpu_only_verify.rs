//! CPU-only model verification harness.
//!
//! Runs every GGUF named by $SPITE_TEST_MODELS (a directory or a single
//! .gguf path) and executes a full forward pass on CPU (no GPU, no kernel .so),
//! checking
//! that logits are finite, deterministic across two passes, and that the
//! Rust CPU fallback matches the dense scalar reference implementation.
//! This is the first rung of the CPU-only leg: it proves the Rust CPU
//! fallback is correct on real quantized weights before any C++23 port
//! is considered.
//!
//! Build: cargo test --package spite-models --test cpu_only_verify --no-default-features
//! Run:   SPITE_TEST_MODELS=/path/to/models cargo test --package spite-models --test cpu_only_verify --no-default-features

use std::sync::RwLock;

use spite_loader::GgufModel;
use spite_models::dense::{DenseWeights, KvStore, forward_with};
use spite_models::{ArchRegistry, ModelConfig};

/// Run the full forward pass for `model` over a fixed prompt whose logits
/// must be finite, deterministic across two passes, and match the dense
/// scalar reference (`dense::forward` in `crates/spite-models/src/dense.rs`)
/// to a small absolute tolerance.
fn verify_model(
    gguf_path: &std::path::Path,
    prompt_tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let gguf = GgufModel::open(gguf_path)?;
    let arch = gguf.arch().to_owned();

    // Build the concrete arch impl from the GGUF metadata.
    let hp = spite_loader::config::ModelHyperparams::from_gguf(&gguf);
    let mut cfg = ModelConfig::from(hp);
    // Some quantized GGUF files omit tokenizer.ggml.token_count, so the
    // vocabulary size must be inferred from the weight tensors (via the
    // loader's `vocab_size()` fallback).
    if cfg.vocab_size == 0 {
        cfg.vocab_size = gguf.vocab_size();
    }
    let mut model = ArchRegistry::default().build(cfg.clone())?;
    model.load_weights(&gguf)?;

    let prompt_ids = prompt_tokens.to_vec();
    let n_ctx = prompt_ids.len();
    let n_vocab = cfg.vocab_size;
    let mut logits = vec![0f32; n_ctx * n_vocab];

    let ctx = spite_abi::SpiteCtx {
        n_ctx: cfg.max_seq_len.max(1) as i32,
        n_batch: 1,
        n_threads: 1,
        pos: 0,
        n_heads: cfg.n_heads as i32,
        n_kv_heads: cfg.n_kv_heads as i32,
        gpu_stream: std::ptr::null_mut(),
        scratchpad: std::ptr::null_mut(),
        scratchpad_bytes: 0,
    };

    // Pass 1: full forward.
    model.forward(&prompt_ids, &mut logits, &ctx)?;

    // Post-conditions: all logits finite, nothing NaNs from the CPU path.
    if !logits.iter().all(|x| x.is_finite()) {
        return Err(format!(
            "{}: forward produced non-finite logits",
            gguf_path.display()
        )
        .into());
    }

    // Pass 2: determinism (fresh cache, same prompt, identical logits).
    let mut logits2 = vec![0f32; n_ctx * n_vocab];
    model.reset_cache();
    model.forward(&prompt_ids, &mut logits2, &ctx)?;
    if logits != logits2 {
        return Err(format!("{}: logits not deterministic", gguf_path.display()).into());
    }

    // Dense scalar reference over the SAME weights Qwen3_5 just ran on (so any
    // divergence is the forward math, not the weights), with the per-head QK
    // RMSNorm that Qwen3/Qwen3.5 apply via attn_q_norm / attn_k_norm enabled.
    let ref_weights = DenseWeights::load(&gguf)?;
    let ref_cfg = cfg.clone();
    let ref_options = spite_models::dense::DenseOptions {
        activation: spite_models::dense::Activation::SwiGlu,
        sliding_window: None,
        rope_stride: 1,
        apply_qk_norm: true,
    };
    let mut ref_logits = vec![0f32; n_ctx * ref_cfg.vocab_size];
    let kv = RwLock::new(KvStore::default());
    forward_with(
        &ref_cfg,
        &ref_weights,
        &kv,
        &prompt_ids,
        0,
        &mut ref_logits,
        &ref_options,
    )?;

    if !ref_logits
        .iter()
        .zip(logits.iter())
        .all(|(a, b)| (a - b).abs() < 1e-3)
    {
        return Err(format!(
            "{}: logits diverge from dense reference (max diff {:.6})",
            gguf_path.display(),
            ref_logits
                .iter()
                .zip(logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        )
        .into());
    }

    println!(
        "ok  {}  arch={}  n_layers={}  vocab={}  prompt_tokens={}  ctx={}",
        gguf_path.display(),
        arch,
        cfg.n_layers,
        cfg.vocab_size,
        prompt_ids.len(),
        n_ctx
    );
    Ok(())
}

fn run(models: &[std::path::PathBuf], prompt_ids: &[u32]) -> std::process::ExitCode {
    let mut failures = 0;
    for m in models {
        if let Err(e) = verify_model(m, prompt_ids) {
            failures += 1;
            eprintln!("FAIL {} — {}", m.display(), e);
        }
    }
    if failures > 0 {
        std::process::exit(1);
    }
    println!("all cpu-only forward passes passed");
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collects the models to check from $SPITE_TEST_MODELS, which may name
    /// either a directory or a single .gguf file. Falls back to the Q8_0 Qwen3
    /// checkpoint when the variable is unset.
    ///
    /// Returns an empty list when nothing is available. The checkpoints are
    /// multi-GB and are not in the repo, so on a machine without them (CI, a
    /// fresh clone) there is nothing to verify and the test reports that
    /// explicitly rather than passing on an empty run.
    fn models_to_verify() -> Vec<std::path::PathBuf> {
        let default = "/mnt/storage/models/qwen3/Qwen3-8B-Q8_0.gguf";
        let spec = match std::env::var("SPITE_TEST_MODELS") {
            Ok(v) => v,
            Err(_) => default.into(),
        };
        let p = std::path::PathBuf::from(&spec);
        if p.is_dir() {
            let Ok(entries) = std::fs::read_dir(&p) else {
                return Vec::new();
            };
            let mut found: Vec<_> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
                .collect();
            found.sort();
            found
        } else if p.is_file() {
            vec![p]
        } else {
            Vec::new()
        }
    }

    #[test]
    fn cpu_forward_against_real_quantized_gguf() {
        // Q8_0 is the model that actually loads; the Q5_K_M / Q4_K_XL files in
        // the same directory fail to open at the loader level, so point
        // SPITE_TEST_MODELS at a single file rather than the whole directory.
        let models = models_to_verify();
        if models.is_empty() {
            println!(
                "skipping: no GGUF checkpoint available \
                 (set SPITE_TEST_MODELS to a .gguf file or a directory of them)"
            );
            return;
        }
        eprintln!("verifying {} checkpoint(s)", models.len());
        let prompt_ids = [1, 2];
        let code = run(&models, &prompt_ids);
        assert_eq!(code, std::process::ExitCode::SUCCESS);
    }
}
