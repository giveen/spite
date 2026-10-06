//! NextN (MTP) draft head of the hybrid Qwen3.5 decoder, on the CPU backend.
//!
//! Fixtures (see `tests/data/make_tiny_qwen35_mtp.py`) are the llama.cpp-pinned
//! `tiny-qwen35-f16.gguf` trunk plus one NextN block:
//!
//! * `identity`: the block reduces to `x = rmsnorm(h_t)`, so the draft logits
//!   must be a positive multiple of the trunk logits that produced `h_t`. That
//!   pins the stem's `[embed || hidden]` packing, the hand-off of the trunk's
//!   final-normed hidden state, the eh_proj / shared-head wiring, and that
//!   loading the block leaves the trunk untouched (llama.cpp golden logits).
//! * `full`: live weights, to check the draft depends on the token and that a
//!   replay after `reset_cache` is bit-identical.

use std::path::{Path, PathBuf};
use std::process::Command;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;
use spite_models::hybrid::{HybridDecoder, LayerSplit, StageReport};
use spite_models::{ModelArch, ModelConfig};

const TOKENS: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
const TOL: f32 = 5e-5;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// Compile the generic kernel into `<tmp>/generic/generic/libkernel_generic.so`.
fn build_generic_kernel(tag: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let kdir = std::env::temp_dir().join(format!("spite-hybrid-mtp-{tag}-{}", std::process::id()));
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
    assert!(
        cmd.status().expect("run C compiler").success(),
        "generic kernel failed to compile"
    );
    kdir
}

fn load(file: &str, tag: &str) -> (HybridDecoder, usize, PathBuf) {
    let kdir = build_generic_kernel(tag);
    let (model, vocab, _) = load_split(file, &kdir, &LayerSplit::default());
    (model, vocab, kdir)
}

fn load_split(
    file: &str,
    kdir: &Path,
    split: &LayerSplit,
) -> (HybridDecoder, usize, Vec<StageReport>) {
    let gguf = GgufModel::open(data(file)).expect("open fixture");
    let cfg = ModelConfig::from(spite_loader::config::ModelHyperparams::from_gguf(&gguf));
    let vocab = cfg.vocab_size;
    let (table, backend) = HybridDecoder::resolve_table(&cfg.arch, "generic", kdir, false)
        .expect("generic kernels provide every hybrid op");
    let (model, report) = HybridDecoder::load_split(cfg, &gguf, table, backend, 16, split)
        .expect("load hybrid decoder");
    assert_eq!(report.mtp, model.has_mtp());
    (model, vocab, report.stages)
}

fn ctx(pos: usize) -> SpiteCtx {
    SpiteCtx {
        n_ctx: 0,
        n_batch: 1,
        n_threads: 1,
        pos: pos as i32,
        n_heads: 0,
        n_kv_heads: 0,
        gpu_stream: std::ptr::null_mut(),
        scratchpad: std::ptr::null_mut(),
        scratchpad_bytes: 0,
    }
}

/// Trunk over TOKENS with one MTP step after every token (drafting the next
/// prompt token, or `last_draft` after the final one). Returns (trunk, draft)
/// logits per position.
fn run(model: &HybridDecoder, vocab: usize, last_draft: u32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (mut trunk, mut draft) = (Vec::new(), Vec::new());
    for (pos, &tok) in TOKENS.iter().enumerate() {
        let mut t = vec![0f32; vocab];
        model.forward(&[tok], &mut t, &ctx(pos)).expect("forward");
        let next = TOKENS.get(pos + 1).copied().unwrap_or(last_draft);
        let mut dl = vec![0f32; vocab];
        model.mtp_step(next, pos, &mut dl).expect("mtp_step");
        trunk.push(t);
        draft.push(dl);
    }
    (trunk, draft)
}

/// Max deviation of `y` from the best fit `c * x`, relative to max |y|; and c.
fn proportional(x: &[f32], y: &[f32]) -> (f32, f32) {
    let c = x.iter().zip(y).map(|(a, b)| a * b).sum::<f32>() / x.iter().map(|a| a * a).sum::<f32>();
    let scale = y.iter().fold(0f32, |m, v| m.max(v.abs()));
    let dev = x
        .iter()
        .zip(y)
        .fold(0f32, |m, (a, b)| m.max((b - c * a).abs()));
    (dev / scale, c)
}

#[test]
fn no_nextn_block_means_no_mtp() {
    let (model, vocab, kdir) = load("tiny-qwen35-f16.gguf", "none");
    assert!(!model.has_mtp());
    assert!(model.mtp_step(1, 0, &mut vec![0f32; vocab]).is_err());
    let _ = std::fs::remove_dir_all(kdir);
}

#[test]
fn identity_head_tracks_trunk_hidden() {
    let (model, vocab, kdir) = load("tiny-qwen35-mtp-identity-f16.gguf", "identity");
    assert!(model.has_mtp(), "fixture carries a usable NextN block");
    let (trunk, draft) = run(&model, vocab, 9);

    let golden: Vec<f32> = std::fs::read(data("tiny-qwen35-f16.llamacpp-logits.f32"))
        .expect("golden logits")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let got: Vec<f32> = trunk.iter().flatten().copied().collect();
    let d = got
        .iter()
        .zip(&golden)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        d <= TOL,
        "trunk drifted from llama.cpp by {d:e} with the NextN block loaded"
    );

    for (pos, (t, dl)) in trunk.iter().zip(&draft).enumerate() {
        let (dev, c) = proportional(t, dl);
        assert!(
            c > 0.0 && dev < 1e-4,
            "pos {pos}: draft not c*trunk (c={c}, rel dev {dev:e})"
        );
    }

    // Chained step: the head's own normed output is the next h_t, and for this
    // head that is still a rescaling of the last trunk hidden state.
    let mut chained = vec![0f32; vocab];
    model
        .mtp_step(3, TOKENS.len(), &mut chained)
        .expect("chained mtp_step");
    let (dev, c) = proportional(&trunk[TOKENS.len() - 1], &chained);
    assert!(c > 0.0 && dev < 1e-4, "chained step: rel dev {dev:e}");
    let _ = std::fs::remove_dir_all(kdir);
}

#[test]
fn full_head_depends_on_token_and_replays_exactly() {
    let (model, vocab, kdir) = load("tiny-qwen35-mtp-full-f16.gguf", "full");
    assert!(model.has_mtp());
    let (_, a) = run(&model, vocab, 9);
    assert!(a.iter().flatten().all(|v| v.is_finite()));

    model.reset_cache();
    let (_, b) = run(&model, vocab, 9);
    assert_eq!(a, b, "replay after reset_cache must be bit-identical");

    model.reset_cache();
    let (_, c) = run(&model, vocab, 10);
    assert_eq!(a[..TOKENS.len() - 1], c[..TOKENS.len() - 1]);
    assert_ne!(
        a[TOKENS.len() - 1],
        c[TOKENS.len() - 1],
        "draft ignores its input token"
    );
    let _ = std::fs::remove_dir_all(kdir);
}

/// A pipeline split hands the residual stream between stages and runs the
/// head and the NextN block on the last stage. On the CPU backend the stages
/// share memory, so every trunk and draft logit must match the unsplit
/// decoder exactly.
#[test]
fn layer_split_matches_single_stage() {
    let file = "tiny-qwen35-mtp-full-f16.gguf";
    let kdir = build_generic_kernel("split");
    let (single, vocab, stages) = load_split(file, &kdir, &LayerSplit::default());
    assert_eq!(stages.len(), 1);
    let n_layers = stages[0].layers.end;
    assert!(n_layers >= 2, "fixture needs two trunk layers to split");
    let (want_trunk, want_draft) = run(&single, vocab, 9);

    let split = LayerSplit {
        devices: vec![0, 0],
        shares: vec![1, 1],
    };
    let (piped, _, stages) = load_split(file, &kdir, &split);
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].layers.start, 0);
    assert_eq!(stages[0].layers.end, stages[1].layers.start);
    assert_eq!(stages[1].layers.end, n_layers);
    assert!(stages.iter().all(|s| !s.layers.is_empty()));
    let (trunk, draft) = run(&piped, vocab, 9);
    assert_eq!(trunk, want_trunk, "split trunk logits differ");
    assert_eq!(draft, want_draft, "split MTP logits differ");

    piped.reset_cache();
    assert_eq!(run(&piped, vocab, 9), (want_trunk, want_draft));
    let _ = std::fs::remove_dir_all(kdir);
}

#[test]
fn layer_split_rejects_bad_shares() {
    let file = "tiny-qwen35-mtp-full-f16.gguf";
    let kdir = build_generic_kernel("badsplit");
    let gguf = GgufModel::open(data(file)).expect("open fixture");
    for (devices, shares) in [
        (vec![0, 0], vec![1]),
        (vec![0, 0], vec![0, 0]),
        (vec![0, 0], vec![1000, 1]),
    ] {
        let cfg = ModelConfig::from(spite_loader::config::ModelHyperparams::from_gguf(&gguf));
        let (table, backend) = HybridDecoder::resolve_table(&cfg.arch, "generic", &kdir, false)
            .expect("generic kernels");
        let split = LayerSplit { devices, shares };
        assert!(
            HybridDecoder::load_split(cfg, &gguf, table, backend, 16, &split).is_err(),
            "{split:?} must be rejected"
        );
    }
    let _ = std::fs::remove_dir_all(kdir);
}

/// The layer-major batched prefill path (one op call per layer for the whole
/// prompt) must produce the same logits as feeding the prompt one token at a
/// time. Runs on the generic reference, which is the only batch-capable kernel.
#[test]
fn batched_prefill_matches_sequential() {
    let (model, vocab, kdir) = load("tiny-qwen35-f16.gguf", "batch");
    let tokens = [1u32, 2, 3, 4, 5];

    model.reset_cache();
    let mut batched = vec![0f32; tokens.len() * vocab];
    model
        .forward(&tokens, &mut batched, &ctx(0))
        .expect("batched forward");

    model.reset_cache();
    let mut seq = Vec::with_capacity(tokens.len() * vocab);
    for (i, &t) in tokens.iter().enumerate() {
        let mut l = vec![0f32; vocab];
        model.forward(&[t], &mut l, &ctx(i)).expect("forward");
        seq.extend_from_slice(&l);
    }

    let d = batched
        .iter()
        .zip(&seq)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(d < 1e-6, "batched vs sequential logits differ by {d:e}");
    let _ = std::fs::remove_dir_all(kdir);
}
