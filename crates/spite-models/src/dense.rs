//! Shared dense-decoder forward pass (ported from llama.cpp `build_llama`).
//!
//! Every dense decoder-only arch is the same loop: embed → per layer
//! (rmsnorm → QKV → RoPE → causal GQA → out-proj → residual → rmsnorm →
//! SwiGLU FFN → residual) → final norm → LM head. Arch impls with that
//! structure call [`forward`] instead of re-implementing it. Archs with
//! MoE, MLA, sliding-window, or linear-attention layers need their own
//! forward and don't use this.
//!
//! Runs on CPU via spite-compute scalar fns.

use std::collections::HashMap;
use std::sync::RwLock;

use spite_abi::SpiteType;
use spite_compute::dequant::dequant_to_f32;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_kvcache::{KvQuant, KvQuantConfig, VbrPolicy, VbrRows};
use spite_loader::GgufModel;
use spite_rope::{RopeConfig, apply_rope};

use crate::{ModelConfig, ModelError};

// ponytail: whole-model dequant at load; per-token streaming dequant if memory matters.

/// One dequantized weight tensor, row-major F32.
pub struct Weight {
    pub data: Vec<f32>,
    /// `ne` from GGUF: ne[0]=cols, ne[1]=rows.
    pub ne: [u32; 4],
}

impl Weight {
    pub(crate) fn rows(&self) -> usize {
        self.ne[1].max(1) as usize
    }
    pub(crate) fn cols(&self) -> usize {
        self.ne[0].max(1) as usize
    }
}

/// All weights dequantized once at load; forward is then pure F32.
#[derive(Default)]
pub struct DenseWeights {
    tensors: HashMap<String, Weight>,
}

impl DenseWeights {
    /// Dequantize every tensor in `model`. Skipped tensors fail at forward.
    pub fn load(model: &GgufModel) -> Result<Self, ModelError> {
        let mut tensors = HashMap::new();
        for name in model.tensor_names() {
            let t = model.tensor(name);
            let n_elem: usize = t.ne.iter().map(|&d| d.max(1) as usize).product();
            let n_bytes = packed_bytes(t.kind, n_elem).ok_or_else(|| {
                ModelError::Forward(format!("unsupported weight dtype for {name}"))
            })?;
            let src = unsafe { std::slice::from_raw_parts(t.data as *const u8, n_bytes) };
            let mut data = vec![0f32; n_elem];
            dequant_to_f32(src, t.kind, n_elem, &mut data)
                .map_err(|e| ModelError::Forward(format!("dequant {name}: {e}")))?;
            tensors.insert(name.to_owned(), Weight { data, ne: t.ne });
        }
        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Result<&Weight, ModelError> {
        self.tensors
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.into()))
    }

    /// Build from explicit (data, shape) pairs. Test-only helper for
    /// constructing tiny synthetic models without a GGUF file.
    #[cfg(test)]
    pub fn from_map(map: HashMap<String, (Vec<f32>, [u32; 4])>) -> Self {
        Self {
            tensors: map
                .into_iter()
                .map(|(k, (data, ne))| (k, Weight { data, ne }))
                .collect(),
        }
    }
}

/// Packed byte length of `n_elem` elements of `kind` (any [`SpiteType`]).
pub(crate) fn packed_bytes(kind: SpiteType, n_elem: usize) -> Option<usize> {
    let n_blocks = n_elem.div_ceil(kind.block_elements() as usize);
    n_blocks.checked_mul(kind.block_bytes() as usize)
}

/// Per-layer K/V cache: post-RoPE keys, raw values, one row per position.
///
/// Rows are stored through [`VbrRows`], which can quantize and auto-degrade
/// them as the sequence grows. The default tier is full precision so the
/// reference forward path stays exact; the executor calls
/// [`set_quant`](KvStore::set_quant) to enable Variable Bit Rate compression
/// before the first token.
pub struct KvStore {
    k: Vec<VbrRows>,
    v: Vec<VbrRows>,
    key_quant: KvQuant,
    val_quant: KvQuant,
}

impl Default for KvStore {
    fn default() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            key_quant: KvQuant::F32,
            val_quant: KvQuant::F32,
        }
    }
}

impl KvStore {
    pub fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
    }

    /// Set the starting K/V tiers. Clears cache content so the new tiers apply
    /// from the first position.
    pub fn set_quant(&mut self, cfg: &KvQuantConfig) {
        self.key_quant = cfg.key;
        self.val_quant = cfg.val;
        self.k.clear();
        self.v.clear();
    }
}

/// FFN activation variants (llama.cpp `llm_ffn_op`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Activation {
    /// SwiGLU: `silu(gate) * up` — LLaMA, Mistral, Qwen, GLM.
    #[default]
    SwiGlu,
    /// GeGLU: `gelu(gate) * up` — Gemma.
    GeGlu,
}

/// Options selecting the dense-decoder variant. Defaults are the plain
/// full-attention SwiGLU decoder (Mistral-style).
#[derive(Debug, Clone, Copy)]
pub struct DenseOptions {
    pub activation: Activation,
    /// Sliding-window attention span (Gemma-style local layers). None = full.
    pub sliding_window: Option<usize>,
    /// Apply RoPE every `rope_stride`-th layer starting at 0. 1 = every
    /// layer; 2 = even layers only (LLaMA-4-style iRoPE NoPE layers).
    pub rope_stride: usize,
    /// Apply per-head QK RMSNorm (`attn_q_norm`/`attn_k_norm`) before RoPE.
    /// Enable for Qwen3/Qwen3.5-style decoders that have QK norms.
    pub apply_qk_norm: bool,
}

impl Default for DenseOptions {
    fn default() -> Self {
        Self {
            activation: Activation::SwiGlu,
            sliding_window: None,
            rope_stride: 1,
            apply_qk_norm: false,
        }
    }
}

/// tanh-approximated GELU, matching llama.cpp ggml.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

/// RMS norm over `x` with `weight`, matching llama.cpp `llm_graph` norm.
pub(crate) fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *o = v * scale * w;
    }
}

/// `out[r] = Σ_c w[c*rows + r] * x[c]` — GGUF weights are `[cols, rows]`.
/// Rows processed per accumulator block. Blocking over rows keeps each
/// column read contiguous (`w[c*rows + r0 .. r0 + n]`) instead of striding by
/// `rows`, which is what made the original row-at-a-time loop cache-bound.
const ROW_BLOCK: usize = 64;

/// `out[j] = Σ_c w[(row0 + j)*cols + c] * x[c]` for `j` in `0..out.len()`.
///
/// GGUF/ggml packs a tensor with `ne[0]` contiguous, so the row feeding output
/// index `r` starts at `r*ne[0]` and walks `c` linearly. Reading it as
/// `c*rows + r` instead transposes every non-square weight (attn_k, ffn_down,
/// the LM head) and silently produces plausible-looking garbage.
///
/// Accumulates over `c` in ascending order, exactly like the naive
/// row-at-a-time loop, so blocking does not change the floating-point result.
#[inline]
fn matvec_rows(w_data: &[f32], cols: usize, x: &[f32], out: &mut [f32], row0: usize) {
    let n = out.len();
    let mut acc = [0f32; ROW_BLOCK];
    acc[..n].fill(0.0);
    for (j, slot) in acc[..n].iter_mut().enumerate() {
        let base = (row0 + j) * cols;
        let mut a = 0f32;
        for (c, &xv) in x.iter().enumerate() {
            a += w_data[base + c] * xv;
        }
        *slot = a;
    }
    out.copy_from_slice(&acc[..n]);
}

/// Worker thread count for CPU matvec. GGUF weights are dequantized to F32 at
/// load, so every matvec is a dense pass over tens of MB; saturating the
/// available cores is worth the spawn cost.
fn matvec_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16)
}

pub(crate) fn matvec(w: &Weight, x: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
    let (rows, cols) = (w.rows(), w.cols());
    if x.len() != cols || out.len() != rows || w.data.len() != rows * cols {
        return Err(ModelError::Forward(format!(
            "matvec shape mismatch: x.len()={} cols={} out.len()={} rows={}",
            x.len(),
            cols,
            out.len(),
            rows
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let n_blocks = rows.div_ceil(ROW_BLOCK);
    let threads = matvec_threads().min(n_blocks);
    if threads <= 1 {
        for b in 0..n_blocks {
            let r0 = b * ROW_BLOCK;
            let r1 = (r0 + ROW_BLOCK).min(rows);
            matvec_rows(&w.data, cols, x, &mut out[r0..r1], r0);
        }
        return Ok(());
    }

    // Split the output into one contiguous slice per worker. The split size is
    // a whole number of row blocks so each worker's slice lines up with the
    // blocked inner loop.
    let per = n_blocks.div_ceil(threads) * ROW_BLOCK;
    let mut chunks: Vec<&mut [f32]> = out.chunks_mut(per).collect();
    let data = &w.data;
    std::thread::scope(|scope| {
        for (i, chunk) in chunks.iter_mut().enumerate() {
            let row0 = i * per;
            let len = chunk.len();
            scope.spawn(move || {
                let mut j = 0;
                while j < len {
                    let n = ROW_BLOCK.min(len - j);
                    matvec_rows(data, cols, x, &mut chunk[j..j + n], row0 + j);
                    j += n;
                }
            });
        }
    });
    Ok(())
}

/// Dense forward over `tokens` starting at absolute position `pos_base`.
///
/// `logits_out` is `[tokens.len() × vocab_size]`; appends K/V to `kv`.
/// Only the last position's logits are needed by callers, but computing all
/// keeps prefill and decode on one path.
pub fn forward(
    cfg: &ModelConfig,
    weights: &DenseWeights,
    kv: &RwLock<KvStore>,
    tokens: &[u32],
    pos_base: usize,
    logits_out: &mut [f32],
) -> Result<(), ModelError> {
    forward_with(
        cfg,
        weights,
        kv,
        tokens,
        pos_base,
        logits_out,
        &DenseOptions::default(),
    )
}

/// Dense forward over `tokens` starting at absolute position `pos_base`.
///
/// `logits_out` is `[tokens.len() × vocab_size]`; appends K/V to `kv`.
/// Only the last position's logits are needed by callers, but computing all
/// keeps prefill and decode on one path.
pub fn forward_with(
    cfg: &ModelConfig,
    weights: &DenseWeights,
    kv: &RwLock<KvStore>,
    tokens: &[u32],
    pos_base: usize,
    logits_out: &mut [f32],
    opts: &DenseOptions,
) -> Result<(), ModelError> {
    let d = cfg.d_model;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = d / n_heads;
    let d_ffn = cfg.d_ffn;
    let vocab = cfg.vocab_size;
    if logits_out.len() != tokens.len() * vocab {
        return Err(ModelError::Forward("logits_out shape mismatch".into()));
    }

    let rope = RopeConfig {
        head_dim,
        theta: cfg.rope_theta,
        ..Default::default()
    };
    let embd = weights.get("token_embd.weight")?;
    let out_norm = weights.get("output_norm.weight")?;
    let out_w = weights.get("output.weight")?;

    let mut kv = kv
        .write()
        .map_err(|_| ModelError::Forward("kv lock".into()))?;
    // ponytail: O(ctx²) scalar attention per token; blocked/GPU path if slow.
    for (ti, &tok) in tokens.iter().enumerate() {
        let pos = pos_base + ti;
        let mut h = vec![0f32; d];
        let row = &embd.data[tok as usize * d..(tok as usize + 1) * d];
        h.copy_from_slice(row);

        for layer in 0..cfg.n_layers {
            let b = format!("blk.{layer}");
            let w_norm = weights.get(&format!("{b}.attn_norm.weight"))?;
            let w_q = weights.get(&format!("{b}.attn_q.weight"))?;
            let w_k = weights.get(&format!("{b}.attn_k.weight"))?;
            let w_v = weights.get(&format!("{b}.attn_v.weight"))?;
            let w_o = weights.get(&format!("{b}.attn_output.weight"))?;
            let w_ffn_norm = weights.get(&format!("{b}.ffn_norm.weight"))?;
            let w_gate = weights.get(&format!("{b}.ffn_gate.weight"))?;
            let w_up = weights.get(&format!("{b}.ffn_up.weight"))?;
            let w_down = weights.get(&format!("{b}.ffn_down.weight"))?;

            // attn block
            let mut n = vec![0f32; d];
            rmsnorm(&h, &w_norm.data, cfg.norm_eps, &mut n);
            let mut q = vec![0f32; n_heads * head_dim];
            let mut k = vec![0f32; n_kv_heads * head_dim];
            let mut v = vec![0f32; n_kv_heads * head_dim];
            matvec(w_q, &n, &mut q)?;
            matvec(w_k, &n, &mut k)?;
            matvec(w_v, &n, &mut v)?;
            // Qwen3/Qwen3.5 style per-head QK RMSNorm (`attn_q_norm`/`attn_k_norm`).
            // Enabled when `apply_qk_norm` is set AND the weights exist.
            if opts.apply_qk_norm
                && let (Ok(w_qn), Ok(w_kn)) = (
                    weights.get(&format!("{b}.attn_q_norm.weight")),
                    weights.get(&format!("{b}.attn_k_norm.weight")),
                )
            {
                for h in 0..n_heads {
                    let mut out = vec![0.0; head_dim];
                    rmsnorm(
                        &q[h * head_dim..(h + 1) * head_dim],
                        &w_qn.data,
                        cfg.norm_eps,
                        &mut out,
                    );
                    q[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out);
                }
                for h in 0..n_kv_heads {
                    let mut out = vec![0.0; head_dim];
                    rmsnorm(
                        &k[h * head_dim..(h + 1) * head_dim],
                        &w_kn.data,
                        cfg.norm_eps,
                        &mut out,
                    );
                    k[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out);
                }
            }
            if layer % opts.rope_stride == 0 {
                apply_rope(&mut q, pos as u32, &rope)
                    .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
                apply_rope(&mut k, pos as u32, &rope)
                    .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
            }

            let row_len = n_kv_heads * head_dim;
            let (kq, vq) = (kv.key_quant, kv.val_quant);
            while kv.k.len() <= layer {
                kv.k.push(VbrRows::new(
                    row_len,
                    VbrPolicy::from_ctx(cfg.max_seq_len, kq),
                ));
                kv.v.push(VbrRows::new(
                    row_len,
                    VbrPolicy::from_ctx(cfg.max_seq_len, vq),
                ));
            }
            kv.k[layer].push(&k);
            kv.v[layer].push(&v);
            // Sliding window: attend only to the trailing span. Rows are
            // dequantized on demand, costing O(ctx) per layer per token — the
            // same order as the scalar attention it feeds.
            let n_prev = kv.k[layer].len();
            let n_attend = opts.sliding_window.map_or(n_prev, |w| n_prev.min(w));
            let from = n_prev - n_attend;
            let kk = kv.k[layer].to_f32_from(from);
            let vv = kv.v[layer].to_f32_from(from);

            let attn_cfg = FlashAttnConfig::new(1, n_attend, n_heads, n_kv_heads, head_dim);
            let mut attn_out = vec![0f32; n_heads * head_dim];
            scalar_attention(&q, &kk, &vv, &mut attn_out, &attn_cfg)
                .map_err(|e| ModelError::Forward(format!("attn: {e}")))?;
            let mut proj = vec![0f32; d];
            matvec(w_o, &attn_out, &mut proj)?;
            for (h_i, &p) in h.iter_mut().zip(proj.iter()) {
                *h_i += p;
            }

            // ffn block
            rmsnorm(&h, &w_ffn_norm.data, cfg.norm_eps, &mut n);
            let mut gate = vec![0f32; d_ffn];
            let mut up = vec![0f32; d_ffn];
            matvec(w_gate, &n, &mut gate)?;
            matvec(w_up, &n, &mut up)?;
            for (g, &u) in gate.iter_mut().zip(up.iter()) {
                let a = match opts.activation {
                    Activation::SwiGlu => *g / (1.0 + (-*g).exp()),
                    Activation::GeGlu => gelu(*g),
                };
                *g = a * u;
            }
            let mut down = vec![0f32; d];
            matvec(w_down, &gate, &mut down)?;
            for (h_i, &dl) in h.iter_mut().zip(down.iter()) {
                *h_i += dl;
            }
        }

        let mut n = vec![0f32; d];
        rmsnorm(&h, &out_norm.data, cfg.norm_eps, &mut n);
        let dst = &mut logits_out[ti * vocab..(ti + 1) * vocab];
        matvec(out_w, &n, dst)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spite_loader::GgufModel;

    pub(crate) fn test_config() -> ModelConfig {
        ModelConfig {
            arch: "mistral4".into(),
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            d_model: 64,
            d_ffn: 128,
            vocab_size: 32,
            max_seq_len: 512,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            ..Default::default()
        }
    }

    #[test]
    fn forward_fake_gguf_finite_and_deterministic() {
        let tmp = spite_testkit::FakeGguf::default()
            .write_to_tempfile()
            .unwrap();
        let model = GgufModel::open(tmp.path()).unwrap();
        let weights = DenseWeights::load(&model).unwrap();
        let kv = RwLock::new(KvStore::default());
        let cfg = test_config();

        let mut logits_a = vec![0f32; 2 * cfg.vocab_size];
        forward(&cfg, &weights, &kv, &[1, 2], 0, &mut logits_a).unwrap();
        assert!(logits_a.iter().all(|v| v.is_finite()));

        // Same prompt from a fresh cache reproduces logits exactly.
        let kv2 = RwLock::new(KvStore::default());
        let mut logits_b = vec![0f32; 2 * cfg.vocab_size];
        forward(&cfg, &weights, &kv2, &[1, 2], 0, &mut logits_b).unwrap();
        assert_eq!(logits_a, logits_b);

        // Incremental decode matches full prefill for the shared prefix.
        let kv3 = RwLock::new(KvStore::default());
        let mut pre = vec![0f32; cfg.vocab_size];
        forward(&cfg, &weights, &kv3, &[1], 0, &mut pre).unwrap();
        let mut step = vec![0f32; cfg.vocab_size];
        forward(&cfg, &weights, &kv3, &[2], 1, &mut step).unwrap();
        assert!(logits_a[..cfg.vocab_size] == pre[..]);
        assert!(logits_a[cfg.vocab_size..] == step[..]);
    }
}

#[cfg(test)]
mod options_tests {
    use super::tests::test_config;
    use super::*;
    use spite_loader::GgufModel;

    #[test]
    fn geglu_windowed_stride_forward_is_finite() {
        let tmp = spite_testkit::FakeGguf::default()
            .write_to_tempfile()
            .unwrap();
        let model = GgufModel::open(tmp.path()).unwrap();
        let weights = DenseWeights::load(&model).unwrap();
        let kv = RwLock::new(KvStore::default());
        let cfg = test_config();
        let opts = DenseOptions {
            activation: Activation::GeGlu,
            sliding_window: Some(1),
            rope_stride: 2,
            apply_qk_norm: false,
        };
        let mut logits = vec![0f32; 2 * cfg.vocab_size];
        forward_with(&cfg, &weights, &kv, &[1, 2], 0, &mut logits, &opts).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
