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

/// Packed byte length of `n_elem` elements, or None if unsupported.
/// Block sizes from llama.cpp ggml.
fn packed_bytes(kind: SpiteType, n_elem: usize) -> Option<usize> {
    let (el_per_block, bytes_per_block): (usize, usize) = match kind {
        SpiteType::F32 => return n_elem.checked_mul(4),
        SpiteType::F16 | SpiteType::Bf16 => return n_elem.checked_mul(2),
        SpiteType::Q8_0 => (32, 34),
        SpiteType::Q4_0 => (32, 18),
        SpiteType::Q4K => (256, 144),
        SpiteType::Q5K => (256, 176),
        SpiteType::Q6K => (256, 210),
        _ => return None,
    };
    let n_blocks = n_elem.div_ceil(el_per_block);
    n_blocks.checked_mul(bytes_per_block)
}

/// Per-layer K/V cache: post-RoPE keys, raw values, one row per position.
#[derive(Default)]
pub struct KvStore {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvStore {
    pub fn reset(&mut self) {
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
}

impl Default for DenseOptions {
    fn default() -> Self {
        Self {
            activation: Activation::SwiGlu,
            sliding_window: None,
            rope_stride: 1,
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
pub(crate) fn matvec(w: &Weight, x: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
    let (rows, cols) = (w.rows(), w.cols());
    if x.len() != cols || out.len() != rows || w.data.len() != rows * cols {
        return Err(ModelError::Forward("matvec shape mismatch".into()));
    }
    for (r, o) in out.iter_mut().enumerate() {
        let mut acc = 0f32;
        for (c, &xv) in x.iter().enumerate() {
            acc += w.data[c * rows + r] * xv;
        }
        *o = acc;
    }
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
            if layer % opts.rope_stride == 0 {
                apply_rope(&mut q, pos as u32, &rope)
                    .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
                apply_rope(&mut k, pos as u32, &rope)
                    .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
            }

            while kv.k.len() <= layer {
                kv.k.push(Vec::new());
                kv.v.push(Vec::new());
            }
            kv.k[layer].extend_from_slice(&k);
            kv.v[layer].extend_from_slice(&v);
            // Sliding window: attend only to the trailing span.
            let row_len = n_kv_heads * head_dim;
            let n_prev = kv.k[layer].len() / row_len;
            let (kk, vv) = match opts.sliding_window {
                Some(w) if n_prev > w => {
                    let skip = (n_prev - w) * row_len;
                    (&kv.k[layer][skip..], &kv.v[layer][skip..])
                }
                _ => (kv.k[layer].as_slice(), kv.v[layer].as_slice()),
            };

            let attn_cfg = FlashAttnConfig::new(
                1,
                n_prev.min(opts.sliding_window.unwrap_or(n_prev)),
                n_heads,
                n_kv_heads,
                head_dim,
            );
            let mut attn_out = vec![0f32; n_heads * head_dim];
            scalar_attention(&q, kk, vv, &mut attn_out, &attn_cfg)
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
        };
        let mut logits = vec![0f32; 2 * cfg.vocab_size];
        forward_with(&cfg, &weights, &kv, &[1, 2], 0, &mut logits, &opts).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
