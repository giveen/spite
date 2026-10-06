//! Dense Qwen3 CPU reference decoder — GGUF arch `qwen3`.
//!
//! Plain GQA transformer: per-head QK RMSNorm, neox RoPE, causal attention
//! over a VBR-quantized KV cache, `ffn_norm` + SwiGLU FFN. Qwen3.5 hybrid
//! (`qwen35`) models run through `crate::hybrid` instead.
//!
//! (The type keeps its historical name `Qwen3_5`; it is the `qwen3` decoder.)

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_kvcache::{KvQuant, KvQuantConfig, VbrPolicy, VbrRows};
use spite_loader::GgufModel;

use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::{ModelArch, ModelConfig, ModelError};

struct LayerState {
    k: VbrRows,
    v: VbrRows,
}

/// Per-head width. Qwen3-family GGUFs store it as `attention.key_length`,
/// which need not equal `d_model / n_heads` (e.g. 64 heads x 128 on a
/// 5120-wide model); older files without the key fall back to the quotient.
fn qwen3_head_dim(cfg: &ModelConfig) -> usize {
    if cfg.key_length > 0 {
        cfg.key_length
    } else {
        cfg.d_model / cfg.n_heads.max(1)
    }
}

pub struct Qwen3_5 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    /// KV quantization policy. Full precision until the executor enables VBR.
    kv_quant: RwLock<KvQuantConfig>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    state: RwLock<Vec<LayerState>>,
}

impl Qwen3_5 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv_quant: RwLock::new(KvQuantConfig::full_precision()),
            state: RwLock::new(Vec::new()),
        }
    }

    /// Current K-cache storage tier for each layer.
    ///
    /// Exposed for diagnostics and tests: it makes VBR degradation observable
    /// from outside the crate without reaching into the layer state.
    pub fn kv_key_tiers(&self) -> Vec<KvQuant> {
        self.state
            .read()
            .map(|s| s.iter().map(|l| l.k.quant()).collect())
            .unwrap_or_default()
    }
}

impl ModelArch for Qwen3_5 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    fn reset_cache(&self) {
        if let Ok(mut s) = self.state.write() {
            s.clear();
        }
    }

    fn set_kv_quant(&self, cfg: KvQuantConfig) {
        if let Ok(mut q) = self.kv_quant.write() {
            *q = cfg;
        }
        // Existing rows were packed at the old tier; drop them.
        self.reset_cache();
    }

    fn forward(
        &self,
        tokens: &[u32],
        logits_out: &mut [f32],
        ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let Some(w) = &self.weights else {
            return Err(ModelError::Forward("load_weights not called".into()));
        };
        let cfg = &self.config;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = qwen3_head_dim(cfg);
        let d_ffn = cfg.d_ffn;
        let vocab = cfg.vocab_size;
        if logits_out.len() != tokens.len() * vocab {
            return Err(ModelError::Forward("logits_out shape mismatch".into()));
        }
        // ponytail: O(ctx²) scalar CPU path; GPU kernels own speed.

        let embd = w.get("token_embd.weight")?;
        let out_norm = w.get("output_norm.weight")?;
        let out_w = w.get("output.weight")?;

        let mut state = self
            .state
            .write()
            .map_err(|_| ModelError::Forward("state lock".into()))?;
        let (kq, vq) = self
            .kv_quant
            .read()
            .map(|c| (c.key, c.val))
            .unwrap_or((KvQuant::F32, KvQuant::F32));
        let kv_row_len = n_kv_heads * head_dim;
        while state.len() < cfg.n_layers {
            state.push(LayerState {
                k: VbrRows::new(kv_row_len, VbrPolicy::from_ctx(cfg.max_seq_len, kq)),
                v: VbrRows::new(kv_row_len, VbrPolicy::from_ctx(cfg.max_seq_len, vq)),
            });
        }

        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            let mut h = vec![0f32; d];
            h.copy_from_slice(&embd.data[tok as usize * d..(tok as usize + 1) * d]);

            for layer in 0..cfg.n_layers {
                let b = format!("blk.{layer}");
                let w_norm = w.get(&format!("{b}.attn_norm.weight"))?;
                let w_post = w.get(&format!("{b}.ffn_norm.weight"))?;
                let w_gate = w.get(&format!("{b}.ffn_gate.weight"))?;
                let w_up = w.get(&format!("{b}.ffn_up.weight"))?;
                let w_down = w.get(&format!("{b}.ffn_down.weight"))?;

                let mut n = vec![0f32; d];
                rmsnorm(&h, &w_norm.data, cfg.norm_eps, &mut n);

                let attn_out = full_attn(
                    w,
                    &b,
                    &n,
                    pos,
                    &mut state[layer],
                    cfg,
                    head_dim,
                    n_heads,
                    n_kv_heads,
                )?;

                for (h_i, &p) in h.iter_mut().zip(attn_out.iter()) {
                    *h_i += p;
                }
                // Post-attention norm → SwiGLU FFN → residual.
                let mut pn = vec![0f32; d];
                rmsnorm(&h, &w_post.data, cfg.norm_eps, &mut pn);
                let mut gate = vec![0f32; d_ffn];
                let mut up = vec![0f32; d_ffn];
                matvec(w_gate, &pn, &mut gate)?;
                matvec(w_up, &pn, &mut up)?;
                for (g, &u) in gate.iter_mut().zip(up.iter()) {
                    *g = *g / (1.0 + (-*g).exp()) * u;
                }
                let mut down = vec![0f32; d];
                matvec(w_down, &gate, &mut down)?;
                for (h_i, &dl) in h.iter_mut().zip(down.iter()) {
                    *h_i += dl;
                }
            }

            let mut n = vec![0f32; d];
            rmsnorm(&h, &out_norm.data, cfg.norm_eps, &mut n);
            matvec(out_w, &n, &mut logits_out[ti * vocab..(ti + 1) * vocab])?;
        }
        Ok(())
    }
}

/// Attention layer: split Q/K/V, per-head QK RMSNorm, RoPE, causal GQA,
/// output proj.
#[allow(clippy::too_many_arguments)]
fn full_attn(
    w: &DenseWeights,
    b: &str,
    n: &[f32],
    pos: usize,
    state: &mut LayerState,
    cfg: &ModelConfig,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Vec<f32>, ModelError> {
    let d = cfg.d_model;
    let w_q = w.get(&format!("{b}.attn_q.weight"))?;
    let w_k = w.get(&format!("{b}.attn_k.weight"))?;
    let w_v = w.get(&format!("{b}.attn_v.weight"))?;
    let mut q = vec![0f32; n_heads * head_dim];
    let mut k = vec![0f32; n_kv_heads * head_dim];
    let mut v = vec![0f32; n_kv_heads * head_dim];
    matvec(w_q, n, &mut q).map_err(|_| ModelError::MissingWeight(format!("{b}.attn_q.weight")))?;
    matvec(w_k, n, &mut k).map_err(|_| ModelError::MissingWeight(format!("{b}.attn_k.weight")))?;
    matvec(w_v, n, &mut v).map_err(|_| ModelError::MissingWeight(format!("{b}.attn_v.weight")))?;

    let w_qn = w.get(&format!("{b}.attn_q_norm.weight"))?;
    let w_kn = w.get(&format!("{b}.attn_k_norm.weight"))?;
    let mut qn = vec![0f32; q.len()];
    let mut kn = vec![0f32; k.len()];
    for h in 0..n_heads {
        rmsnorm(
            &q[h * head_dim..(h + 1) * head_dim],
            &w_qn.data,
            cfg.norm_eps,
            &mut qn[h * head_dim..(h + 1) * head_dim],
        );
    }
    for h in 0..n_kv_heads {
        rmsnorm(
            &k[h * head_dim..(h + 1) * head_dim],
            &w_kn.data,
            cfg.norm_eps,
            &mut kn[h * head_dim..(h + 1) * head_dim],
        );
    }
    let rope = spite_rope::RopeConfig {
        head_dim,
        theta: cfg.rope_theta,
        ..Default::default()
    };
    spite_rope::apply_rope(&mut qn, pos as u32, &rope)
        .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
    spite_rope::apply_rope(&mut kn, pos as u32, &rope)
        .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;

    let LayerState { k: kk, v: vv } = state;
    kk.push(&kn);
    vv.push(&v);
    let n_prev = kk.len();
    // Dequantize the history on demand; same O(ctx) order as the attention.
    let kd = kk.to_f32();
    let vd = vv.to_f32();
    let attn_cfg = FlashAttnConfig::new(1, n_prev, n_heads, n_kv_heads, head_dim);
    let mut attn_out = vec![0f32; n_heads * head_dim];
    scalar_attention(&qn, &kd, &vd, &mut attn_out, &attn_cfg)
        .map_err(|e| ModelError::Forward(format!("attn: {e}")))?;
    let w_o = w.get(&format!("{b}.attn_output.weight"))?;
    let mut proj = vec![0f32; d];
    matvec(w_o, &attn_out, &mut proj)?;
    Ok(proj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny dense model: 4 layers, d=8, 2 heads, head_dim 4. All weights
    /// constant except norms (ones).
    fn tiny_weights() -> (ModelConfig, DenseWeights) {
        let d = 8usize;
        let hd = 4usize;
        let nh = 2usize;
        let ff = 16usize;
        let v = 8usize;

        let mut map: HashMap<String, (Vec<f32>, [u32; 4])> = HashMap::new();
        let w = |rows: usize, cols: usize, fill: f32| {
            (vec![fill; rows * cols], [cols as u32, rows as u32, 1, 1])
        };
        map.insert("token_embd.weight".into(), w(v, d, 0.1));
        map.insert(
            "output_norm.weight".into(),
            (vec![1.0; d], [d as u32, 1, 1, 1]),
        );
        map.insert("output.weight".into(), w(v, d, 0.1));
        for layer in 0..4 {
            let b = format!("blk.{layer}");
            map.insert(
                format!("{b}.attn_norm.weight"),
                (vec![1.0; d], [d as u32, 1, 1, 1]),
            );
            map.insert(
                format!("{b}.ffn_norm.weight"),
                (vec![1.0; d], [d as u32, 1, 1, 1]),
            );
            map.insert(format!("{b}.ffn_gate.weight"), w(ff, d, 0.05));
            map.insert(format!("{b}.ffn_up.weight"), w(ff, d, 0.05));
            map.insert(format!("{b}.ffn_down.weight"), w(d, ff, 0.05));
            map.insert(format!("{b}.attn_q.weight"), w(nh * hd, d, 0.05));
            map.insert(format!("{b}.attn_k.weight"), w(nh * hd, d, 0.05));
            map.insert(format!("{b}.attn_v.weight"), w(nh * hd, d, 0.05));
            map.insert(
                format!("{b}.attn_q_norm.weight"),
                (vec![1.0; hd], [hd as u32, 1, 1, 1]),
            );
            map.insert(
                format!("{b}.attn_k_norm.weight"),
                (vec![1.0; hd], [hd as u32, 1, 1, 1]),
            );
            map.insert(format!("{b}.attn_output.weight"), w(d, hd * nh, 0.05));
        }
        let cfg = ModelConfig {
            arch: "qwen3".into(),
            n_layers: 4,
            n_heads: nh,
            n_kv_heads: nh,
            d_model: d,
            d_ffn: ff,
            vocab_size: v,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            ..Default::default()
        };
        (cfg, DenseWeights::from_map(map))
    }

    #[test]
    fn forward_finite_and_incremental() {
        let (cfg, weights) = tiny_weights();
        let mut model = Qwen3_5::new(cfg.clone());
        model.weights = Some(weights);
        let ctx = SpiteCtx {
            n_ctx: 64,
            n_batch: 1,
            n_threads: 1,
            pos: 0,
            n_heads: 2,
            n_kv_heads: 2,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: std::ptr::null_mut(),
            scratchpad_bytes: 0,
        };
        let mut full = vec![0f32; 2 * cfg.vocab_size];
        model.forward(&[1, 2], &mut full, &ctx).unwrap();
        assert!(full.iter().all(|x| x.is_finite()));

        // Incremental decode matches the joint prefill.
        model.reset_cache();
        let mut pre = vec![0f32; cfg.vocab_size];
        let mut step = vec![0f32; cfg.vocab_size];
        model.forward(&[1], &mut pre, &ctx).unwrap();
        let mut ctx1 = ctx;
        ctx1.pos = 1;
        model.forward(&[2], &mut step, &ctx1).unwrap();
        assert!(full[..cfg.vocab_size] == pre[..]);
        assert!(full[cfg.vocab_size..] == step[..]);
    }

    #[test]
    fn vbr_degrades_kv_with_depth() {
        let (cfg, weights) = tiny_weights();
        let mut model = Qwen3_5::new(cfg.clone());
        model.weights = Some(weights);
        // Opt into VBR at f16. max_seq_len is 64, so thresholds land at
        // 16, 32 and 48 tokens.
        model.set_kv_quant(KvQuantConfig {
            key: KvQuant::F16,
            val: KvQuant::F16,
        });
        let ctx = SpiteCtx {
            n_ctx: 64,
            n_batch: 1,
            n_threads: 1,
            pos: 0,
            n_heads: 2,
            n_kv_heads: 2,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: std::ptr::null_mut(),
            scratchpad_bytes: 0,
        };
        let tokens: Vec<u32> = (0..50).map(|i| (i % cfg.vocab_size) as u32).collect();
        let mut logits = vec![0f32; tokens.len() * cfg.vocab_size];
        model.forward(&tokens, &mut logits, &ctx).unwrap();
        assert!(logits.iter().all(|x| x.is_finite()));
        // From f16 the 50 tokens cross all three thresholds (16, 32, 48) and
        // every layer lands on the q4 floor.
        assert_eq!(model.kv_key_tiers(), vec![KvQuant::Q4; 4]);

        // Starting at full precision means the same 50 tokens only reach
        // q5_1 — the f32 start adds one step before the ladder runs out.
        model.set_kv_quant(KvQuantConfig::full_precision());
        model.forward(&tokens, &mut logits, &ctx).unwrap();
        assert_eq!(model.kv_key_tiers(), vec![KvQuant::Q5_1; 4]);
    }
}
