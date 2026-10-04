//! Alibaba Qwen 3.5 — GGUF arch `qwen35`.
//!
//! Hybrid decoder (ported from llama.cpp `models/qwen35.cpp`): every Nth
//! layer (default interval 4) is full GQA attention with fused Q+gate
//! projection, per-head QK RMSNorm, and interleaved RoPE; the rest are
//! gated-delta-net linear attention layers (fused QKV, short conv, GDN
//! recurrence, gated output norm). Both share one post-attention norm +
//! SwiGLU FFN tail.
//!
//! MTP draft heads are not wired; the trunk runs standalone.

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_compute::linear_attn::{gdn_l2_norm, gdn_step, ssm_conv_step};
use spite_loader::GgufModel;
use spite_rope::apply_irope;

use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::{ModelArch, ModelConfig, ModelError};

/// Per-layer recurrent state for one linear layer.
struct LinearState {
    /// GDN state: Hv S×S matrices, row-major.
    gdn: Vec<f32>,
    /// Conv history: previous K−1 fused-qkv vectors, oldest first.
    conv_hist: Vec<Vec<f32>>,
}

enum LayerState {
    Full { k: Vec<f32>, v: Vec<f32> },
    Linear(LinearState),
}

pub struct Qwen3_5 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    state: RwLock<Vec<LayerState>>,
}

impl Qwen3_5 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            state: RwLock::new(Vec::new()),
        }
    }

    fn is_recurrent(&self, layer: usize) -> bool {
        self.config
            .recurrent_layers
            .get(layer)
            .copied()
            .unwrap_or(true)
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
        let head_dim = d / n_heads;
        let d_ffn = cfg.d_ffn;
        let vocab = cfg.vocab_size;
        if logits_out.len() != tokens.len() * vocab {
            return Err(ModelError::Forward("logits_out shape mismatch".into()));
        }
        // ponytail: O(ctx²)/O(ctx·S²) scalar CPU path; GPU kernels own speed.
        let sect = cfg.rope_sections;
        let has_sections = sect.iter().any(|&s| s > 0);

        let embd = w.get("token_embd.weight")?;
        let out_norm = w.get("output_norm.weight")?;
        let out_w = w.get("output.weight")?;

        let mut state = self
            .state
            .write()
            .map_err(|_| ModelError::Forward("state lock".into()))?;
        while state.len() < cfg.n_layers {
            let layer = state.len();
            if self.is_recurrent(layer) {
                state.push(LayerState::Linear(LinearState {
                    gdn: Vec::new(),
                    conv_hist: Vec::new(),
                }));
            } else {
                state.push(LayerState::Full {
                    k: Vec::new(),
                    v: Vec::new(),
                });
            }
        }

        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            let mut h = vec![0f32; d];
            h.copy_from_slice(&embd.data[tok as usize * d..(tok as usize + 1) * d]);

            for layer in 0..cfg.n_layers {
                let b = format!("blk.{layer}");
                let w_norm = w.get(&format!("{b}.attn_norm.weight"))?;
                let w_post = w.get(&format!("{b}.attn_post_norm.weight"))?;
                let w_gate = w.get(&format!("{b}.ffn_gate.weight"))?;
                let w_up = w.get(&format!("{b}.ffn_up.weight"))?;
                let w_down = w.get(&format!("{b}.ffn_down.weight"))?;

                let mut n = vec![0f32; d];
                rmsnorm(&h, &w_norm.data, cfg.norm_eps, &mut n);

                let attn_out = if !self.is_recurrent(layer) {
                    full_attn(
                        w,
                        &b,
                        &n,
                        pos,
                        layer,
                        cfg,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                        sect,
                        has_sections,
                        &mut state,
                    )?
                } else {
                    linear_attn(w, &b, &n, cfg, &mut state, layer)?
                };

                for (h_i, &p) in h.iter_mut().zip(attn_out.iter()) {
                    *h_i += p;
                }
                // Shared tail: post-attention norm → SwiGLU FFN → residual.
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

/// Full-attention layer: fused Q+gate (or split Q/K/V), per-head QK
/// RMSNorm, iRoPE, causal GQA, sigmoid gate, output proj.
#[allow(clippy::too_many_arguments)]
fn full_attn(
    w: &DenseWeights,
    b: &str,
    n: &[f32],
    pos: usize,
    layer: usize,
    cfg: &ModelConfig,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    sect: [u32; 4],
    has_sections: bool,
    state: &mut [LayerState],
) -> Result<Vec<f32>, ModelError> {
    let d = cfg.d_model;
    // Fused Q+gate+K+V preferred (qwen35 full layers); split fallback.
    let (q, gate, k, v) = if let Ok(wqkv) = w.get(&format!("{b}.attn_qkv.weight")) {
        let qg_dim = 2 * head_dim * n_heads;
        let k_dim = n_kv_heads * head_dim;
        let mut qkv = vec![0f32; wqkv.rows()];
        matvec(wqkv, n, &mut qkv)?;
        let gate = qkv[qg_dim / 2..qg_dim].to_vec();
        (
            qkv[..qg_dim / 2].to_vec(),
            Some(gate),
            qkv[qg_dim..qg_dim + k_dim].to_vec(),
            qkv[qg_dim + k_dim..].to_vec(),
        )
    } else {
        let w_q = w.get(&format!("{b}.attn_q.weight"))?;
        let w_k = w.get(&format!("{b}.attn_k.weight"))?;
        let w_v = w.get(&format!("{b}.attn_v.weight"))?;
        let mut q = vec![0f32; n_heads * head_dim];
        let mut k = vec![0f32; n_kv_heads * head_dim];
        let mut v = vec![0f32; n_kv_heads * head_dim];
        matvec(w_q, n, &mut q)?;
        matvec(w_k, n, &mut k)?;
        matvec(w_v, n, &mut v)?;
        (q, None, k, v)
    };

    // Per-head QK RMSNorm.
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
    if has_sections {
        let p = [pos as u32; 4];
        apply_irope(&mut qn, p, sect, head_dim, head_dim, cfg.rope_theta);
        apply_irope(&mut kn, p, sect, head_dim, head_dim, cfg.rope_theta);
    } else {
        let rope = spite_rope::RopeConfig {
            head_dim,
            theta: cfg.rope_theta,
            ..Default::default()
        };
        spite_rope::apply_rope(&mut qn, pos as u32, &rope)
            .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
        spite_rope::apply_rope(&mut kn, pos as u32, &rope)
            .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
    }

    let LayerState::Full { k: kk, v: vv } = &mut state[layer] else {
        return Err(ModelError::Forward("layer state kind mismatch".into()));
    };
    kk.extend_from_slice(&kn);
    vv.extend_from_slice(&v);
    let n_prev = kk.len() / (n_kv_heads * head_dim);
    let attn_cfg = FlashAttnConfig::new(1, n_prev, n_heads, n_kv_heads, head_dim);
    let mut attn_out = vec![0f32; n_heads * head_dim];
    scalar_attention(&qn, kk, vv, &mut attn_out, &attn_cfg)
        .map_err(|e| ModelError::Forward(format!("attn: {e}")))?;
    if let Some(g) = gate {
        for (o, &gg) in attn_out.iter_mut().zip(g.iter()) {
            *o *= 1.0 / (1.0 + (-gg).exp());
        }
    }
    let w_o = w.get(&format!("{b}.attn_output.weight"))?;
    let mut proj = vec![0f32; d];
    matvec(w_o, &attn_out, &mut proj)?;
    Ok(proj)
}

/// Linear GDN layer: fused QKV + gate, beta/alpha, short conv, GDN
/// recurrence with persistent state, gated output norm, out proj.
fn linear_attn(
    w: &DenseWeights,
    b: &str,
    n: &[f32],
    cfg: &ModelConfig,
    state: &mut [LayerState],
    layer: usize,
) -> Result<Vec<f32>, ModelError> {
    let d = cfg.d_model;
    let s = cfg.ssm_d_state.max(1);
    let n_kh = cfg.ssm_n_group.max(1);
    let n_vh = cfg.ssm_dt_rank.max(1);
    let d_inner = cfg.ssm_d_inner.max(1);
    let head_v = d_inner / n_vh;
    if head_v != s {
        return Err(ModelError::Forward("gdn head dims must match".into()));
    }
    let key_dim = s * n_kh;
    let value_dim = head_v * n_vh;

    let w_qkv = w.get(&format!("{b}.attn_qkv.weight"))?;
    let w_gate = w.get(&format!("{b}.attn_gate.weight"))?;
    let mut qkv = vec![0f32; 2 * key_dim + value_dim];
    let mut z = vec![0f32; value_dim];
    matvec(w_qkv, n, &mut qkv)?;
    matvec(w_gate, n, &mut z)?;

    let w_beta = w.get(&format!("{b}.ssm_beta.weight"))?;
    let w_alpha = w.get(&format!("{b}.ssm_alpha.weight"))?;
    let w_dt = w
        .get(&format!("{b}.ssm_dt"))
        .or_else(|_| w.get(&format!("{b}.ssm_dt.bias")))
        .or_else(|_| w.get(&format!("{b}.ssm_dt.weight")))?;
    let w_a = w
        .get(&format!("{b}.ssm_a"))
        .or_else(|_| w.get(&format!("{b}.ssm_a.weight")))?;
    let mut beta_raw = vec![0f32; n_vh];
    let mut alpha_raw = vec![0f32; n_vh];
    matvec(w_beta, n, &mut beta_raw)?;
    matvec(w_alpha, n, &mut alpha_raw)?;

    // Short conv over fused qkv with cached history.
    let w_conv = w.get(&format!("{b}.ssm_conv1d.weight"))?;
    let k_size = cfg.ssm_d_conv.max(1);
    let LayerState::Linear(ls) = &mut state[layer] else {
        // First visit: convert the placeholder Full state.
        return Err(ModelError::Forward("layer state kind mismatch".into()));
    };
    while ls.conv_hist.len() + 1 < k_size {
        ls.conv_hist.insert(0, vec![0f32; qkv.len()]);
    }
    let conv = ssm_conv_step(&ls.conv_hist, &qkv, &w_conv.data);
    ls.conv_hist.push(qkv.clone());
    while ls.conv_hist.len() + 1 > k_size {
        ls.conv_hist.remove(0);
    }
    let conv_silu: Vec<f32> = conv.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    let (q_raw, rest) = conv_silu.split_at(key_dim);
    let (k_raw, v_raw) = rest.split_at(key_dim);

    // L2-normalize q/k per head; repeat k-heads to value-head count.
    let mut qn = vec![0f32; key_dim];
    let mut kn = vec![0f32; key_dim];
    for h in 0..n_kh {
        gdn_l2_norm(
            &q_raw[h * s..(h + 1) * s],
            cfg.norm_eps,
            &mut qn[h * s..(h + 1) * s],
        );
        gdn_l2_norm(
            &k_raw[h * s..(h + 1) * s],
            cfg.norm_eps,
            &mut kn[h * s..(h + 1) * s],
        );
    }

    if ls.gdn.len() < n_vh * s * s {
        ls.gdn.resize(n_vh * s * s, 0.0);
    }
    let scale = 1.0 / (s as f32).sqrt();
    let mut gdn_out = vec![0f32; value_dim];
    for vh in 0..n_vh {
        let kh = vh % n_kh;
        let qs: Vec<f32> = qn[kh * s..(kh + 1) * s]
            .iter()
            .map(|&x| x * scale)
            .collect();
        let beta = 1.0 / (1.0 + (-beta_raw[vh]).exp());
        // gate = exp(softplus(alpha + dt) * a); gdn_step applies exp itself.
        let gate = (alpha_raw[vh] + w_dt.data[vh]).ln_1p().exp() * w_a.data[vh];
        let st = &mut ls.gdn[vh * s * s..(vh + 1) * s * s];
        let mut o = vec![0f32; s];
        gdn_step(
            st,
            &qs,
            &kn[kh * s..(kh + 1) * s],
            &v_raw[vh * head_v..(vh + 1) * head_v],
            gate,
            beta,
            &mut o,
        )
        .map_err(|e| ModelError::Forward(format!("gdn: {e}")))?;
        gdn_out[vh * head_v..(vh + 1) * head_v].copy_from_slice(&o);
    }

    // Gated norm: rmsnorm(out) * silu(z), then out proj.
    let w_norm = w.get(&format!("{b}.ssm_norm.weight"))?;
    let mut normed = vec![0f32; value_dim];
    crate::dense::rmsnorm(&gdn_out, &w_norm.data, cfg.norm_eps, &mut normed);
    for (y, &zz) in normed.iter_mut().zip(z.iter()) {
        *y *= zz / (1.0 + (-zz).exp());
    }
    let w_out = w.get(&format!("{b}.ssm_out.weight"))?;
    let mut proj = vec![0f32; d];
    matvec(w_out, &normed, &mut proj)?;
    Ok(proj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny hybrid model: 4 layers (interval 4 → layer 3 is full attention),
    /// d=8, 2 heads, S=4 GDN state. All weights zero except norms (ones).
    fn tiny_weights() -> (ModelConfig, DenseWeights) {
        let d = 8usize;
        let hd = 4usize;
        let nh = 2usize;
        let ff = 16usize;
        let v = 8usize;
        let s = 4usize;
        let nkh = 1usize;
        let nvh = 2usize;
        let hv = 4usize;
        let key_dim = s * nkh;
        let value_dim = hv * nvh;
        let conv_c = 2 * key_dim + value_dim;
        let k_size = 2usize;

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
                format!("{b}.attn_post_norm.weight"),
                (vec![1.0; d], [d as u32, 1, 1, 1]),
            );
            map.insert(format!("{b}.ffn_gate.weight"), w(ff, d, 0.05));
            map.insert(format!("{b}.ffn_up.weight"), w(ff, d, 0.05));
            map.insert(format!("{b}.ffn_down.weight"), w(d, ff, 0.05));
            if layer == 3 {
                // Full layer: fused Q+gate, split K/V, QK norms.
                map.insert(
                    format!("{b}.attn_qkv.weight"),
                    w(2 * hd * nh + 2 * nh * hd, d, 0.05),
                );
                map.insert(
                    format!("{b}.attn_q_norm.weight"),
                    (vec![1.0; hd], [hd as u32, 1, 1, 1]),
                );
                map.insert(
                    format!("{b}.attn_k_norm.weight"),
                    (vec![1.0; hd], [hd as u32, 1, 1, 1]),
                );
                map.insert(format!("{b}.attn_output.weight"), w(d, hd * nh, 0.05));
            } else {
                // Linear layer: fused QKV + gate, SSM params.
                map.insert(
                    format!("{b}.attn_qkv.weight"),
                    w(2 * key_dim + value_dim, d, 0.05),
                );
                map.insert(format!("{b}.attn_gate.weight"), w(value_dim, d, 0.05));
                map.insert(format!("{b}.ssm_beta.weight"), w(nvh, d, 0.05));
                map.insert(format!("{b}.ssm_alpha.weight"), w(nvh, d, 0.05));
                map.insert(
                    format!("{b}.ssm_dt"),
                    (vec![0.1; nvh], [nvh as u32, 1, 1, 1]),
                );
                map.insert(
                    format!("{b}.ssm_a"),
                    (vec![0.5; nvh], [nvh as u32, 1, 1, 1]),
                );
                map.insert(
                    format!("{b}.ssm_conv1d.weight"),
                    (
                        vec![0.25; k_size * conv_c],
                        [conv_c as u32, k_size as u32, 1, 1],
                    ),
                );
                map.insert(
                    format!("{b}.ssm_norm.weight"),
                    (vec![1.0; hv], [hv as u32, 1, 1, 1]),
                );
                map.insert(format!("{b}.ssm_out.weight"), w(d, value_dim, 0.05));
            }
        }
        let cfg = ModelConfig {
            arch: "qwen35".into(),
            n_layers: 4,
            n_heads: nh,
            n_kv_heads: nh,
            d_model: d,
            d_ffn: ff,
            vocab_size: v,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            ssm_d_conv: k_size,
            ssm_d_inner: value_dim,
            ssm_d_state: s,
            ssm_dt_rank: nvh,
            ssm_n_group: nkh,
            ..Default::default()
        };
        (cfg, DenseWeights::from_map(map))
    }

    #[test]
    fn hybrid_forward_finite_and_incremental() {
        let (cfg, weights) = tiny_weights();
        let mut model = Qwen3_5::new(cfg.clone());
        model.weights = Some(weights);
        // recurrent_layers derived from interval: layers 0-2 linear, 3 full.
        model.config.recurrent_layers = vec![true, true, true, false];
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
}
