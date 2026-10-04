//! Meta Llama 4 — GGUF arch `llama4`.
//!
//! Scout (16 experts) / Maverick (128 experts). Ported from llama.cpp
//! `models/llama4.cpp`: standard GQA with NoPE layers (no rope every
//! Nth layer), plain RMS QK-norm on roped layers (Scout only), chunked
//! SWA pattern, Q temperature scaling on NoPE layers, interleaved MoE
//! layers (top-k sigmoid router, weight-before-FFN, shared expert).

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_loader::GgufModel;
use spite_rope::{RopeConfig, apply_rope};

use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::moe::moe_swiglu;
use crate::{ModelArch, ModelConfig, ModelError};

pub struct Llama4 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    kv: RwLock<KvStore>,
}

struct KvStore {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Llama4 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv: RwLock::new(KvStore {
                k: Vec::new(),
                v: Vec::new(),
            }),
        }
    }

    fn is_moe_layer(&self, layer: usize) -> bool {
        let step = self.config.moe_layer_step;
        step > 0 && (layer + 1).is_multiple_of(step)
    }
}

impl ModelArch for Llama4 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        for name in model.tensor_names() {
            if name.contains("rope_freqs") {
                return Err(ModelError::Forward(
                    "per-layer rope_freqs tensors are not yet supported".into(),
                ));
            }
        }
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    fn reset_cache(&self) {
        if let Ok(mut kv) = self.kv.write() {
            kv.k.clear();
            kv.v.clear();
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
        // ponytail: O(ctx²) scalar CPU path; GPU kernels own speed.
        let swa = cfg.sliding_window.is_some();
        let no_rope_step = if swa { 4 } else { cfg.n_layers.max(1) };
        let use_kq_norm = cfg.n_expert != 128;
        let rope = RopeConfig {
            head_dim,
            theta: cfg.rope_theta,
            ..Default::default()
        };

        let embd = w.get("token_embd.weight")?;
        let out_norm = w.get("output_norm.weight")?;
        let out_w = w.get("output.weight")?;

        let mut kv = self
            .kv
            .write()
            .map_err(|_| ModelError::Forward("kv lock".into()))?;
        while kv.k.len() < cfg.n_layers {
            kv.k.push(Vec::new());
            kv.v.push(Vec::new());
        }

        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            let mut h = vec![0f32; d];
            h.copy_from_slice(&embd.data[tok as usize * d..(tok as usize + 1) * d]);

            for layer in 0..cfg.n_layers {
                let b = format!("blk.{layer}");
                let w_norm = w.get(&format!("{b}.attn_norm.weight"))?;
                let w_ffn_norm = w.get(&format!("{b}.ffn_norm.weight"))?;

                let mut n = vec![0f32; d];
                rmsnorm(&h, &w_norm.data, cfg.norm_eps, &mut n);

                // QKV: fused preferred, split fallback.
                let (mut q, mut k, v) = if let Ok(wqkv) = w.get(&format!("{b}.attn_qkv.weight")) {
                    let qd = n_heads * head_dim;
                    let kd = n_kv_heads * head_dim;
                    let mut qkv = vec![0f32; wqkv.ne[1] as usize];
                    matvec(wqkv, &n, &mut qkv)?;
                    (
                        qkv[..qd].to_vec(),
                        qkv[qd..qd + kd].to_vec(),
                        qkv[qd + kd..].to_vec(),
                    )
                } else {
                    let w_q = w.get(&format!("{b}.attn_q.weight"))?;
                    let w_k = w.get(&format!("{b}.attn_k.weight"))?;
                    let w_v = w.get(&format!("{b}.attn_v.weight"))?;
                    let (mut q, mut k, mut vv) = (
                        vec![0f32; n_heads * head_dim],
                        vec![0f32; n_kv_heads * head_dim],
                        vec![0f32; n_kv_heads * head_dim],
                    );
                    matvec(w_q, &n, &mut q)?;
                    matvec(w_k, &n, &mut k)?;
                    matvec(w_v, &n, &mut vv)?;
                    (q, k, vv)
                };

                let use_rope = (layer + 1) % no_rope_step != 0;
                if use_rope {
                    apply_rope(&mut q, pos as u32, &rope)
                        .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
                    apply_rope(&mut k, pos as u32, &rope)
                        .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
                    if use_kq_norm {
                        // Plain RMS QK-norm, no weights (Scout only).
                        let ones = vec![1f32; head_dim];
                        for (buf, n_h) in [(&mut q, n_heads), (&mut k, n_kv_heads)] {
                            for h in 0..n_h {
                                let mut nn = vec![0f32; head_dim];
                                rmsnorm(
                                    &buf[h * head_dim..(h + 1) * head_dim],
                                    &ones,
                                    cfg.norm_eps,
                                    &mut nn,
                                );
                                buf[h * head_dim..(h + 1) * head_dim].copy_from_slice(&nn);
                            }
                        }
                    }
                } else {
                    // Temperature scaling on NoPE layers (hardcoded in llama4).
                    let scale = (((pos + 1) / 8192) as f32 + 1.0).ln() * 0.1 + 1.0;
                    for x in q.iter_mut() {
                        *x *= scale;
                    }
                }

                kv.k[layer].extend_from_slice(&k);
                kv.v[layer].extend_from_slice(&v);
                let row_len = n_kv_heads * head_dim;
                let n_prev = kv.k[layer].len() / row_len;
                let (kk, vv, n_kv) = match cfg.sliding_window {
                    Some(win)
                        if cfg.swa_layers.get(layer).copied().unwrap_or(false) && n_prev > win =>
                    {
                        let skip = (n_prev - win) * row_len;
                        (&kv.k[layer][skip..], &kv.v[layer][skip..], win)
                    }
                    _ => (kv.k[layer].as_slice(), kv.v[layer].as_slice(), n_prev),
                };
                let attn_cfg = FlashAttnConfig::new(1, n_kv, n_heads, n_kv_heads, head_dim);
                let mut attn_out = vec![0f32; n_heads * head_dim];
                scalar_attention(&q, kk, vv, &mut attn_out, &attn_cfg)
                    .map_err(|e| ModelError::Forward(format!("attn: {e}")))?;
                let w_o = w.get(&format!("{b}.attn_output.weight"))?;
                let mut proj = vec![0f32; d];
                matvec(w_o, &attn_out, &mut proj)?;
                for (h_i, &p) in h.iter_mut().zip(proj.iter()) {
                    *h_i += p;
                }

                // FFN: MoE or dense.
                let mut fn_ = vec![0f32; d];
                rmsnorm(&h, &w_ffn_norm.data, cfg.norm_eps, &mut fn_);
                let ffn_out = if self.is_moe_layer(layer) {
                    let w_inp = w.get(&format!("{b}.ffn_gate_inp.weight"))?;
                    let w_up = w.get(&format!("{b}.ffn_up_exps.weight"))?;
                    let w_gate = w.get(&format!("{b}.ffn_gate_exps.weight"))?;
                    let w_down = w.get(&format!("{b}.ffn_down_exps.weight"))?;
                    let n_exp = cfg.n_expert;
                    let ff_exp = w_up.ne[1] as usize;
                    let mut out = moe_swiglu(
                        &fn_,
                        w_inp,
                        w_up,
                        w_gate,
                        w_down,
                        n_exp,
                        cfg.n_expert_used,
                        ff_exp,
                        cfg.expert_weights_scale,
                        cfg.swiglu_clamp_exp.get(layer).copied().unwrap_or(0.0),
                    )?;
                    // Shared expert.
                    let w_su = w.get(&format!("{b}.ffn_up_shexp.weight"))?;
                    let w_sg = w.get(&format!("{b}.ffn_gate_shexp.weight"))?;
                    let w_sd = w.get(&format!("{b}.ffn_down_shexp.weight"))?;
                    let ff_se = w_su.ne[1] as usize;
                    let mut up = vec![0f32; ff_se];
                    let mut gate = vec![0f32; ff_se];
                    matvec(w_su, &fn_, &mut up)?;
                    matvec(w_sg, &fn_, &mut gate)?;
                    let lim = cfg.swiglu_clamp_shexp.get(layer).copied().unwrap_or(0.0);
                    for (u, &g) in up.iter_mut().zip(gate.iter()) {
                        if lim > 1e-6 {
                            *u = u.clamp(-lim, lim);
                        }
                        let mut act = g / (1.0 + (-g).exp());
                        if lim > 1e-6 {
                            act = act.min(lim);
                        }
                        *u *= act;
                    }
                    let mut shared = vec![0f32; d];
                    matvec(w_sd, &up, &mut shared)?;
                    for (o, &s) in out.iter_mut().zip(shared.iter()) {
                        *o += s;
                    }
                    out
                } else {
                    let w_gate = w.get(&format!("{b}.ffn_gate.weight"))?;
                    let w_up = w.get(&format!("{b}.ffn_up.weight"))?;
                    let w_down = w.get(&format!("{b}.ffn_down.weight"))?;
                    let mut gate = vec![0f32; d_ffn];
                    let mut up = vec![0f32; d_ffn];
                    matvec(w_gate, &fn_, &mut gate)?;
                    matvec(w_up, &fn_, &mut up)?;
                    for (g, &u) in gate.iter_mut().zip(up.iter()) {
                        *g = *g / (1.0 + (-*g).exp()) * u;
                    }
                    let mut down = vec![0f32; d];
                    matvec(w_down, &gate, &mut down)?;
                    down
                };
                for (h_i, &f) in h.iter_mut().zip(ffn_out.iter()) {
                    *h_i += f;
                }
            }

            let mut n = vec![0f32; d];
            rmsnorm(&h, &out_norm.data, cfg.norm_eps, &mut n);
            matvec(out_w, &n, &mut logits_out[ti * vocab..(ti + 1) * vocab])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny 2-layer model: layer 0 dense, layer 1 MoE (step 2), 4 experts top-1.
    fn tiny_weights() -> (ModelConfig, DenseWeights) {
        let d = 8usize;
        let hd = 4usize;
        let nh = 2usize;
        let ff = 16usize;
        let ffe = 8usize;
        let v = 8usize;
        let ne = 4usize;

        let mut map: HashMap<String, (Vec<f32>, [u32; 4])> = HashMap::new();
        let w = |rows: usize, cols: usize, fill: f32| {
            (vec![fill; rows * cols], [cols as u32, rows as u32, 1, 1])
        };
        let ones = |n: usize| (vec![1.0; n], [n as u32, 1, 1, 1]);
        map.insert("token_embd.weight".into(), w(v, d, 0.1));
        map.insert("output_norm.weight".into(), ones(d));
        map.insert("output.weight".into(), w(v, d, 0.1));
        for layer in 0..2 {
            let b = format!("blk.{layer}");
            map.insert(format!("{b}.attn_norm.weight"), ones(d));
            map.insert(format!("{b}.ffn_norm.weight"), ones(d));
            map.insert(format!("{b}.attn_q.weight"), w(nh * hd, d, 0.05));
            map.insert(format!("{b}.attn_k.weight"), w(nh * hd, d, 0.05));
            map.insert(format!("{b}.attn_v.weight"), w(nh * hd, d, 0.05));
            map.insert(format!("{b}.attn_output.weight"), w(d, nh * hd, 0.05));
            if layer == 1 {
                map.insert(format!("{b}.ffn_gate_inp.weight"), w(ne, d, 0.5));
                let stack = |fill: f32| {
                    (
                        vec![fill; ne * d * ffe],
                        [d as u32, ffe as u32, ne as u32, 1],
                    )
                };
                map.insert(format!("{b}.ffn_up_exps.weight"), stack(0.05));
                map.insert(format!("{b}.ffn_gate_exps.weight"), stack(0.05));
                map.insert(
                    format!("{b}.ffn_down_exps.weight"),
                    (
                        vec![0.05; ne * ffe * d],
                        [ffe as u32, d as u32, ne as u32, 1],
                    ),
                );
                map.insert(format!("{b}.ffn_up_shexp.weight"), w(ffe, d, 0.05));
                map.insert(format!("{b}.ffn_gate_shexp.weight"), w(ffe, d, 0.05));
                map.insert(format!("{b}.ffn_down_shexp.weight"), w(d, ffe, 0.05));
            } else {
                map.insert(format!("{b}.ffn_gate.weight"), w(ff, d, 0.05));
                map.insert(format!("{b}.ffn_up.weight"), w(ff, d, 0.05));
                map.insert(format!("{b}.ffn_down.weight"), w(d, ff, 0.05));
            }
        }
        let cfg = ModelConfig {
            arch: "llama4".into(),
            n_layers: 2,
            n_heads: nh,
            n_kv_heads: nh,
            d_model: d,
            d_ffn: ff,
            vocab_size: v,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            n_expert: ne,
            n_expert_used: 1,
            moe_layer_step: 2,
            ..Default::default()
        };
        (cfg, DenseWeights::from_map(map))
    }

    #[test]
    fn moe_forward_finite_and_incremental() {
        let (cfg, weights) = tiny_weights();
        let mut model = Llama4::new(cfg.clone());
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
