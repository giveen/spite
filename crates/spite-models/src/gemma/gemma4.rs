//! Google Gemma 4 — GGUF arch `gemma4`.
//!
//! Variant: Gemma 4 (2025/2026).
//!
//! Architectural lineage and features:
//! - 5:1 alternating attention: 5 Local SWA layers (1024 window, head_dim 256, GQA)
//!   followed by 1 Global Full Attention layer (head_dim 512, MQA, V=K).
//! - Input embedding scaled by sqrt(d_model).
//! - Direct multiplicative RMSNorm (no 1+w offset).
//! - Per-head RMSNorm on Q and K; unscaled RMSNorm on V.
//! - Dual RoPE: theta=10,000 on SWA layers; proportional rope_freqs table on Global layers.
//! - Post-attention RMSNorm and post-FFW RMSNorm before residual additions.
//! - GeGLU FFN with tanh-approximated GELU.
//! - Per-layer scalar output scaling (layer_output_scale).
//! - Tied LM head (token_embd.weight) when output.weight is omitted.
//! - Logit soft-capping: logits = cap * tanh(logits / cap).

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_kvcache::{KvQuant, KvQuantConfig, VbrPolicy, VbrRows};
use spite_loader::GgufModel;
use spite_rope::{RopeConfig, apply_rope};

use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::{ModelArch, ModelConfig, ModelError};

/// Unscaled RMS norm over `x`: out[i] = x[i] / sqrt(mean(x²) + eps).
/// Gemma 4 applies this to V heads after projection.
fn rmsnorm_unscaled(x: &[f32], eps: f32, out: &mut [f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v * scale;
    }
}

/// tanh-approximated GELU matching llama.cpp ggml.
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

/// Proportional RoPE for global layers using the `rope_freqs` table.
fn apply_proportional_rope(
    qk: &mut [f32],
    pos: u32,
    head_dim: usize,
    theta: f32,
    rope_freqs: &[f32],
) {
    let half = head_dim / 2;
    for head in qk.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let ff = rope_freqs.get(i).copied().unwrap_or(1.0);
            let effective_theta = if ff != 0.0 { theta / ff } else { theta };
            let freq = 1.0 / effective_theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
    }
}

/// Per-layer K/V cache accommodating Gemma 4's alternating row lengths
/// (2048 floats for SWA layers vs 512 floats for Global layers).
struct Gemma4KvStore {
    k: Vec<VbrRows>,
    v: Vec<VbrRows>,
    key_quant: KvQuant,
    val_quant: KvQuant,
}

impl Default for Gemma4KvStore {
    fn default() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            key_quant: KvQuant::F16,
            val_quant: KvQuant::F16,
        }
    }
}

impl Gemma4KvStore {
    fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
    }

    fn set_quant(&mut self, cfg: &KvQuantConfig) {
        self.key_quant = cfg.key;
        self.val_quant = cfg.val;
        self.reset();
    }
}

pub struct Gemma4 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    kv: RwLock<Gemma4KvStore>,
}

impl Gemma4 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv: RwLock::new(Gemma4KvStore::default()),
        }
    }
}

impl ModelArch for Gemma4 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    fn reset_cache(&self) {
        if let Ok(mut kv) = self.kv.write() {
            kv.reset();
        }
    }

    fn set_kv_quant(&self, cfg: KvQuantConfig) {
        if let Ok(mut kv) = self.kv.write() {
            kv.set_quant(&cfg);
        }
    }

    fn forward(
        &self,
        tokens: &[u32],
        logits_out: &mut [f32],
        ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let Some(weights) = &self.weights else {
            return Err(ModelError::Forward("load_weights not called".into()));
        };

        let cfg = &self.config;
        let d = cfg.d_model;
        let d_ffn = cfg.d_ffn;
        let vocab = cfg.vocab_size;
        let eps = cfg.norm_eps;
        let n_layers = cfg.n_layers;

        if logits_out.len() != tokens.len() * vocab {
            return Err(ModelError::Forward("logits_out shape mismatch".into()));
        }

        let embd = weights.get("token_embd.weight")?;
        let out_norm = weights.get("output_norm.weight")?;
        // Tied LM head fallback: Gemma models reuse token_embd when output.weight is absent
        let out_w = weights
            .get("output.weight")
            .or_else(|_| weights.get("token_embd.weight"))?;

        let rope_freqs = weights
            .get("rope_freqs.weight")
            .map(|w| &w.data[..])
            .unwrap_or(&[]);

        let tok_scale = (d as f32).sqrt();
        let mut kv = self
            .kv
            .write()
            .map_err(|_| ModelError::Forward("kv lock error".into()))?;

        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            let mut h = vec![0f32; d];
            let row = &embd.data[tok as usize * d..(tok as usize + 1) * d];
            h.copy_from_slice(row);
            // Gemma embedding scaling
            for v in h.iter_mut() {
                *v *= tok_scale;
            }

            for layer in 0..n_layers {
                let is_swa = if !cfg.swa_layers.is_empty() {
                    cfg.swa_layers.get(layer).copied().unwrap_or(true)
                } else {
                    // Default Gemma 4 5:1 pattern: every 6th layer (5, 11, ...) is global
                    layer % 6 != 5
                };

                let n_heads = cfg.n_heads.max(1);
                let head_dim = if is_swa {
                    if cfg.key_length_swa > 0 {
                        cfg.key_length_swa
                    } else {
                        (d / n_heads).max(1)
                    }
                } else {
                    if cfg.key_length > 0 {
                        cfg.key_length
                    } else {
                        (d / n_heads).max(1)
                    }
                };

                let n_kv_heads = if is_swa {
                    if !cfg.head_count_kv_arr.is_empty() {
                        cfg.head_count_kv_arr
                            .get(layer)
                            .copied()
                            .unwrap_or(cfg.n_kv_heads)
                    } else {
                        cfg.n_kv_heads.max(1)
                    }
                } else {
                    1
                };

                let b = format!("blk.{layer}");
                let w_norm = weights.get(&format!("{b}.attn_norm.weight"))?;
                let w_q = weights.get(&format!("{b}.attn_q.weight"))?;
                let w_k = weights.get(&format!("{b}.attn_k.weight"))?;
                let w_o = weights.get(&format!("{b}.attn_output.weight"))?;

                let w_qn = weights.get(&format!("{b}.attn_q_norm.weight")).ok();
                let w_kn = weights.get(&format!("{b}.attn_k_norm.weight")).ok();
                let w_post_attn_norm = weights.get(&format!("{b}.post_attention_norm.weight")).ok();

                let w_ffn_norm = weights.get(&format!("{b}.ffn_norm.weight"))?;
                let w_gate = weights.get(&format!("{b}.ffn_gate.weight"))?;
                let w_up = weights.get(&format!("{b}.ffn_up.weight"))?;
                let w_down = weights.get(&format!("{b}.ffn_down.weight"))?;
                let w_post_ffw_norm = weights.get(&format!("{b}.post_ffw_norm.weight")).ok();
                let w_out_scale = weights.get(&format!("{b}.layer_output_scale.weight")).ok();

                // 1. Pre-attention RMSNorm
                let mut n_attn = vec![0f32; d];
                rmsnorm(&h, &w_norm.data, eps, &mut n_attn);

                // 2. Projections
                let mut q = vec![0f32; n_heads * head_dim];
                let mut k = vec![0f32; n_kv_heads * head_dim];
                let mut v = vec![0f32; n_kv_heads * head_dim];

                matvec(w_q, &n_attn, &mut q)?;
                matvec(w_k, &n_attn, &mut k)?;

                if is_swa {
                    let w_v = weights.get(&format!("{b}.attn_v.weight"))?;
                    matvec(w_v, &n_attn, &mut v)?;
                } else {
                    // Global layers in Gemma 4 omit attn_v.weight: V = K
                    if let Ok(w_v) = weights.get(&format!("{b}.attn_v.weight")) {
                        matvec(w_v, &n_attn, &mut v)?;
                    } else {
                        v.copy_from_slice(&k);
                    }
                }

                // 3. Per-head RMSNorm on Q, K, and unscaled on V
                if let Some(w_qn) = w_qn {
                    for h_idx in 0..n_heads {
                        let head = &mut q[h_idx * head_dim..(h_idx + 1) * head_dim];
                        let mut out = vec![0f32; head_dim];
                        rmsnorm(head, &w_qn.data, eps, &mut out);
                        head.copy_from_slice(&out);
                    }
                }
                if let Some(w_kn) = w_kn {
                    for h_idx in 0..n_kv_heads {
                        let head = &mut k[h_idx * head_dim..(h_idx + 1) * head_dim];
                        let mut out = vec![0f32; head_dim];
                        rmsnorm(head, &w_kn.data, eps, &mut out);
                        head.copy_from_slice(&out);
                    }
                }
                for h_idx in 0..n_kv_heads {
                    let head = &mut v[h_idx * head_dim..(h_idx + 1) * head_dim];
                    let mut out = vec![0f32; head_dim];
                    rmsnorm_unscaled(head, eps, &mut out);
                    head.copy_from_slice(&out);
                }

                // 4. RoPE
                if is_swa {
                    let rope = RopeConfig {
                        head_dim,
                        theta: cfg.rope_freq_base_swa,
                        ..Default::default()
                    };
                    apply_rope(&mut q, pos as u32, &rope)
                        .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
                    apply_rope(&mut k, pos as u32, &rope)
                        .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
                } else {
                    apply_proportional_rope(
                        &mut q,
                        pos as u32,
                        head_dim,
                        cfg.rope_theta,
                        rope_freqs,
                    );
                    apply_proportional_rope(
                        &mut k,
                        pos as u32,
                        head_dim,
                        cfg.rope_theta,
                        rope_freqs,
                    );
                }

                // 5. KV Cache & Attention
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

                let n_prev = kv.k[layer].len();
                let n_attend = if is_swa {
                    n_prev.min(cfg.sliding_window.unwrap_or(1024))
                } else {
                    n_prev
                };
                let from = n_prev - n_attend;
                let kk = kv.k[layer].to_f32_from(from);
                let vv = kv.v[layer].to_f32_from(from);

                // Gemma 4 uses 1.0 attention scale since Q and K are pre-normalized
                let attn_cfg = FlashAttnConfig {
                    n_q: 1,
                    n_kv: n_attend,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scale: 1.0,
                    causal: true,
                    sinks: None,
                };
                let mut attn_out = vec![0f32; n_heads * head_dim];
                scalar_attention(&q, &kk, &vv, &mut attn_out, &attn_cfg)
                    .map_err(|e| ModelError::Forward(format!("attn: {e}")))?;

                let mut proj = vec![0f32; d];
                matvec(w_o, &attn_out, &mut proj)?;

                // 6. Post-attention RMSNorm & Residual add
                if let Some(w_post_attn) = w_post_attn_norm {
                    let mut norm_proj = vec![0f32; d];
                    rmsnorm(&proj, &w_post_attn.data, eps, &mut norm_proj);
                    for (h_i, &p) in h.iter_mut().zip(norm_proj.iter()) {
                        *h_i += p;
                    }
                } else {
                    for (h_i, &p) in h.iter_mut().zip(proj.iter()) {
                        *h_i += p;
                    }
                }

                // 7. FFN block (GeGLU)
                let mut n_ffn = vec![0f32; d];
                rmsnorm(&h, &w_ffn_norm.data, eps, &mut n_ffn);
                let mut gate = vec![0f32; d_ffn];
                let mut up = vec![0f32; d_ffn];
                matvec(w_gate, &n_ffn, &mut gate)?;
                matvec(w_up, &n_ffn, &mut up)?;
                for (g, &u) in gate.iter_mut().zip(up.iter()) {
                    *g = gelu(*g) * u;
                }
                let mut down = vec![0f32; d];
                matvec(w_down, &gate, &mut down)?;

                // 8. Post-FFW RMSNorm & Residual add
                if let Some(w_post_ffw) = w_post_ffw_norm {
                    let mut norm_down = vec![0f32; d];
                    rmsnorm(&down, &w_post_ffw.data, eps, &mut norm_down);
                    for (h_i, &dl) in h.iter_mut().zip(norm_down.iter()) {
                        *h_i += dl;
                    }
                } else {
                    for (h_i, &dl) in h.iter_mut().zip(down.iter()) {
                        *h_i += dl;
                    }
                }

                // 9. Per-layer scalar output scale
                if let Some(w_scale) = w_out_scale {
                    let s = w_scale.data.first().copied().unwrap_or(1.0);
                    for val in h.iter_mut() {
                        *val *= s;
                    }
                }
            }

            // 10. Final RMSNorm
            let mut n_final = vec![0f32; d];
            rmsnorm(&h, &out_norm.data, eps, &mut n_final);

            // 11. LM Head
            let dst = &mut logits_out[ti * vocab..(ti + 1) * vocab];
            matvec(out_w, &n_final, dst)?;

            // 12. Final logit soft-capping
            let cap = cfg.final_logit_softcapping;
            if cap > 0.0 {
                for l in dst.iter_mut() {
                    *l = cap * (*l / cap).tanh();
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_gemma4_weights() -> (ModelConfig, DenseWeights) {
        let d = 8usize;
        let hd_swa = 4usize;
        let hd_global = 4usize;
        let nh = 2usize;
        let n_kv_swa = 1usize;
        let n_kv_global = 1usize;
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
        map.insert(
            "rope_freqs.weight".into(),
            (vec![1.0; hd_global / 2], [(hd_global / 2) as u32, 1, 1, 1]),
        );

        for layer in 0..6 {
            let is_swa = layer % 6 != 5;
            let hd = if is_swa { hd_swa } else { hd_global };
            let n_kv = if is_swa { n_kv_swa } else { n_kv_global };
            let b = format!("blk.{layer}");

            map.insert(
                format!("{b}.attn_norm.weight"),
                (vec![1.0; d], [d as u32, 1, 1, 1]),
            );
            map.insert(
                format!("{b}.ffn_norm.weight"),
                (vec![1.0; d], [d as u32, 1, 1, 1]),
            );
            map.insert(
                format!("{b}.post_attention_norm.weight"),
                (vec![1.0; d], [d as u32, 1, 1, 1]),
            );
            map.insert(
                format!("{b}.post_ffw_norm.weight"),
                (vec![1.0; d], [d as u32, 1, 1, 1]),
            );
            map.insert(
                format!("{b}.layer_output_scale.weight"),
                (vec![1.0; 1], [1, 1, 1, 1]),
            );

            map.insert(format!("{b}.ffn_gate.weight"), w(ff, d, 0.05));
            map.insert(format!("{b}.ffn_up.weight"), w(ff, d, 0.05));
            map.insert(format!("{b}.ffn_down.weight"), w(d, ff, 0.05));

            map.insert(format!("{b}.attn_q.weight"), w(nh * hd, d, 0.05));
            map.insert(format!("{b}.attn_k.weight"), w(n_kv * hd, d, 0.05));
            if is_swa {
                map.insert(format!("{b}.attn_v.weight"), w(n_kv * hd, d, 0.05));
            }
            map.insert(
                format!("{b}.attn_q_norm.weight"),
                (vec![1.0; hd], [hd as u32, 1, 1, 1]),
            );
            map.insert(
                format!("{b}.attn_k_norm.weight"),
                (vec![1.0; hd], [hd as u32, 1, 1, 1]),
            );
            map.insert(format!("{b}.attn_output.weight"), w(d, nh * hd, 0.05));
        }

        let cfg = ModelConfig {
            arch: "gemma4".into(),
            n_layers: 6,
            n_heads: nh,
            n_kv_heads: n_kv_swa,
            d_model: d,
            d_ffn: ff,
            vocab_size: v,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            rope_freq_base_swa: 10_000.0,
            key_length_swa: hd_swa,
            key_length: hd_global,
            final_logit_softcapping: 30.0,
            norm_eps: 1e-5,
            sliding_window: Some(4),
            ..Default::default()
        };
        (cfg, DenseWeights::from_map(map))
    }

    #[test]
    fn gemma4_forward_finite_and_deterministic() {
        let (cfg, weights) = tiny_gemma4_weights();
        let mut model = Gemma4::new(cfg.clone());
        model.weights = Some(weights);

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

        let prompt = [1u32, 2, 3];
        let mut logits1 = vec![0.0; prompt.len() * cfg.vocab_size];
        model
            .forward(&prompt, &mut logits1, &ctx)
            .expect("forward pass 1");

        assert!(
            logits1.iter().all(|x| x.is_finite()),
            "logits must be finite"
        );

        model.reset_cache();
        let mut logits2 = vec![0.0; prompt.len() * cfg.vocab_size];
        model
            .forward(&prompt, &mut logits2, &ctx)
            .expect("forward pass 2");

        assert_eq!(
            logits1, logits2,
            "logits must be deterministic across passes"
        );
    }
}
