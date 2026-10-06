//! CPU Reference implementation for Gemma 4 Assistant / Multi-Token Prediction (MTP) block.
//!
//! Follows the Gemma 4 assistant architecture:
//! - Scaled token embedding (multiplied by sqrt(d_model)) concatenated with trunk hidden state
//! - Pre-projection through `nextn_proj_pre.weight` into draft residual stream
//! - Transformer blocks with Gemma 4 norms and GeGLU FFN
//! - Post-projection through `nextn_proj_post.weight` into continuation hidden representation
//! - Shared LM head projection (tied to token_embd.weight)

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_kvcache::{KvQuant, KvQuantConfig, VbrPolicy, VbrRows};
use spite_loader::GgufModel;

use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::{ModelConfig, ModelError};

struct GemmaMtpLayerState {
    k: VbrRows,
    v: VbrRows,
}

pub struct Gemma4Mtp {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    kv_quant: RwLock<KvQuantConfig>,
    state: RwLock<Vec<GemmaMtpLayerState>>,
}

impl Gemma4Mtp {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv_quant: RwLock::new(KvQuantConfig::full_precision()),
            state: RwLock::new(Vec::new()),
        }
    }

    pub fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    pub fn reset_cache(&self) {
        if let Ok(mut s) = self.state.write() {
            s.clear();
        }
    }

    /// Run one Gemma 4 assistant / MTP draft step.
    pub fn forward_step(
        &self,
        prev_token: u32,
        trunk_hidden: &[f32],
        pos: usize,
        logits_out: &mut [f32],
        next_hidden_out: Option<&mut [f32]>,
        _ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let Some(w) = &self.weights else {
            return Err(ModelError::Forward("weights not loaded".into()));
        };
        let cfg = &self.config;
        let d = cfg.d_model;
        let d_ffn = cfg.d_ffn;
        let n_heads = cfg.n_heads.max(1);
        let n_kv_heads = cfg.n_kv_heads.max(1);
        let head_dim = d / n_heads;
        let vocab = cfg.vocab_size;
        let eps = cfg.norm_eps;

        if trunk_hidden.len() != d {
            return Err(ModelError::Forward(format!(
                "trunk_hidden length mismatch: expected {d}, got {}",
                trunk_hidden.len()
            )));
        }
        if logits_out.len() != vocab {
            return Err(ModelError::Forward(format!(
                "logits_out length mismatch: expected {vocab}, got {}",
                logits_out.len()
            )));
        }

        // 1. Scaled token embedding
        let embd = w.get("token_embd.weight")?;
        let tok_slice = &embd.data[prev_token as usize * d..(prev_token as usize + 1) * d];
        let emb_scale = (d as f32).sqrt();
        let mut scaled_emb = vec![0f32; d];
        for (out, &val) in scaled_emb.iter_mut().zip(tok_slice.iter()) {
            *out = val * emb_scale;
        }

        // 2. Concatenate: [scaled_emb || trunk_hidden] in R^{2d}
        let mut concat = Vec::with_capacity(2 * d);
        concat.extend_from_slice(&scaled_emb);
        concat.extend_from_slice(trunk_hidden);

        // 3. Pre-projection: nextn_proj_pre
        let w_pre = w
            .get("nextn_proj_pre.weight")
            .or_else(|_| w.get(&format!("blk.{}.nextn.eh_proj.weight", cfg.n_layers)))?;
        let mut x = vec![0f32; d];
        matvec(w_pre, &concat, &mut x)?;

        // 4. Draft transformer block
        let mtp_layer = cfg.n_layers;
        let b = format!("blk.{mtp_layer}");

        let w_attn_norm = w
            .get(&format!("{b}.attn_norm.weight"))
            .or_else(|_| w.get("blk.0.attn_norm.weight"))?;
        let mut n_attn = vec![0f32; d];
        rmsnorm(&x, &w_attn_norm.data, eps, &mut n_attn);

        // QKV
        let w_q = w
            .get(&format!("{b}.attn_q.weight"))
            .or_else(|_| w.get("blk.0.attn_q.weight"))?;
        let w_k = w
            .get(&format!("{b}.attn_k.weight"))
            .or_else(|_| w.get("blk.0.attn_k.weight"))?;
        let w_v = w
            .get(&format!("{b}.attn_v.weight"))
            .or_else(|_| w.get("blk.0.attn_v.weight"))
            .or_else(|_| w.get(&format!("{b}.attn_k.weight")))
            .or_else(|_| w.get("blk.0.attn_k.weight"))?;

        let mut q = vec![0f32; n_heads * head_dim];
        let mut k = vec![0f32; n_kv_heads * head_dim];
        let mut v = vec![0f32; n_kv_heads * head_dim];
        matvec(w_q, &n_attn, &mut q)?;
        matvec(w_k, &n_attn, &mut k)?;
        matvec(w_v, &n_attn, &mut v)?;

        // QK per-head RMSNorm
        if let Ok(w_qn) = w
            .get(&format!("{b}.attn_q_norm.weight"))
            .or_else(|_| w.get("blk.0.attn_q_norm.weight"))
        {
            let mut qn = vec![0f32; q.len()];
            for h in 0..n_heads {
                rmsnorm(
                    &q[h * head_dim..(h + 1) * head_dim],
                    &w_qn.data,
                    eps,
                    &mut qn[h * head_dim..(h + 1) * head_dim],
                );
            }
            q = qn;
        }

        if let Ok(w_kn) = w
            .get(&format!("{b}.attn_k_norm.weight"))
            .or_else(|_| w.get("blk.0.attn_k_norm.weight"))
        {
            let mut kn = vec![0f32; k.len()];
            for h in 0..n_kv_heads {
                rmsnorm(
                    &k[h * head_dim..(h + 1) * head_dim],
                    &w_kn.data,
                    eps,
                    &mut kn[h * head_dim..(h + 1) * head_dim],
                );
            }
            k = kn;
        }

        // RoPE
        let rope = spite_rope::RopeConfig {
            head_dim,
            theta: cfg.rope_theta,
            ..Default::default()
        };
        spite_rope::apply_rope(&mut q, pos as u32, &rope)
            .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
        spite_rope::apply_rope(&mut k, pos as u32, &rope)
            .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;

        // KV Cache
        let mut state = self
            .state
            .write()
            .map_err(|_| ModelError::Forward("state lock".into()))?;
        if state.is_empty() {
            let kv_row_len = n_kv_heads * head_dim;
            let (kq, vq) = self
                .kv_quant
                .read()
                .map(|c| (c.key, c.val))
                .unwrap_or((KvQuant::F32, KvQuant::F32));
            state.push(GemmaMtpLayerState {
                k: VbrRows::new(kv_row_len, VbrPolicy::from_ctx(cfg.max_seq_len, kq)),
                v: VbrRows::new(kv_row_len, VbrPolicy::from_ctx(cfg.max_seq_len, vq)),
            });
        }

        state[0].k.push(&k);
        state[0].v.push(&v);
        let n_prev = state[0].k.len();
        let kd = state[0].k.to_f32();
        let vd = state[0].v.to_f32();

        let attn_cfg = FlashAttnConfig::new(1, n_prev, n_heads, n_kv_heads, head_dim);
        let mut attn_out = vec![0f32; n_heads * head_dim];
        scalar_attention(&q, &kd, &vd, &mut attn_out, &attn_cfg)
            .map_err(|e| ModelError::Forward(format!("gemma mtp attn: {e}")))?;

        let w_o = w
            .get(&format!("{b}.attn_output.weight"))
            .or_else(|_| w.get("blk.0.attn_output.weight"))?;
        let mut proj = vec![0f32; d];
        matvec(w_o, &attn_out, &mut proj)?;

        // Post-attention norm
        if let Ok(w_post_attn) = w
            .get(&format!("{b}.post_attention_norm.weight"))
            .or_else(|_| w.get("blk.0.post_attention_norm.weight"))
        {
            let mut post_norm = vec![0f32; d];
            rmsnorm(&proj, &w_post_attn.data, eps, &mut post_norm);
            proj = post_norm;
        }

        // Residual
        for (x_i, p) in x.iter_mut().zip(proj.iter()) {
            *x_i += p;
        }

        // FFN: GeGLU
        let w_ffn_norm = w
            .get(&format!("{b}.ffn_norm.weight"))
            .or_else(|_| w.get("blk.0.ffn_norm.weight"))?;
        let w_gate = w
            .get(&format!("{b}.ffn_gate.weight"))
            .or_else(|_| w.get("blk.0.ffn_gate.weight"))?;
        let w_up = w
            .get(&format!("{b}.ffn_up.weight"))
            .or_else(|_| w.get("blk.0.ffn_up.weight"))?;
        let w_down = w
            .get(&format!("{b}.ffn_down.weight"))
            .or_else(|_| w.get("blk.0.ffn_down.weight"))?;

        let mut pn = vec![0f32; d];
        rmsnorm(&x, &w_ffn_norm.data, eps, &mut pn);

        let mut gate = vec![0f32; d_ffn];
        let mut up = vec![0f32; d_ffn];
        matvec(w_gate, &pn, &mut gate)?;
        matvec(w_up, &pn, &mut up)?;

        // GeGLU activation
        for (g, &u) in gate.iter_mut().zip(up.iter()) {
            let x_val = *g;
            let gelu =
                0.5 * x_val * (1.0 + (0.797_884_6 * (x_val + 0.044715 * x_val.powi(3))).tanh());
            *g = gelu * u;
        }

        let mut down = vec![0f32; d];
        matvec(w_down, &gate, &mut down)?;

        if let Ok(w_post_ffw) = w
            .get(&format!("{b}.post_ffw_norm.weight"))
            .or_else(|_| w.get("blk.0.post_ffw_norm.weight"))
        {
            let mut post_norm = vec![0f32; d];
            rmsnorm(&down, &w_post_ffw.data, eps, &mut post_norm);
            down = post_norm;
        }

        for (x_i, dl) in x.iter_mut().zip(down.iter()) {
            *x_i += dl;
        }

        // Layer output scale if present
        if let Ok(w_scale) = w
            .get(&format!("{b}.layer_output_scale.weight"))
            .or_else(|_| w.get("blk.0.layer_output_scale.weight"))
            && !w_scale.data.is_empty()
        {
            let s = w_scale.data[0];
            for val in x.iter_mut() {
                *val *= s;
            }
        }

        // Post projection to next hidden state
        if let Ok(w_post) = w.get("nextn_proj_post.weight") {
            let mut h_post = vec![0f32; d];
            matvec(w_post, &x, &mut h_post)?;
            if let Some(out_h) = next_hidden_out
                && out_h.len() == d
            {
                out_h.copy_from_slice(&h_post);
            }
        } else if let Some(out_h) = next_hidden_out
            && out_h.len() == d
        {
            out_h.copy_from_slice(&x);
        }

        // Final norm & output
        let out_norm = w
            .get(&format!("{b}.nextn.shared_head_norm.weight"))
            .or_else(|_| w.get("output_norm.weight"))?;
        let mut final_normed = vec![0f32; d];
        rmsnorm(&x, &out_norm.data, eps, &mut final_normed);

        let out_w = w
            .get("output.weight")
            .or_else(|_| w.get("token_embd.weight"))?;
        matvec(out_w, &final_normed, logits_out)?;

        // Logit soft-capping
        if cfg.final_logit_softcapping > 0.0 {
            let cap = cfg.final_logit_softcapping;
            for val in logits_out.iter_mut() {
                *val = cap * (*val / cap).tanh();
            }
        }

        Ok(())
    }
}
