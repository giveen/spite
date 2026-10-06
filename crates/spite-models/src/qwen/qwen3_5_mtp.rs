//! CPU Reference implementation for Qwen3.5 Multi-Token Prediction (MTP) block.
//!
//! MTP acts as a self-speculative draft head. The stem combines the normalized token
//! embedding and the normalized trunk hidden state, projects them through `eh_proj`,
//! passes them through one full-attention GQA transformer block with SwiGLU, and
//! projects through the shared LM head.

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_kvcache::{KvQuant, KvQuantConfig, VbrPolicy, VbrRows};
use spite_loader::GgufModel;

use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::{ModelConfig, ModelError};

struct MtpLayerState {
    k: VbrRows,
    v: VbrRows,
}

pub struct Qwen3_5Mtp {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    kv_quant: RwLock<KvQuantConfig>,
    state: RwLock<Vec<MtpLayerState>>,
}

impl Qwen3_5Mtp {
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

    /// Run one MTP draft step.
    ///
    /// Takes:
    /// - `prev_token`: token sampled from trunk at step t ($x_{t+1}$)
    /// - `trunk_hidden`: final hidden state of the trunk at step t ($h_t$), length `d_model`
    /// - `pos`: sequence position
    /// - `logits_out`: slice of length `vocab_size` to receive draft logits for step t+2 ($\hat{x}_{t+2}$)
    /// - `next_hidden_out`: optional slice of length `d_model` to receive $h_{t+1}^{mtp}$ for multi-step chaining
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

        // MTP block index is typically the layer immediately following the trunk
        let mtp_layer = cfg.n_layers;
        let b = format!("blk.{mtp_layer}");

        // 1. Fetch stem weights
        let embd = w.get("token_embd.weight")?;
        let w_enorm = w.get(&format!("{b}.nextn.enorm.weight"))?;
        let w_hnorm = w.get(&format!("{b}.nextn.hnorm.weight"))?;
        let w_eh_proj = w.get(&format!("{b}.nextn.eh_proj.weight"))?;

        // 2. Token embedding + norm
        let tok_slice = &embd.data[prev_token as usize * d..(prev_token as usize + 1) * d];
        let mut e_norm = vec![0f32; d];
        rmsnorm(tok_slice, &w_enorm.data, cfg.norm_eps, &mut e_norm);

        // 3. Trunk hidden norm
        let mut h_norm = vec![0f32; d];
        rmsnorm(trunk_hidden, &w_hnorm.data, cfg.norm_eps, &mut h_norm);

        // 4. Concatenation: [e_norm || h_norm] in R^{2d}
        let mut concat = Vec::with_capacity(2 * d);
        concat.extend_from_slice(&e_norm);
        concat.extend_from_slice(&h_norm);

        // 5. Project through eh_proj -> initial residual x in R^d
        let mut x = vec![0f32; d];
        matvec(w_eh_proj, &concat, &mut x)?;

        // 6. Transformer block
        let w_attn_norm = w.get(&format!("{b}.attn_norm.weight"))?;
        let mut n_attn = vec![0f32; d];
        rmsnorm(&x, &w_attn_norm.data, cfg.norm_eps, &mut n_attn);

        // Attention projections
        let w_q = w.get(&format!("{b}.attn_q.weight"))?;
        let w_k = w.get(&format!("{b}.attn_k.weight"))?;
        let w_v = w.get(&format!("{b}.attn_v.weight"))?;
        let mut q = vec![0f32; n_heads * head_dim];
        let mut k = vec![0f32; n_kv_heads * head_dim];
        let mut v = vec![0f32; n_kv_heads * head_dim];
        matvec(w_q, &n_attn, &mut q)?;
        matvec(w_k, &n_attn, &mut k)?;
        matvec(w_v, &n_attn, &mut v)?;

        // QK per-head RMSNorm
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

        // RoPE
        let rope = spite_rope::RopeConfig {
            head_dim,
            theta: cfg.rope_theta,
            ..Default::default()
        };
        spite_rope::apply_rope(&mut qn, pos as u32, &rope)
            .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;
        spite_rope::apply_rope(&mut kn, pos as u32, &rope)
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
            state.push(MtpLayerState {
                k: VbrRows::new(kv_row_len, VbrPolicy::from_ctx(cfg.max_seq_len, kq)),
                v: VbrRows::new(kv_row_len, VbrPolicy::from_ctx(cfg.max_seq_len, vq)),
            });
        }

        state[0].k.push(&kn);
        state[0].v.push(&v);
        let n_prev = state[0].k.len();
        let kd = state[0].k.to_f32();
        let vd = state[0].v.to_f32();

        let attn_cfg = FlashAttnConfig::new(1, n_prev, n_heads, n_kv_heads, head_dim);
        let mut attn_out = vec![0f32; n_heads * head_dim];
        scalar_attention(&qn, &kd, &vd, &mut attn_out, &attn_cfg)
            .map_err(|e| ModelError::Forward(format!("mtp attn: {e}")))?;

        let w_o = w.get(&format!("{b}.attn_output.weight"))?;
        let mut proj = vec![0f32; d];
        matvec(w_o, &attn_out, &mut proj)?;

        // Residual add
        for (x_i, p) in x.iter_mut().zip(proj.iter()) {
            *x_i += p;
        }

        // Post-attention norm + SwiGLU FFN
        let w_post_norm = w.get(&format!("{b}.ffn_norm.weight"))?;
        let w_gate = w.get(&format!("{b}.ffn_gate.weight"))?;
        let w_up = w.get(&format!("{b}.ffn_up.weight"))?;
        let w_down = w.get(&format!("{b}.ffn_down.weight"))?;

        let mut pn = vec![0f32; d];
        rmsnorm(&x, &w_post_norm.data, cfg.norm_eps, &mut pn);

        let mut gate = vec![0f32; d_ffn];
        let mut up = vec![0f32; d_ffn];
        matvec(w_gate, &pn, &mut gate)?;
        matvec(w_up, &pn, &mut up)?;
        for (g, &u) in gate.iter_mut().zip(up.iter()) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
        let mut down = vec![0f32; d];
        matvec(w_down, &gate, &mut down)?;

        for (x_i, dl) in x.iter_mut().zip(down.iter()) {
            *x_i += dl;
        }

        // 7. Shared Head Norm & LM Head
        let w_head_norm = w
            .get(&format!("{b}.nextn.shared_head_norm.weight"))
            .or_else(|_| w.get("output_norm.weight"))?;
        let mut h_final = vec![0f32; d];
        rmsnorm(&x, &w_head_norm.data, cfg.norm_eps, &mut h_final);

        if let Some(out_h) = next_hidden_out
            && out_h.len() == d
        {
            out_h.copy_from_slice(&h_final);
        }

        let out_w = w
            .get("output.weight")
            .or_else(|_| w.get("token_embd.weight"))?;
        matvec(out_w, &h_final, logits_out)?;

        Ok(())
    }
}
