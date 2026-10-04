//! Model architecture implementations.
//!
//! Each architecture is a separate module inside its family directory.
//! The host forward-pass loop lives here; GPU kernels live under kernels/
//! and are called via spite-dispatch.
//!
//! # Directory layout
//!
//!   src/<family>/mod.rs      — exports all variants in that family
//!   src/<family>/<model>.rs  — implements ModelArch for one variant
//!   kernels/<family>/<model>/<gpu_arch>/  — kernel .so per GPU target
//!
//! # Adding a new architecture
//!
//!   1. Create src/<family>/<model>.rs implementing ModelArch
//!   2. Add it to src/<family>/mod.rs
//!   3. Register GGUF arch strings in ArchRegistry::default() below
//!   4. Add kernel dirs: kernels/<family>/<model>/<gpu_arch>/

// 2026 model families. Each family keeps only its current-generation
// variant; older generations were removed. See kernels/<family>/<model>/.
pub mod deepseek;
pub mod dense;
pub mod eagle;
pub mod gemma;
pub mod glm;
pub mod kimi;
pub mod llama;
pub mod mellum;
pub mod minimax;
pub mod mistral;
pub mod moe;
pub mod qwen;

use spite_abi::SpiteCtx;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unknown architecture: {0}")]
    UnknownArch(String),
    #[error("weight not found: {0}")]
    MissingWeight(String),
    #[error("shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<u32>,
        actual: Vec<u32>,
    },
    #[error("forward pass error: {0}")]
    Forward(String),
}

/// Hyperparameters common to all transformer architectures.
/// Populated from GGUF metadata by `spite-loader`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_model: usize,
    pub d_ffn: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// Sliding-window span for local-attention archs. None = full attention.
    pub sliding_window: Option<usize>,
    /// Gated-delta-net geometry for hybrid archs (0 = not hybrid).
    pub ssm_d_conv: usize,
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_dt_rank: usize,
    pub ssm_n_group: usize,
    /// iRoPE dimension sections (Qwen3.5-style interleaved rope).
    pub rope_sections: [u32; 4],
    /// Per-layer recurrent flags for hybrid archs.
    pub recurrent_layers: Vec<bool>,
    /// MoE geometry (0 experts = dense arch).
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub moe_layer_step: usize,
    pub expert_weights_scale: f32,
    /// Per-layer SWA flags (llama4-style chunked pattern fallback).
    pub swa_layers: Vec<bool>,
    /// Per-layer SwiGLU clamp limits (empty = no clamp).
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,
    /// MLA / KDA geometry (0 = unused).
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub key_length: usize,
    pub value_length: usize,
    pub rope_dim_count: usize,
    pub kda_head_dim: usize,
    pub kda_gate_lower_bound: f32,
    pub n_layer_dense_lead: usize,
    pub n_expert_latent: usize,
    pub expert_weights_norm: bool,
    pub expert_gating_func: u32,
    pub attn_res_block_size: usize,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub head_count_kv_arr: Vec<usize>,
    pub hc_mult: usize,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: usize,
    pub o_group_count: usize,
    pub o_lora_rank: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_base: f32,
    pub n_expert_shared: usize,

    // ── RoPE scaling (YaRN / linear / NTK) ───────────────────────────────
    /// Multiplicative rope scale factor. 1.0 = no scaling (default).
    /// Linear scaling: divide all frequencies by this factor.
    pub rope_scale_factor: f32,
    /// Original training context length, used by YaRN to compute the
    /// interpolation factor. 0 means "use max_seq_len".
    pub rope_original_ctx: usize,
    /// YaRN β_fast: high-frequency threshold (dimensions above this get
    /// no interpolation). Typical: 32.0.
    pub yarn_beta_fast: f32,
    /// YaRN β_slow: low-frequency threshold (dimensions below this get
    /// linear scaling). Typical: 1.0.
    pub yarn_beta_slow: f32,
    /// YaRN attention factor (scales the attention output after YaRN).
    /// 0.0 = compute from scale_factor automatically.
    pub yarn_attn_factor: f32,
}

impl ModelConfig {
    /// Return the rope scaling factor, defaulting to 1.0 when unset.
    pub fn effective_rope_scale(&self) -> f32 {
        if self.rope_scale_factor <= 0.0 {
            1.0
        } else {
            self.rope_scale_factor
        }
    }

    /// True if any YaRN extension parameters are active.
    pub fn uses_yarn(&self) -> bool {
        self.rope_original_ctx > 0 && self.rope_scale_factor > 1.0
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            arch: String::new(),
            n_layers: 0,
            n_heads: 0,
            n_kv_heads: 0,
            d_model: 0,
            d_ffn: 0,
            vocab_size: 0,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            sliding_window: None,
            ssm_d_conv: 0,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_dt_rank: 0,
            ssm_n_group: 0,
            rope_sections: [0; 4],
            recurrent_layers: Vec::new(),
            n_expert: 0,
            n_expert_used: 0,
            moe_layer_step: 0,
            expert_weights_scale: 0.0,
            swa_layers: Vec::new(),
            swiglu_clamp_exp: Vec::new(),
            swiglu_clamp_shexp: Vec::new(),
            q_lora_rank: 0,
            kv_lora_rank: 0,
            key_length: 0,
            value_length: 0,
            rope_dim_count: 0,
            kda_head_dim: 0,
            kda_gate_lower_bound: f32::NEG_INFINITY,
            n_layer_dense_lead: 0,
            n_expert_latent: 0,
            expert_weights_norm: false,
            expert_gating_func: 0,
            attn_res_block_size: 0,
            situ_beta: 0.0,
            situ_linear_beta: 0.0,
            head_count_kv_arr: Vec::new(),
            hc_mult: 0,
            hc_eps: 0.0,
            hc_sinkhorn_iters: 0,
            o_group_count: 0,
            o_lora_rank: 0,
            compress_ratios: Vec::new(),
            compress_rope_base: 0.0,
            n_expert_shared: 0,
            rope_scale_factor: 1.0,
            rope_original_ctx: 0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_attn_factor: 0.0,
        }
    }
}

impl From<spite_loader::config::ModelHyperparams> for ModelConfig {
    fn from(h: spite_loader::config::ModelHyperparams) -> Self {
        Self {
            arch: h.arch,
            n_layers: h.n_layers as usize,
            n_heads: h.n_heads as usize,
            n_kv_heads: h.n_kv_heads.max(1) as usize,
            d_model: h.d_model as usize,
            d_ffn: h.d_ffn as usize,
            vocab_size: h.vocab_size as usize,
            max_seq_len: h.max_seq_len.max(1) as usize,
            rope_theta: h.rope_theta,
            norm_eps: h.norm_eps,
            sliding_window: (h.sliding_window > 0).then_some(h.sliding_window as usize),
            ssm_d_conv: h.ssm_d_conv as usize,
            ssm_d_inner: h.ssm_d_inner as usize,
            ssm_d_state: h.ssm_d_state as usize,
            ssm_dt_rank: h.ssm_dt_rank as usize,
            ssm_n_group: h.ssm_n_group as usize,
            rope_sections: h.rope_sections,
            recurrent_layers: h.recurrent_layers,
            n_expert: h.n_expert as usize,
            n_expert_used: h.n_expert_used as usize,
            moe_layer_step: h.moe_layer_step as usize,
            expert_weights_scale: h.expert_weights_scale,
            swa_layers: h.swa_layers,
            swiglu_clamp_exp: h.swiglu_clamp_exp,
            swiglu_clamp_shexp: h.swiglu_clamp_shexp,
            q_lora_rank: h.q_lora_rank as usize,
            kv_lora_rank: h.kv_lora_rank as usize,
            key_length: h.key_length as usize,
            value_length: h.value_length as usize,
            rope_dim_count: h.rope_dim_count as usize,
            kda_head_dim: h.kda_head_dim as usize,
            kda_gate_lower_bound: h.kda_gate_lower_bound,
            n_layer_dense_lead: h.n_layer_dense_lead as usize,
            n_expert_latent: h.n_expert_latent as usize,
            expert_weights_norm: h.expert_weights_norm,
            expert_gating_func: h.expert_gating_func,
            attn_res_block_size: h.attn_res_block_size as usize,
            situ_beta: h.situ_beta,
            situ_linear_beta: h.situ_linear_beta,
            head_count_kv_arr: h.head_count_kv_arr.iter().map(|&x| x as usize).collect(),
            hc_mult: h.hc_mult as usize,
            hc_eps: h.hc_eps,
            hc_sinkhorn_iters: h.hc_sinkhorn_iters as usize,
            o_group_count: h.o_group_count as usize,
            o_lora_rank: h.o_lora_rank as usize,
            compress_ratios: h.compress_ratios.iter().map(|&x| x as usize).collect(),
            compress_rope_base: h.compress_rope_base,
            n_expert_shared: h.n_expert_shared as usize,
            ..Default::default()
        }
    }
}

/// Every model architecture must implement this trait.
pub trait ModelArch: Send + Sync {
    fn config(&self) -> &ModelConfig;

    /// Bind the mmap'd GGUF weight tensors to this model.
    ///
    /// Called once after construction, before the first `forward`.
    /// The default no-op is intentional — implementations override it
    /// to store tensor pointers from the GGUF buffer.
    fn load_weights(&mut self, _model: &spite_loader::GgufModel) -> Result<(), ModelError> {
        Ok(())
    }

    /// Clear cached K/V state. Called when starting a new sequence.
    fn reset_cache(&self) {}

    /// Run one forward pass.
    ///
    /// `tokens`:     input token ids `[seq_len]`
    /// `logits_out`: pre-allocated `[seq_len × vocab_size]` F32 — caller zeroes
    /// `ctx`:        batch/threading/position context (pos, n_heads, n_kv_heads)
    fn forward(
        &self,
        tokens: &[u32],
        logits_out: &mut [f32],
        ctx: &SpiteCtx,
    ) -> Result<(), ModelError>;
}

/// Maps GGUF `general.architecture` strings to constructors.
pub struct ArchRegistry {
    entries: Vec<(&'static str, ArchCtor)>,
}

/// Constructor for one model architecture from its `ModelConfig`.
type ArchCtor = fn(ModelConfig) -> Box<dyn ModelArch>;

impl Default for ArchRegistry {
    fn default() -> Self {
        Self {
            entries: vec![
                // ── Llama family ─────────────────────────────────────────────
                ("llama4", |c| Box::new(llama::Llama4::new(c))),
                // ── Mistral family ───────────────────────────────────────────
                ("mistral4", |c| Box::new(mistral::Mistral4::new(c))),
                ("magistral", |c| Box::new(mistral::Mistral4::new(c))),
                // ── Qwen family ──────────────────────────────────────────────
                ("qwen35", |c| Box::new(qwen::Qwen3_5::new(c))),
                ("qwen35moe", |c| Box::new(qwen::Qwen3_5::new(c))),
                ("qwen4", |c| Box::new(qwen::Qwen4::new(c))),
                ("qwen4exp", |c| Box::new(qwen::Qwen4::new(c))),
                // ── DeepSeek family ──────────────────────────────────────────
                ("deepseek4", |c| Box::new(deepseek::DeepSeekV4::new(c))),
                // ── Gemma family ─────────────────────────────────────────────
                ("gemma4", |c| Box::new(gemma::Gemma4::new(c))),
                // ── GLM family ───────────────────────────────────────────────
                ("glm-dsa", |c| Box::new(glm::GlmDsa::new(c))),
                ("glm5", |c| Box::new(glm::Glm5::new(c))),
                ("glm5-next", |c| Box::new(glm::Glm5::new(c))),
                // ── MiniMax family ───────────────────────────────────────────
                ("minimax-m3", |c| Box::new(minimax::MinimaxM3::new(c))),
                // ── Kimi family ──────────────────────────────────────────────
                ("kimi-k3", |c| Box::new(kimi::KimiK3::new(c))),
                // ── Draft / speculative ──────────────────────────────────────
                ("eagle3", |c| Box::new(eagle::Eagle3::new(c))),
                // ── Code completion ──────────────────────────────────────────
                ("mellum", |c| Box::new(mellum::Mellum::new(c))),
            ],
        }
    }
}

impl ArchRegistry {
    pub fn build(&self, config: ModelConfig) -> Result<Box<dyn ModelArch>, ModelError> {
        let arch = config.arch.clone();
        self.entries
            .iter()
            .find(|(name, _)| *name == arch.as_str())
            .map(|(_, ctor)| ctor(config))
            .ok_or(ModelError::UnknownArch(arch))
    }
}
