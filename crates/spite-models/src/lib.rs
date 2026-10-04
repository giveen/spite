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
pub mod llama;
pub mod mistral;
pub mod qwen;
pub mod deepseek;
pub mod gemma;
pub mod glm;
pub mod minimax;
pub mod kimi;
pub mod eagle;
pub mod mellum;

use spite_abi::SpiteCtx;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unknown architecture: {0}")]
    UnknownArch(String),
    #[error("weight not found: {0}")]
    MissingWeight(String),
    #[error("shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { name: String, expected: Vec<u32>, actual: Vec<u32> },
    #[error("forward pass error: {0}")]
    Forward(String),
}

/// Hyperparameters common to all transformer architectures.
/// Populated from GGUF metadata by `spite-loader`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch:         String,
    pub n_layers:     usize,
    pub n_heads:      usize,
    pub n_kv_heads:   usize,
    pub d_model:      usize,
    pub d_ffn:        usize,
    pub vocab_size:   usize,
    pub max_seq_len:  usize,
    pub rope_theta:   f32,
    pub norm_eps:     f32,

    // ── RoPE scaling (YaRN / linear / NTK) ───────────────────────────────
    /// Multiplicative rope scale factor. 1.0 = no scaling (default).
    /// Linear scaling: divide all frequencies by this factor.
    pub rope_scale_factor: f32,
    /// Original training context length, used by YaRN to compute the
    /// interpolation factor. 0 means "use max_seq_len".
    pub rope_original_ctx: usize,
    /// YaRN β_fast: high-frequency threshold (dimensions above this get
    /// no interpolation). Typical: 32.0.
    pub yarn_beta_fast:    f32,
    /// YaRN β_slow: low-frequency threshold (dimensions below this get
    /// linear scaling). Typical: 1.0.
    pub yarn_beta_slow:    f32,
    /// YaRN attention factor (scales the attention output after YaRN).
    /// 0.0 = compute from scale_factor automatically.
    pub yarn_attn_factor:  f32,
}

impl ModelConfig {
    /// Return the rope scaling factor, defaulting to 1.0 when unset.
    pub fn effective_rope_scale(&self) -> f32 {
        if self.rope_scale_factor <= 0.0 { 1.0 } else { self.rope_scale_factor }
    }

    /// True if any YaRN extension parameters are active.
    pub fn uses_yarn(&self) -> bool {
        self.rope_original_ctx > 0 && self.rope_scale_factor > 1.0
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            arch:              String::new(),
            n_layers:          0,
            n_heads:           0,
            n_kv_heads:        0,
            d_model:           0,
            d_ffn:             0,
            vocab_size:        0,
            max_seq_len:       4096,
            rope_theta:        10000.0,
            norm_eps:          1e-5,
            rope_scale_factor: 1.0,
            rope_original_ctx: 0,
            yarn_beta_fast:    32.0,
            yarn_beta_slow:    1.0,
            yarn_attn_factor:  0.0,
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

    /// Run one forward pass.
    ///
    /// `tokens`:     input token ids `[seq_len]`
    /// `logits_out`: pre-allocated `[seq_len × vocab_size]` F32 — caller zeroes
    /// `ctx`:        batch/threading/position context (pos, n_heads, n_kv_heads)
    fn forward(
        &self,
        tokens:     &[u32],
        logits_out: &mut [f32],
        ctx:        &SpiteCtx,
    ) -> Result<(), ModelError>;
}

/// Maps GGUF `general.architecture` strings to constructors.
pub struct ArchRegistry {
    entries: Vec<(&'static str, fn(ModelConfig) -> Box<dyn ModelArch>)>,
}

impl Default for ArchRegistry {
    fn default() -> Self {
        Self {
            entries: vec![
                // ── Llama family ─────────────────────────────────────────────
                ("llama4",       |c| Box::new(llama::Llama4::new(c))),

                // ── Mistral family ───────────────────────────────────────────
                ("mistral4",     |c| Box::new(mistral::Mistral4::new(c))),
                ("magistral",    |c| Box::new(mistral::Mistral4::new(c))),

                // ── Qwen family ──────────────────────────────────────────────
                ("qwen35",       |c| Box::new(qwen::Qwen3_5::new(c))),
                ("qwen35moe",    |c| Box::new(qwen::Qwen3_5::new(c))),
                ("qwen4",        |c| Box::new(qwen::Qwen4::new(c))),
                ("qwen4exp",     |c| Box::new(qwen::Qwen4::new(c))),

                // ── DeepSeek family ──────────────────────────────────────────
                ("deepseek4",    |c| Box::new(deepseek::DeepSeekV4::new(c))),

                // ── Gemma family ─────────────────────────────────────────────
                ("gemma4",       |c| Box::new(gemma::Gemma4::new(c))),

                // ── GLM family ───────────────────────────────────────────────
                ("glm-dsa",      |c| Box::new(glm::GlmDsa::new(c))),
                ("glm5",         |c| Box::new(glm::Glm5::new(c))),
                ("glm5-next",    |c| Box::new(glm::Glm5::new(c))),

                // ── MiniMax family ───────────────────────────────────────────
                ("minimax-m3",   |c| Box::new(minimax::MinimaxM3::new(c))),

                // ── Kimi family ──────────────────────────────────────────────
                ("kimi-k3",      |c| Box::new(kimi::KimiK3::new(c))),

                // ── Draft / speculative ──────────────────────────────────────
                ("eagle3",       |c| Box::new(eagle::Eagle3::new(c))),

                // ── Code completion ──────────────────────────────────────────
                ("mellum",       |c| Box::new(mellum::Mellum::new(c))),
            ],
        }
    }
}

impl ArchRegistry {
    pub fn build(&self, config: ModelConfig) -> Result<Box<dyn ModelArch>, ModelError> {
        let arch = config.arch.clone();
        self.entries.iter()
            .find(|(name, _)| *name == arch.as_str())
            .map(|(_, ctor)| ctor(config))
            .ok_or(ModelError::UnknownArch(arch))
    }
}
