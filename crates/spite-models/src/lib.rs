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

// Existing families
pub mod llama;
pub mod mistral;
pub mod phi;

// 2025 model families
pub mod qwen;
pub mod deepseek;
pub mod gemma;
pub mod falcon;
pub mod rwkv;
pub mod mamba;
pub mod glm;
pub mod granite;
pub mod nemotron;
pub mod olmo;
pub mod jamba;
pub mod minimax;
pub mod modern_bert;

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
}

/// Every model architecture must implement this trait.
pub trait ModelArch: Send + Sync {
    fn config(&self) -> &ModelConfig;

    /// Run one forward pass.
    ///
    /// `tokens`:     input token ids `[seq_len]`
    /// `logits_out`: pre-allocated `[seq_len × vocab_size]` F32 — caller zeroes
    /// `ctx`:        batch/threading context
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
                ("llama",        |c| Box::new(llama::Llama3::new(c))),
                ("llama3",       |c| Box::new(llama::Llama3::new(c))),
                ("llama4",       |c| Box::new(llama::Llama4::new(c))),

                // ── Mistral family ───────────────────────────────────────────
                ("mistral",      |c| Box::new(mistral::Mistral::new(c))),
                ("mistral3",     |c| Box::new(mistral::Mistral3::new(c))),

                // ── Phi family ───────────────────────────────────────────────
                ("phi3",         |c| Box::new(phi::Phi3::new(c))),
                ("phi4",         |c| Box::new(phi::Phi4::new(c))),

                // ── Qwen family ──────────────────────────────────────────────
                ("qwen3",        |c| Box::new(qwen::Qwen3::new(c))),
                ("qwen3moe",     |c| Box::new(qwen::Qwen3Moe::new(c))),
                ("qwen3next",    |c| Box::new(qwen::QwQ::new(c))),
                ("qwq",          |c| Box::new(qwen::QwQ::new(c))),
                ("qwen35",       |c| Box::new(qwen::Qwen3_5::new(c))),
                ("qwen35moe",    |c| Box::new(qwen::Qwen3_5::new(c))),
                ("qwen3vl",      |c| Box::new(qwen::Qwen3Vl::new(c))),

                // ── DeepSeek family ──────────────────────────────────────────
                ("deepseek2",    |c| Box::new(deepseek::DeepSeekV3::new(c))),
                ("deepseek32",   |c| Box::new(deepseek::DeepSeekV3::new(c))),

                // ── Gemma family ─────────────────────────────────────────────
                ("gemma3",       |c| Box::new(gemma::Gemma3::new(c))),
                ("gemma3n",      |c| Box::new(gemma::Gemma3n::new(c))),

                // ── Falcon family ────────────────────────────────────────────
                ("falcon-h1",    |c| Box::new(falcon::FalconH1::new(c))),

                // ── RWKV family ──────────────────────────────────────────────
                ("rwkv7",        |c| Box::new(rwkv::Rwkv7::new(c))),

                // ── Mamba family ─────────────────────────────────────────────
                ("mamba2",       |c| Box::new(mamba::Mamba2::new(c))),

                // ── GLM family ───────────────────────────────────────────────
                ("glm4",         |c| Box::new(glm::Glm4::new(c))),
                ("glm4moe",      |c| Box::new(glm::Glm4::new(c))),

                // ── Granite family ───────────────────────────────────────────
                ("granitehybrid", |c| Box::new(granite::GraniteHybrid::new(c))),

                // ── Nemotron family ──────────────────────────────────────────
                ("nemotron",     |c| Box::new(nemotron::Nemotron::new(c))),
                ("nemotron_h",   |c| Box::new(nemotron::NemotronH::new(c))),

                // ── OLMo family ──────────────────────────────────────────────
                ("olmo2",        |c| Box::new(olmo::OLMo2::new(c))),

                // ── Jamba family ─────────────────────────────────────────────
                ("jamba",        |c| Box::new(jamba::Jamba::new(c))),

                // ── MiniMax family ───────────────────────────────────────────
                ("minimax-01",   |c| Box::new(minimax::MinimaxText01::new(c))),

                // ── Encoder / embedding models ───────────────────────────────
                ("modern-bert",  |c| Box::new(modern_bert::ModernBert::new(c))),
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
