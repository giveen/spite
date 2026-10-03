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

// Additional 2025 families
pub mod cohere;
pub mod exaone;
pub mod hunyuan;
pub mod ernie;
pub mod lfm;
pub mod kimi;
pub mod smollm;
pub mod minicpm;
pub mod plamo;
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
                ("mistral4",     |c| Box::new(mistral::Mistral4::new(c))),
                ("magistral",    |c| Box::new(mistral::Mistral4::new(c))),

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
                ("qwen4",        |c| Box::new(qwen::Qwen4::new(c))),
                ("qwen4exp",     |c| Box::new(qwen::Qwen4::new(c))),

                // ── DeepSeek family ──────────────────────────────────────────
                ("deepseek2",    |c| Box::new(deepseek::DeepSeekV3::new(c))),
                ("deepseek32",   |c| Box::new(deepseek::DeepSeekV3::new(c))),
                ("deepseek4",    |c| Box::new(deepseek::DeepSeekV4::new(c))),

                // ── Gemma family ─────────────────────────────────────────────
                ("gemma3",       |c| Box::new(gemma::Gemma3::new(c))),
                ("gemma3n",      |c| Box::new(gemma::Gemma3n::new(c))),
                ("gemma4",       |c| Box::new(gemma::Gemma4::new(c))),

                // ── Falcon family ────────────────────────────────────────────
                ("falcon-h1",    |c| Box::new(falcon::FalconH1::new(c))),

                // ── RWKV family ──────────────────────────────────────────────
                ("rwkv7",        |c| Box::new(rwkv::Rwkv7::new(c))),
                ("arwkv7",       |c| Box::new(rwkv::ARwkv7::new(c))),

                // ── Mamba family ─────────────────────────────────────────────
                ("mamba2",       |c| Box::new(mamba::Mamba2::new(c))),

                // ── GLM family ───────────────────────────────────────────────
                ("glm4",         |c| Box::new(glm::Glm4::new(c))),
                ("glm4moe",      |c| Box::new(glm::Glm4Moe::new(c))),
                ("glm-dsa",      |c| Box::new(glm::GlmDsa::new(c))),
                ("glm5",         |c| Box::new(glm::Glm5::new(c))),
                ("glm5-next",    |c| Box::new(glm::Glm5::new(c))),

                // ── Granite family ───────────────────────────────────────────
                ("granitehybrid",  |c| Box::new(granite::GraniteHybrid::new(c))),
                ("graniteswitch",  |c| Box::new(granite::GraniteSwitch::new(c))),
                ("granite_swa",    |c| Box::new(granite::GraniteSwa::new(c))),

                // ── Nemotron family ──────────────────────────────────────────
                ("nemotron",     |c| Box::new(nemotron::Nemotron::new(c))),
                ("nemotron_h",   |c| Box::new(nemotron::NemotronH::new(c))),

                // ── OLMo family ──────────────────────────────────────────────
                ("olmo2",        |c| Box::new(olmo::OLMo2::new(c))),
                ("olmoe",        |c| Box::new(olmo::OLMoE::new(c))),

                // ── Jamba family ─────────────────────────────────────────────
                ("jamba",        |c| Box::new(jamba::Jamba::new(c))),

                // ── MiniMax family ───────────────────────────────────────────
                ("minimax-01",   |c| Box::new(minimax::MinimaxText01::new(c))),
                ("minimax-m2",   |c| Box::new(minimax::MinimaxM2::new(c))),
                ("minimax-m3",   |c| Box::new(minimax::MinimaxM3::new(c))),

                // ── Encoder / embedding models ───────────────────────────────
                ("modern-bert",  |c| Box::new(modern_bert::ModernBert::new(c))),

                // ── Cohere family ────────────────────────────────────────────
                ("cohere2",      |c| Box::new(cohere::CommandR2::new(c))),
                ("cohere2moe",   |c| Box::new(cohere::CommandR2Moe::new(c))),

                // ── ExaOne family ────────────────────────────────────────────
                ("exaone4",      |c| Box::new(exaone::ExaOne4::new(c))),
                ("exaone-moe",   |c| Box::new(exaone::ExaOne4Moe::new(c))),

                // ── Hunyuan family ───────────────────────────────────────────
                ("hunyuan-dense", |c| Box::new(hunyuan::HunyuanDense::new(c))),
                ("hunyuan-moe",   |c| Box::new(hunyuan::HunyuanMoe::new(c))),

                // ── ERNIE family ─────────────────────────────────────────────
                ("ernie4_5",     |c| Box::new(ernie::Ernie4_5::new(c))),
                ("ernie4_5-moe", |c| Box::new(ernie::Ernie4_5Moe::new(c))),

                // ── LFM family ───────────────────────────────────────────────
                ("lfm2",         |c| Box::new(lfm::Lfm2::new(c))),
                ("lfm2moe",      |c| Box::new(lfm::Lfm2Moe::new(c))),

                // ── Kimi family ──────────────────────────────────────────────
                ("kimi-k3",      |c| Box::new(kimi::KimiK3::new(c))),

                // ── SmolLM family ────────────────────────────────────────────
                ("smollm3",      |c| Box::new(smollm::SmolLm3::new(c))),

                // ── MiniCPM family ───────────────────────────────────────────
                ("minicpm3",     |c| Box::new(minicpm::MiniCpm3::new(c))),

                // ── PLaMo family ─────────────────────────────────────────────
                ("plamo2",       |c| Box::new(plamo::PLaMo2::new(c))),
                ("plamo3",       |c| Box::new(plamo::PLaMo2::new(c))),

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
