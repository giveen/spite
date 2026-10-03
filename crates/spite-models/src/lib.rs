//! Model architecture implementations.
//!
//! Each architecture is a separate module with its own weight-name mapping,
//! layer ordering, and hyperparameter interpretation. The host forward-pass
//! loop lives here; GPU kernels live under kernels/ and are called via
//! spite-dispatch.
//!
//! Adding a new architecture
//! -------------------------
//! 1. Create `src/<arch>.rs` implementing `ModelArch`
//! 2. Register it in `ArchRegistry::default()`
//! 3. Add a `kernels/<arch>/` directory for GPU-specific ops

pub mod llama3;
pub mod mistral;
pub mod phi3;

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
                ("llama",   |c| Box::new(llama3::Llama3::new(c))),
                ("llama3",  |c| Box::new(llama3::Llama3::new(c))),
                ("mistral", |c| Box::new(mistral::Mistral::new(c))),
                ("phi3",    |c| Box::new(phi3::Phi3::new(c))),
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
