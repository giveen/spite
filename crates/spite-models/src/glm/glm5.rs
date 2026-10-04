//! Zhipu AI GLM-5 Next — GGUF arch `glm5-next`.
//!
//! Next-generation GLM after the previous generation.
//! Architecture details TBD once released publicly.
//!
//! Expected to retain:
//! - Causal decoder (full, not prefix-LM)
//! - RoPE, GQA
//! - Likely expands vocabulary and context window from earlier GLM
//! - May incorporate lessons from GLM-DSA sparse attention
//!
//! This stub is registered so GGUF files load without error when weights ship.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Glm5 {
    config: ModelConfig,
}

impl Glm5 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Glm5 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: implement once GLM-5 architecture is documented.
        Err(ModelError::Forward("not implemented".into()))
    }
}
