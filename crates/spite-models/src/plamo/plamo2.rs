//! Preferred Networks PLaMo2 — GGUF arch `plamo2`.
//!
//! Japanese-English bilingual model from Preferred Networks (2025).
//!
//! - Dense decoder, GQA, RoPE, RMSNorm, SwiGLU
//! - Strong Japanese performance via curated Japanese-English training corpus
//! - Sizes: 1B, 8B released; larger variants planned
//! - rope_theta=500K for extended context

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct PLaMo2 {
    config: ModelConfig,
}

impl PLaMo2 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for PLaMo2 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        Err(ModelError::Forward("not implemented".into()))
    }
}
