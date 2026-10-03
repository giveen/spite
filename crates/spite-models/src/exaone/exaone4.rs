//! LG AI Research ExaOne 4 — GGUF arch `exaone4`.
//!
//! Dense bilingual (Korean-English) decoder, 2025.
//! - GQA, RoPE, RMSNorm, SwiGLU
//! - Long context: rope_theta=500K
//! - Sizes: 2.4B, 7.8B, 32B

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct ExaOne4 {
    config: ModelConfig,
}

impl ExaOne4 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for ExaOne4 {
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
