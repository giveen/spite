//! LG AI Research ExaOne 4 MoE — GGUF arch `exaone-moe`.
//!
//! ExaOne 4 with Mixture-of-Experts FFN (Deep MoE variant).
//! Same attention as ExaOne4.
//! MoE: top-k routing, details to be filled from GGUF metadata.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct ExaOne4Moe {
    config: ModelConfig,
}

impl ExaOne4Moe {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for ExaOne4Moe {
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
