//! Cohere Command R+ 2 MoE — GGUF arch `cohere2moe`.
//!
//! Command R 2 with Mixture-of-Experts FFN.
//! Same attention as CommandR2 (SWA + global layers).
//! MoE routing: top-2 of N experts, no shared expert.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct CommandR2Moe {
    config: ModelConfig,
}

impl CommandR2Moe {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for CommandR2Moe {
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
