//! Liquid AI LFM2 MoE — GGUF arch `lfm2moe`.
//!
//! LFM2 with Mixture-of-Experts in the FFN position of attention blocks.
//! Conv blocks are unchanged (dense FFN inside the conv layer).

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Lfm2Moe {
    config: ModelConfig,
}

impl Lfm2Moe {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Lfm2Moe {
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
