//! Baidu ERNIE 4.5 MoE — GGUF arch `ernie4_5-moe`.
//!
//! ERNIE 4.5 with Mixture-of-Experts FFN.
//! Same MLA attention as Ernie4_5.
//! MoE: top-k sparse routing, n shared experts.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Ernie4_5Moe {
    config: ModelConfig,
}

impl Ernie4_5Moe {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Ernie4_5Moe {
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
