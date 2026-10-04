//! MiniMax M3 — GGUF arch `minimax-m3`.
//!
//! Third-generation MiniMax model (2025).
//! Stub registered for forward compatibility.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct MinimaxM3 {
    config: ModelConfig,
}

impl MinimaxM3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for MinimaxM3 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(
        &self,
        _tokens: &[u32],
        _logits_out: &mut [f32],
        _ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        Err(ModelError::Forward("not implemented".into()))
    }
}
