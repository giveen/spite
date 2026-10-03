//! MiniMax M2 — GGUF arch `minimax-m2`.
//!
//! Successor to MiniMax-Text-01 (2025).
//! Architecture details will be filled in once released.
//!
//! Expected to retain Lightning Attention hybrid approach from text01
//! (linear attention for most layers, softmax for a minority),
//! with improvements to the MoE routing and expert capacity.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct MinimaxM2 {
    config: ModelConfig,
}

impl MinimaxM2 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for MinimaxM2 {
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
