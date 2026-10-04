//! Alibaba Qwen 3.5 — GGUF arch `qwen35`.
//!
//! Successor to Qwen3; expected mid-2025.
//! Dense and MoE variants (`qwen35` / `qwen35moe`).
//!
//! Architecture details will be filled in once released. Expected to retain:
//! - QK-Norm from Qwen3
//! - Extended context window (≥ 128K)
//! - Updated pre-training data recipe
//! - Larger vocabulary (potentially 200K+)
//!
//! This stub is registered so GGUF files load without error once weights ship.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Qwen3_5 {
    config: ModelConfig,
}

impl Qwen3_5 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Qwen3_5 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(
        &self,
        _tokens: &[u32],
        _logits_out: &mut [f32],
        _ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: implement when Qwen 3.5 architecture is publicly documented.
        Err(ModelError::Forward("not implemented".into()))
    }
}
