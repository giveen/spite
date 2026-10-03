//! Microsoft Phi-4 — GGUF arch `phi4`.
//!
//! Variant: Phi-4 14B (December 2024), Phi-4-mini 3.8B (2025).
//!
//! Key differences from Phi-3:
//! - **Phi-4 14B**: dense transformer; synthetic data–heavy pre-training pipeline;
//!   stronger reasoning benchmarks than similarly-sized models.
//! - **Phi-4-mini**: updated from Phi-3.8B; grouped-query attention; RoPE theta 250k.
//! - Full rotary (not partial as in Phi-3); no LongRoPE re-scaling at base config.
//! - Shared input/output embedding weights.
//! - `norm_eps` typically 1e-5; SwiGLU FFN.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Phi4 {
    config: ModelConfig,
}

impl Phi4 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Phi4 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: same loop structure as Llama3 (embed → layers → norm → logits)
        // No partial rotary: apply full RoPE to all head dimensions.
        Err(ModelError::Forward("not implemented".into()))
    }
}
