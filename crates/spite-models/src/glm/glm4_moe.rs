//! Zhipu AI GLM-4 MoE — GGUF arch `glm4moe`.
//!
//! GLM-4 base architecture with a Mixture-of-Experts FFN replacing the
//! dense SwiGLU feed-forward layer.
//!
//! Same attention as GLM-4 (RoPE, GQA), but:
//! - MoE FFN with top-k routing (exact expert count TBD from GGUF metadata)
//! - Shared expert option (read `glm4moe.n_shared_experts` from GGUF)
//! - Designed for stronger multilingual and code performance at lower active param count

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Glm4Moe {
    config: ModelConfig,
}

impl Glm4Moe {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Glm4Moe {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: same as glm4 but replace dense SwiGLU FFN with MoE routing.
        // Read n_experts / n_experts_used from GGUF metadata.
        Err(ModelError::Forward("not implemented".into()))
    }
}
