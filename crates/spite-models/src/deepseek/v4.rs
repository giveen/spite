//! DeepSeek V4 — GGUF arch `deepseek4`.
//!
//! Next-generation DeepSeek after V3/R1 (2025).
//! Architecture details are not yet fully public; this stub is registered
//! so GGUF files load without error once weights ship.
//!
//! Expected to retain/extend from V3:
//! - MLA (Multi-Head Latent Attention) with possibly higher kv_lora_rank
//! - Fine-grained MoE with ≥256 routed experts
//! - Native FP8 / MX-FP8 quantization-aware training
//! - Potentially larger expert count or MTP (Multi-Token Prediction) head
//!
//! Update this stub once the V4 architecture paper is released.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct DeepSeekV4 {
    config: ModelConfig,
}

impl DeepSeekV4 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for DeepSeekV4 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(
        &self,
        _tokens: &[u32],
        _logits_out: &mut [f32],
        _ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: implement once DeepSeek V4 architecture is documented.
        // See deepseek/v3.rs for the V3 MLA + MoE forward reference.
        Err(ModelError::Forward("not implemented".into()))
    }
}
