//! Baidu ERNIE 4.5 — GGUF arch `ernie4_5`.
//!
//! ERNIE 4.5 dense decoder (2025).
//! - Multi-head Latent Attention (MLA) similar to DeepSeek-V3:
//!   low-rank KV compression, separate RoPE and NoPE head dims
//! - RMSNorm, SwiGLU FFN
//! - Long context up to 128K tokens

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Ernie4_5 {
    config: ModelConfig,
}

impl Ernie4_5 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Ernie4_5 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: MLA low-rank KV compression (same pattern as DeepSeek-V3 v3.rs)
        Err(ModelError::Forward("not implemented".into()))
    }
}
