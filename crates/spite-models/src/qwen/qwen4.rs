//! Alibaba Qwen 4 — GGUF arch `qwen4exp`.
//!
//! Experimental / early Qwen 4 weights (2025).
//! Architecture details TBD; this stub ensures the arch string is recognized.
//!
//! Likely to extend Qwen3/Qwen3.5 with:
//! - Larger vocabulary
//! - Extended context window
//! - Possibly MoE by default at larger scales
//! - QK-Norm retained from Qwen3
//!
//! Update this stub once Qwen 4 is publicly released with architecture docs.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Qwen4 {
    config: ModelConfig,
}

impl Qwen4 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Qwen4 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(
        &self,
        _tokens: &[u32],
        _logits_out: &mut [f32],
        _ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: implement once Qwen 4 architecture is documented.
        Err(ModelError::Forward("not implemented".into()))
    }
}
