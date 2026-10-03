//! Alibaba QwQ-32B — GGUF arch `qwen3next`.
//!
//! Variant: QwQ-32B (reasoning/thinking model, March 2025).
//!
//! Architecture is a Qwen3-32B dense base with an extended context window
//! and reasoning-mode training. Same QK-Norm attention.
//!
//! Key notes:
//! - Identical forward pass to `qwen3` — the arch string differs only
//!   to allow different sampling defaults (longer context, lower temperature).
//! - RoPE theta = 40 000 000 (40M) for extended sequence lengths.
//! - Trained with RLHF to stream `<think>…</think>` reasoning chains.
//! - 32K context for short tasks; extendable via YaRN to 128K.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct QwQ {
    config: ModelConfig,
}

impl QwQ {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for QwQ {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: identical forward path to qwen3 (QK-Norm + GQA + SwiGLU).
        // Only differences: rope_theta = 40_000_000, head_dim = 128, n_kv_heads = 8.
        Err(ModelError::Forward("not implemented".into()))
    }
}
