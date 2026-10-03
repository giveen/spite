//! AI21 Labs Jamba — GGUF arch `jamba`.
//!
//! Variants: Jamba 1.5 Mini (12B active / 52B total), Jamba 1.5 Large
//! (94B active / 398B total), Jamba 2 (2025).
//!
//! Jamba pioneered **large-scale Transformer + Mamba + MoE** hybrid at
//! production scale (competing with GPT-4 class models).
//!
//! # Layer structure
//!
//! Jamba interleaves blocks in a fixed pattern (read from GGUF metadata):
//! - **Mamba blocks**: Mamba-1 recurrence (SSM). No attention, no KV cache.
//! - **Attention blocks**: full causal GQA.
//! - **MoE blocks**: occur at certain attention and Mamba layers simultaneously.
//!   Top-2 of 16 experts active (Jamba 1.5 Mini config).
//!
//! Typical interleaving for Jamba 1.5 Mini: MAMMA MAMMA MAMMA AT MAMMA AT ...
//! (exact ratio in GGUF `jamba.attention_layer_step` metadata).
//!
//! # Mamba-1 recurrence
//!
//! Simpler than Mamba-2 (no SSD, no parallel chunked scan):
//!   h_t = Ā·h_{t-1} + B_t·x_t
//!   y_t = C_t·h_t + D·x_t
//! where Ā = exp(dt * A), A is diagonal and input-dependent in selective mode.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Jamba {
    config: ModelConfig,
}

impl Jamba {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Jamba {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: read layer_types from GGUF metadata; for each block:
        //   if "mamba"     → Mamba-1 recurrence (maintains separate mamba state)
        //   if "attention" → standard GQA
        //   if moe variant → MoE FFN (top-2 of 16 experts)
        //   else           → dense SwiGLU FFN
        Err(ModelError::Forward("not implemented".into()))
    }
}
