//! IBM Granite SWA — GGUF arch `granite_swa`.
//!
//! Granite base with Sliding Window Attention.
//!
//! Standard Granite dense architecture but with a local sliding window
//! constraint on attention: each token can only attend to the nearest
//! `sliding_window` positions. This reduces KV cache size proportionally
//! and improves throughput on long sequences.
//!
//! Read `sliding_window` from GGUF metadata (`granite_swa.attention_window_len`).

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct GraniteSwa {
    config: ModelConfig,
}

impl GraniteSwa {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for GraniteSwa {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: standard Granite forward but attention mask is sliding window.
        // KV cache only retains the most recent `sliding_window` tokens per layer.
        Err(ModelError::Forward("not implemented".into()))
    }
}
