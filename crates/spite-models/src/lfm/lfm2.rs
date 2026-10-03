//! Liquid AI LFM2 — GGUF arch `lfm2`.
//!
//! Liquid Foundation Model 2: hybrid Conv + Attention architecture (2025).
//!
//! # Architecture
//!
//! Interleaves two block types per layer:
//! - **Liquid Conv block**: causal 1D depthwise conv with gating (SSM-like).
//!   No KV cache; processes tokens via convolution state (fixed-width window).
//! - **Attention block**: standard causal GQA with RoPE and KV cache.
//!   Appears every Nth layer (read `lfm2.attention_layer_ids` from GGUF).
//!
//! Result: sub-linear memory scaling with sequence length — conv blocks use
//! O(conv_width) state vs O(seq_len) KV cache for pure-attention models.
//!
//! Sizes: 1.3B, 3.1B, 40B (all released 2025-Q2).

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Lfm2 {
    config: ModelConfig,
}

impl Lfm2 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Lfm2 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: per layer_idx:
        //   attention layers (lfm2.attention_layer_ids): GQA + KV cache
        //   conv layers: depthwise causal conv with gating, conv state only
        Err(ModelError::Forward("not implemented".into()))
    }
}
