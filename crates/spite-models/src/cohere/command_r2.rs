//! Cohere Command R+ 2 — GGUF arch `cohere2`.
//!
//! Command R 2025 refresh. Dense decoder with:
//! - Sliding window local attention (window=4096) with full global layers every 8th
//! - RoPE θ=8M, GQA (n_kv_heads=8)
//! - Logit softcapping (final_logit_softcap=30.0 like Gemma)
//! - SwiGLU FFN, RMSNorm
//!
//! Read `sliding_window` from GGUF: `cohere2.attention_window_len`.
//! Global attention layers: indices where (layer_idx % 8 == 7).

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct CommandR2 {
    config: ModelConfig,
}

impl CommandR2 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for CommandR2 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: alternating local SWA / global attention
        //   local: causal mask limited to sliding_window positions
        //   global (every 8th): full causal mask, same KV cache slot
        //   final logit softcap: tanh(logits / 30.0) * 30.0
        Err(ModelError::Forward("not implemented".into()))
    }
}
