//! ARWKV-7 — GGUF arch `arwkv7`.
//!
//! Attention-augmented RWKV-7: the pure-RNN RWKV-7 backbone with a small
//! number of full softmax attention layers inserted at specific positions
//! for global context lookup.
//!
//! # Motivation
//!
//! RWKV-7's recurrence is excellent at local pattern recognition but may
//! miss long-range dependencies that attention handles well. ARWKV-7 adds
//! attention at every Nth layer (ratio configurable in GGUF metadata).
//!
//! # Per-layer routing
//!
//! Read `arwkv7.attn_layer_ids` from GGUF metadata — a list of layer indices
//! that use standard softmax GQA instead of RWKV-7 time_mixing.
//!
//! Attention layers in ARWKV-7:
//! - Use a standard KV cache (unlike RWKV-7 which uses recurrent state)
//! - Full causal mask, RoPE, no sliding window
//! - The mixed model requires BOTH a KV cache AND an RWKV recurrent state

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct ARwkv7 {
    config: ModelConfig,
}

impl ARwkv7 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for ARwkv7 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: for each layer:
        //   if layer_idx in attn_layer_ids → standard GQA (KV cache)
        //   else → RWKV-7 time_mixing recurrence (no KV cache, recurrent state)
        //   all layers: RWKV-7 channel_mixing for FFN
        Err(ModelError::Forward("not implemented".into()))
    }
}
