//! Phi-3 / Phi-3.5 architecture (Microsoft).
//!
//! Differences from LLaMA:
//!   - Combined QKV projection: single `attn_qkv.weight` instead of separate Q/K/V
//!   - Combined gate+up projection: `ffn_up.weight` contains both halves, split at runtime
//!   - Uses a partial rotary embedding: RoPE applied to the first `partial_rotary_factor`
//!     fraction of head dimensions (default 0.5 → first half rotated, second half unchanged)
//!   - Context window up to 128K with LongRoPE scaling
//!
//! Extra GGUF metadata:
//!   phi3.attention.head_count              u32
//!   phi3.rope.dimension_count              u32
//!   phi3.context_length                    u32

use spite_abi::SpiteCtx;
use crate::{ModelArch, ModelConfig, ModelError};

pub struct Phi3 {
    config:               ModelConfig,
    pub partial_rotary:   f32, // fraction of head_dim that gets RoPE, default 0.5
}

impl Phi3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config, partial_rotary: 0.5 }
    }
}

impl ModelArch for Phi3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: same as Llama3::forward but:
        //   - qkv = h @ attn_qkv.T  then split into Q, K, V along head dim
        //   - gate, up = split(h @ ffn_up.T, 2, dim=-1)
        //   - apply RoPE only to q[..partial_head_dim] and k[..partial_head_dim]
        Ok(())
    }
}
