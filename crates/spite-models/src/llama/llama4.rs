//! Meta Llama 4 — GGUF arch `llama4`.
//!
//! Variants: Scout (17B active / 109B total), Maverick (17B active / 400B total).
//!
//! Key differences from Llama 3:
//! - **iRoPE**: interleaved layers — even layers have full RoPE, odd layers have
//!   NoPE (No Positional Encoding). NoPE layers omit all rotary application.
//! - **Native MoE**: 16 routed experts per layer, top-1 active per token.
//! - **Early-exit routing**: router can skip MoE entirely for some tokens.
//! - **GQA** with n_kv_heads < n_heads.
//! - Multimodal vision encoder integrated (text-only path uses this arch string).
//! - Chunked prefill friendly: each NoPE layer is position-agnostic.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Llama4 {
    config: ModelConfig,
}

impl Llama4 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Llama4 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO (per layer):
        //   if layer_idx % 2 == 0 → apply RoPE to Q,K (full RoPE, standard theta)
        //   else → NoPE: skip RoPE, Q,K unrotated
        //   MoE FFN: router → top-1 expert → (gate·up)·down with SwiGLU per expert
        //   early-exit: if router confidence > threshold, skip remaining experts
        Err(ModelError::Forward("not implemented".into()))
    }
}
