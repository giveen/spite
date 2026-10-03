//! HuggingFace SmolLM3 — GGUF arch `smollm3`.
//!
//! Lightweight on-device model (2025, HuggingFace).
//! 3B parameters, optimised for edge / mobile deployment.
//!
//! # NoPE (No Positional Encoding)
//!
//! SmolLM3 uses **NoPE** — no RoPE, ALiBi or learned position embeddings.
//! The model learns to handle position from data alone (like LLaMA-1 ablations
//! and RWKV linear-attn variants showed is possible for shorter contexts).
//! At inference: skip RoPE entirely; raw QK dot-products are used.
//!
//! - GQA, SwiGLU, RMSNorm
//! - Long-context fine-tuned to 32K without positional encoding tricks

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct SmolLm3 {
    config: ModelConfig,
}

impl SmolLm3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for SmolLm3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: standard GQA forward — NO RoPE application on Q/K
        Err(ModelError::Forward("not implemented".into()))
    }
}
