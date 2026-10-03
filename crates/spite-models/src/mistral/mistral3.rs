//! Mistral 3 — GGUF arch `mistral3`.
//!
//! Variant: Mistral Small 3.1 (24B), March 2025.
//!
//! Key differences from Mistral base:
//! - **Vision support**: SigLIP vision encoder + MLP projection, same VLM
//!   pattern as Llama 3.2 Vision (image patch tokens prepended to text).
//! - **128K context** default; RoPE theta increased accordingly.
//! - GQA (8 KV heads on the 24B).
//! - Sliding-window attention removed (full attention on all layers).
//! - Updated tokenizer with 32K vocabulary.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Mistral3 {
    config: ModelConfig,
}

impl Mistral3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Mistral3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: standard Llama-style forward; image tokens handled upstream by
        // spite-vision before being mixed into the token stream.
        // No SWA: all layers use full causal attention.
        Err(ModelError::Forward("not implemented".into()))
    }
}
