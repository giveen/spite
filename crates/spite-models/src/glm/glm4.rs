//! Zhipu AI GLM-4 — GGUF arch `glm4`.
//!
//! Variants: GLM-4-9B, GLM-4-32B (2024-2025).
//!
//! GLM-4 drops the earlier bidirectional prefix-LM design of ChatGLM and
//! adopts a standard causal decoder, but retains some GLM-specific quirks:
//!
//! - **RoPE** (not the original 2D position encoding of GLM-1/2).
//! - **GQA**: n_kv_heads = 2 for 9B, 8 for 32B.
//! - **SwiGLU FFN** with no bias.
//! - **Multi-token prediction** output head (optional).
//! - `<|user|>` / `<|assistant|>` / `<|observation|>` special tokens
//!   for tool-call round trips baked into tokenizer.
//! - Vocabulary: 151 329 tokens (same base as Qwen).

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Glm4 {
    config: ModelConfig,
}

impl Glm4 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Glm4 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: standard causal decoder (same loop as Llama3):
        //   embed → rms_norm → RoPE → GQA → SwiGLU FFN → rms_norm → logits
        // No per-layer post-norm; no attention softcap.
        Err(ModelError::Forward("not implemented".into()))
    }
}
