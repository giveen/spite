//! JetBrains Mellum — GGUF arch `mellum`.
//!
//! JetBrains' Fill-in-the-Middle (FIM) code completion model (2025).
//! Optimised for IDE inline completion (prefix + suffix → middle).
//!
//! # FIM format
//!
//! Uses sentinel tokens for FIM:
//! - `<|fim_prefix|>` — precedes the code before the cursor
//! - `<|fim_suffix|>` — precedes the code after the cursor
//! - `<|fim_middle|>` — model generates the completion here
//!
//! At inference: assemble prompt as:
//!   `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>`
//! then sample until `<|fim_pad|>` or `<|endoftext|>`.
//!
//! # Architecture
//!
//! - Dense decoder (Llama-style): GQA, RoPE, RMSNorm, SwiGLU
//! - Sizes: 1B, 9B (2025)
//! - Trained on multi-repo code (JetBrains internal crawl + permissive OSS)

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Mellum {
    config: ModelConfig,
}

impl Mellum {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Mellum {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(
        &self,
        _tokens: &[u32],
        _logits_out: &mut [f32],
        _ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: standard Llama-style dense forward
        //   FIM token handling is at the tokenizer / prompt assembly layer,
        //   not in the forward pass itself.
        Err(ModelError::Forward("not implemented".into()))
    }
}
