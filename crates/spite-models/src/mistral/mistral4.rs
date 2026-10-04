//! Mistral 4 / Magistral — GGUF arch `mistral4`.
//!
//! Variant: Magistral Medium (123B), Magistral Small (24B), 2025.
//!
//! Magistral is Mistral AI's reasoning/thinking model family, combining
//! long-chain-of-thought RLHF training with the Mistral architecture.
//!
//! Key notes:
//! - Dense base (no SWA, full causal attention, GQA)
//! - Trained with extended reasoning traces similar to DeepSeek-R1
//! - Streams `<think>…</think>` reasoning before the answer
//! - 128K context window; RoPE theta extended accordingly
//! - Magistral Medium: 128 attention heads; Magistral Small: same as Mistral Small 3.1
//! - Separate arch string (`mistral4`) to allow different sampling defaults
//!   (temperature, top-p, thinking budget tokens) without touching earlier
//!   generations

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Mistral4 {
    config: ModelConfig,
}

impl Mistral4 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Mistral4 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: standard GQA + SwiGLU forward (no SWA).
        // Reasoning behavior is purely a sampling/prompt concern, not architecture.
        Err(ModelError::Forward("not implemented".into()))
    }
}
