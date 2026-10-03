//! Google Gemma 4 — GGUF arch `gemma4`.
//!
//! Variant: Gemma 4 (2025). Separate instruct variant uses arch `gemma4-assistant`.
//!
//! Builds on Gemma 3's foundation. Expected to retain:
//! - Local + global alternating attention (5:1 ratio)
//! - GeGLU FFN activation
//! - Pre- and post-norm around both sublayers
//! - Logit soft-capping (`final_logit_softcapping`)
//! - SigLIP vision encoder integration (multimodal)
//!
//! Potential additions over Gemma 3:
//! - Extended context (>32K)
//! - Updated KV head ratio (GQA n_kv_heads)
//! - Improved RoPE scaling
//!
//! Update this stub once Gemma 4 architecture details are public.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Gemma4 {
    config: ModelConfig,
}

impl Gemma4 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Gemma4 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: implement once Gemma 4 architecture is documented.
        // See gemma/gemma3.rs for the Gemma 3 forward reference.
        Err(ModelError::Forward("not implemented".into()))
    }
}
