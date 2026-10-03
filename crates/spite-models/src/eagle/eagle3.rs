//! EAGLE3 speculative decoding draft model — GGUF arch `eagle3`.
//!
//! EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) v3.
//! A lightweight draft model that shares the target model's embeddings and
//! predicts multiple future tokens for speculative decoding.
//!
//! # How EAGLE3 differs from the target model
//!
//! - Shallow decoder: 1–4 layers (much smaller than the target)
//! - Takes as input: target hidden states (from the last token) + token embeddings
//! - Output: logits over the target vocabulary (shared embed table, no copy)
//! - Generates a draft tree of N tokens; target verifies the whole tree in one pass
//! - EAGLE3 adds feature alignment loss (hidden-state distillation) vs EAGLE1/2
//!
//! At runtime, spite-dispatch pairs the draft model with a target model handle.
//! The draft forward generates candidates; the target forward verifies them.
//!
//! Read `eagle3.draft_layers` and `eagle3.hidden_size` from GGUF metadata.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Eagle3 {
    config: ModelConfig,
}

impl Eagle3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Eagle3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: draft forward
        //   input: concat(target_last_hidden_state, token_embed)
        //   N shallow decoder layers
        //   linear head over shared vocab embedding
        Err(ModelError::Forward("not implemented".into()))
    }
}
