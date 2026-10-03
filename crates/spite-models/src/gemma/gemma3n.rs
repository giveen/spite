//! Google Gemma 3n — GGUF arch `gemma3n`.
//!
//! Variants: E2B (effective 2B), E4B (effective 4B). Designed for on-device
//! deployment on mobile/embedded hardware (Pixel phones, etc.).
//!
//! Key differences from Gemma 3:
//! - **Per-Layer Embeddings (PLE)**: each transformer layer has its own small
//!   embedding table for a subset of the vocabulary (high-frequency tokens).
//!   These PLE activations are *added* to the standard token embedding before
//!   the layer input, acting as a lightweight learned bias per-layer.
//!   This allows aggressive weight sharing and model compression:
//!   the base embedding table is smaller, and most of the per-token signal
//!   comes from the PLE residuals.
//! - Same local+global attention ratio as Gemma 3 (5:1, sliding window 512).
//! - GeGLU FFN, pre+post norm — same as Gemma 3.
//! - Embedding dimensions interleaved with audio/image tokens (multimodal E4B).
//! - MatMul-free attention projection option for extreme quantization.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Gemma3n {
    config: ModelConfig,
}

impl Gemma3n {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Gemma3n {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO:
        //   For each token t with id i at position pos:
        //     base_embed = embed_table[i]
        //     ple_embed  = ple_tables[layer_idx][i]  // per-layer lookup
        //     layer_input = base_embed + ple_embed
        //   Then same Gemma3 local/global attention + GeGLU FFN pipeline.
        //   PLE tables stored under GGUF key `per_layer_embed_table.<layer>`.
        Err(ModelError::Forward("not implemented".into()))
    }
}
