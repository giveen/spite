//! NVIDIA Nemotron-H — GGUF arch `nemotron_h`.
//!
//! Variants: Nemotron-H 8B, 47B, 56B (2025).
//!
//! Hybrid Mamba-2 + full-attention Transformer, with most layers being Mamba-2.
//! Ratio is approximately 94% Mamba-2 and 6% full-attention layers.
//!
//! Key differences from Mamba-2 / plain Nemotron:
//! - Full-attention layers use **GQA with RoPE** (identical to Llama-3 attn).
//! - Mamba-2 layers use SSD recurrence (see `mamba2` module).
//! - Very strong long-context performance: Mamba-2 layers handle most of the
//!   local context cheaply; attention layers supply global lookup at low frequency.
//! - Layer type per index is stored in GGUF metadata as `model.layer_types`.
//! - MoE variant: `nemotron_h_moe` — same hybrid structure + top-k MoE FFN.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct NemotronH {
    config: ModelConfig,
}

impl NemotronH {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for NemotronH {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: read model.layer_types from GGUF; for each layer:
        //   if "mamba2"    → SSD recurrence (no KV cache)
        //   if "attention" → standard GQA + RoPE
        //   FFN always: SwiGLU (or MoE for nemotron_h_moe variant)
        Err(ModelError::Forward("not implemented".into()))
    }
}
