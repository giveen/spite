//! IBM Granite Hybrid — GGUF arch `granitehybrid`.
//!
//! Variant: IBM Granite 4.0 Tiny (and larger siblings, 2025).
//!
//! A Mamba-2 + sparse-Transformer hybrid in a **9:1 ratio**:
//! - Every 10th layer is a sparse Transformer attention layer (full GQA).
//! - The other 9 layers are Mamba-2 SSD blocks.
//! - MoE variant available (`graniteswitch`) using Switch-style top-1 routing.
//!
//! IBM reports 3.3× higher decode throughput vs Qwen3-30B-A3B at similar quality.
//! This is because 9 out of 10 layers require no KV cache and run in O(1) memory.
//!
//! # Attention layers
//!
//! Sparse "attention" layers use GQA with a *limited* context window; they do
//! NOT attend to the full prefix. This controls memory without SWA sliding.
//!
//! # Mamba-2 layers
//!
//! Identical SSD recurrence to `mamba2` module — see that module for detail.
//! State is maintained in a `GraniteState` side-channel (not KV cache).

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct GraniteHybrid {
    config: ModelConfig,
}

impl GraniteHybrid {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for GraniteHybrid {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: check layer type from GGUF metadata (mamba2 vs attn):
        //   if layer is mamba2 → SSD recurrence (see mamba2 module)
        //   if layer is attn   → standard GQA (limited window)
        //   alternating 9:1 based on `model.layer_types` in GGUF metadata
        Err(ModelError::Forward("not implemented".into()))
    }
}
