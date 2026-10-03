//! NVIDIA Nemotron — GGUF arch `nemotron`.
//!
//! Variants: Minitron 4B/8B (pruned from Llama-3), Nemotron-4-340B,
//! Nemotron-70B (2024).
//!
//! Nemotron models are produced via **structured pruning + continued pre-training
//! + knowledge distillation**:
//! - Width pruning: remove attention heads and FFN channels.
//! - Depth pruning: remove entire layers.
//! - Knowledge distillation: student (pruned) trained against teacher logits.
//!
//! Architecture after pruning:
//! - **SquaredReLU** activation instead of SiLU/GELU: `(relu(x))²`.
//!   Produces sparser activations which aids distillation quality.
//! - Standard GQA (may differ per size from base model).
//! - RoPE; RMSNorm; no bias.
//! - Logit soft-capping at final layer (Nemotron-70B): same as Gemma 3,
//!   `tanh(logits / cap) * cap`, cap = 30.0.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Nemotron {
    config: ModelConfig,
}

impl Nemotron {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Nemotron {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: standard Llama-style loop but with SquaredReLU FFN:
        //   ffn_out = (relu(W_gate · x))² ⊙ W_up · x → W_down
        //   (no gating; SquaredReLU replaces the full SwiGLU gate structure)
        // Apply final logit softcap if `final_logit_softcapping` is in GGUF metadata.
        Err(ModelError::Forward("not implemented".into()))
    }
}
