//! Allen AI OLMoE — GGUF arch `olmoe`.
//!
//! Variant: OLMoE-1B-7B (1B active parameters from a 7B total MoE).
//! The first fully-open MoE LLM: open weights, open training data (Dolma), open code.
//!
//! Key differences from OLMo 2 (dense):
//! - **MoE FFN**: 64 experts per layer, top-8 active (top8 of 64).
//!   Each expert is a small SwiGLU FFN (d_ffn / n_experts sized).
//! - Same QK-Norm as OLMo 2 (separate RMSNorm on Q and K).
//! - Same post-norm residual structure as OLMo 2.
//! - GQA; RoPE.
//! - Expert parallelism designed for multi-GPU inference.
//! - Load-balancing auxiliary loss during training; no bias routing at inference.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct OLMoE {
    config: ModelConfig,
}

impl OLMoE {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for OLMoE {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: same as OLMo2 attention (QK-Norm + post-norm), but FFN is MoE:
        //   router_scores = router(x) → top-8 indices from 64 experts
        //   out = sum over top-8 experts: softmax_score_i * swiglu_ffn_i(x)
        Err(ModelError::Forward("not implemented".into()))
    }
}
