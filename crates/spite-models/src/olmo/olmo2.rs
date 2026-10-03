//! Allen AI OLMo 2 — GGUF arch `olmo2`.
//!
//! Variants: OLMo 2 7B, 13B (November 2024). Fully open: weights + data + code.
//!
//! Key differences from Llama 3:
//! - **QK-Norm** (same as Qwen3): separate RMSNorm on Q and K before attention.
//! - **Post-norm inside attention** (not pre-norm): norm is applied to the
//!   attention output *before* the residual add, rather than to the input.
//!   GGUF weight names: `attn_norm` (post-attn) and `ffn_norm` (post-FFN).
//! - **z-loss regularization** during training for norm stability.
//! - SwiGLU FFN; GQA; RoPE; no bias.
//! - Tokenizer: updated GPT-NeoX BPE, 100 277 vocabulary.
//! - Fully reproducible training data via Dolmino Mix / Dolma 1.7 dataset.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct OLMo2 {
    config: ModelConfig,
}

impl OLMo2 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for OLMo2 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: modified pre/post-norm Llama-style loop:
        //   for each layer:
        //     norm_q = q_norm(Q), norm_k = k_norm(K)  // QK-Norm
        //     attn_out = gqa(norm_q, norm_k, V)
        //     x = x + attn_norm(attn_out)              // post-norm on attn output
        //     ffn_out = swiglu_ffn(x)
        //     x = x + ffn_norm(ffn_out)                // post-norm on FFN output
        Err(ModelError::Forward("not implemented".into()))
    }
}
