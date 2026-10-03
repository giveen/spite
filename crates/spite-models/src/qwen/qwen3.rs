//! Alibaba Qwen 3 (dense) — GGUF arch `qwen3`.
//!
//! Variants: 0.6B, 1.7B, 4B, 8B, 14B, 32B.
//!
//! Key differences from Llama 3:
//! - **QK-Norm**: layer-norm applied to Q and K projections *before* the
//!   dot-product attention. Stabilises training at scale.
//!   `q_norm(Q) · k_norm(K)ᵀ / sqrt(head_dim)` instead of `Q · Kᵀ`.
//! - Vocabulary ~151 865 tokens (expanded multilingual BPE).
//! - GQA with n_kv_heads = 8 (for ≥ 8B models).
//! - RoPE theta = 1 000 000 (1M), native context 32K.
//! - `<think>` token triggers chain-of-thought mode; model streams reasoning
//!   inside `<think>…</think>` before the answer.
//! - SwiGLU FFN; RMSNorm; no bias in attention projections.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Qwen3 {
    config: ModelConfig,
}

impl Qwen3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Qwen3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO per layer:
        //   rms_norm(x) → Q,K,V projections
        //   q = q_norm(Q),  k = k_norm(K)  ← QK-Norm (separate RMSNorm weights)
        //   apply RoPE(q), apply RoPE(k)
        //   grouped-query attention (n_kv_heads = 8 for large variants)
        //   SwiGLU FFN
        Err(ModelError::Forward("not implemented".into()))
    }
}
