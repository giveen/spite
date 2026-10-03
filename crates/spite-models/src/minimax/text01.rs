//! MiniMax-Text-01 — GGUF arch `minimax-01`.
//!
//! Variant: MiniMax-Text-01 (456B total / 45.9B active MoE, January 2025).
//!
//! Known for its **1M+ native context window** achieved via Lightning Attention.
//!
//! # Lightning Attention (Linear Attention with IO-aware tiling)
//!
//! Most layers use Lightning Attention instead of softmax attention:
//!   Q·K is replaced with a linear kernel φ(Q)·φ(K)ᵀ where φ = elu + 1.
//!   This gives O(n) complexity (vs O(n²) for softmax).
//!   "Lightning" refers to the Flash-Attention-style tiling trick that makes
//!   linear attention IO-efficient on GPUs despite the different recurrence.
//!
//! A minority of layers use standard full softmax attention for global recall.
//!
//! # MoE
//!
//! Fine-grained MoE with 32 routed experts, top-2 active per token.
//! Shared expert: 1 always-on per layer.
//!
//! # Why 1M context is practical
//!
//! Linear attention prefill is O(n): processing 1M tokens costs the same as
//! 10 × 100K, not 100 × 100K as softmax would require.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct MinimaxText01 {
    config: ModelConfig,
}

impl MinimaxText01 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for MinimaxText01 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: read layer_types from GGUF; for each layer:
        //   if "linear_attn":
        //     φ_Q = elu(Q) + 1,  φ_K = elu(K) + 1
        //     out = φ_Q · (φ_K^T · V)  // associativity: O(n) when left-to-right
        //   if "softmax_attn":
        //     standard causal attention
        //   MoE FFN: 1 shared expert + top-2 of 32 routed
        Err(ModelError::Forward("not implemented".into()))
    }
}
