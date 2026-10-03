//! Zhipu AI GLM with Dynamic Sparse Attention — GGUF arch `glm-dsa`.
//!
//! GLM base with Dynamic Sparse Attention (DSA): the attention pattern is
//! computed dynamically per-layer and per-head at runtime rather than using
//! a fixed local/global split.
//!
//! # How DSA works
//!
//! A lightweight router scores each query against all keys and selects only
//! the top-k most relevant key positions to attend to. This gives:
//! - O(n·k) attention cost instead of O(n²), k << n
//! - Adaptive sparsity: different positions attend to different context regions
//! - Trained with a differentiable top-k approximation (straight-through or Gumbel)
//!
//! The sparsity pattern is not cached between steps — it must be recomputed
//! from scratch each decode step from the current query.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct GlmDsa {
    config: ModelConfig,
}

impl GlmDsa {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for GlmDsa {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: for each attention layer:
        //   scores = Q · K^T / sqrt(head_dim)    // full pairwise for routing
        //   mask = top_k(scores, k=sparse_k)      // only top-k positions
        //   out  = softmax(scores * mask) · V
        Err(ModelError::Forward("not implemented".into()))
    }
}
