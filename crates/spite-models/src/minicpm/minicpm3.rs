//! ModelBest MiniCPM3 — GGUF arch `minicpm3`.
//!
//! MiniCPM3: efficient small model with Dynamic Compression Attention (2025).
//!
//! # Dynamic Compression Attention (DCA)
//!
//! DCA is a KV cache compression technique:
//! - Queries attend to a mix of full-resolution recent tokens and
//!   compressed distant tokens (average-pooled KV groups).
//! - Two KV cache tiers: recent (full-res) + compressed (pooled).
//! - Compression window: `compress_ratio` tokens → 1 compressed KV.
//! - Read `minicpm3.kv_compress_ratio` and `minicpm3.compress_window` from GGUF.
//!
//! Result: sub-linear KV cache growth while retaining full attention on recent tokens.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct MiniCpm3 {
    config: ModelConfig,
}

impl MiniCpm3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for MiniCpm3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: DCA two-tier KV cache
        //   tier-1: last compress_window tokens at full resolution
        //   tier-2: earlier tokens average-pooled by compress_ratio groups
        //   Q attends to both tiers in one attention op
        Err(ModelError::Forward("not implemented".into()))
    }
}
