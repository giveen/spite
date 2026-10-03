//! Moonshot AI Kimi K3 — GGUF arch `kimi-k3`.
//!
//! Moonshot's third-generation flagship model (2025).
//!
//! # Architecture highlights
//!
//! - MoE with top-k routing; 16 routed experts + 2 shared experts per layer
//! - MLA (Multi-head Latent Attention) for KV compression
//! - Long context: 128K tokens native, 1M with YaRN scaling
//! - **Muon optimizer** was used for training (first large-scale Muon run):
//!   this affects nothing at inference time but is notable for reproducibility
//! - RMSNorm, SwiGLU, RoPE θ=10M

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct KimiK3 {
    config: ModelConfig,
}

impl KimiK3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for KimiK3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: MLA attention (see deepseek/v3.rs for reference impl pattern)
        //   MoE FFN: top-k of n_routed_experts + n_shared_experts always active
        Err(ModelError::Forward("not implemented".into()))
    }
}
