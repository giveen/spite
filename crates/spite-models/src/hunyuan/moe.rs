//! Tencent Hunyuan MoE — GGUF arch `hunyuan-moe`.
//!
//! Tencent's Mixture-of-Experts transformer (2025).
//! HunyuanDense attention (KV-Norm, RoPE YaRN, GQA) with MoE FFN:
//! - Top-k routing (k=2) from n_experts
//! - 1 shared expert always active
//! - Expert capacity buffer per Switch Transformers convention

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct HunyuanMoe {
    config: ModelConfig,
}

impl HunyuanMoe {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for HunyuanMoe {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        Err(ModelError::Forward("not implemented".into()))
    }
}
