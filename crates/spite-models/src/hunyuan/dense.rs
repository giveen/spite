//! Tencent Hunyuan Dense — GGUF arch `hunyuan-dense`.
//!
//! Tencent's dense transformer (2025).
//! - KV-Norm: RMSNorm applied to K before dot-product (stabilizes long-context)
//! - RoPE with YaRN scaling for extended context
//! - GQA, SwiGLU FFN, RMSNorm pre-norm
//! - WQA (Weight-Quantized Attention): optional INT8 KV cache path

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct HunyuanDense {
    config: ModelConfig,
}

impl HunyuanDense {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for HunyuanDense {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: standard GQA forward with KV-Norm on K vectors
        Err(ModelError::Forward("not implemented".into()))
    }
}
