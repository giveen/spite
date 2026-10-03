//! Alibaba Qwen3-VL — GGUF arch `qwen3vl`.
//!
//! Vision-language variant of Qwen3.
//!
//! Key additions over text-only Qwen3:
//! - **M-RoPE** (Multimodal RoPE): 3D positional encoding for image patches.
//!   Image patches get (time=0, height=h, width=w) position IDs;
//!   text tokens get (time=t, height=0, width=0).
//!   The RoPE frequency tensor is split 3 ways across head_dim.
//! - Vision encoder: ViT-style patch embedder + MLP projection into d_model.
//!   Processed by `spite-vision` before this forward pass is called.
//! - Special `<image_pad>` tokens in the token stream replaced by patch embeddings.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Qwen3Vl {
    config: ModelConfig,
}

impl Qwen3Vl {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Qwen3Vl {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: same as Qwen3 dense but with M-RoPE:
        //   split each head's dim_k into 3 equal parts
        //   apply 1D RoPE to text-axis dims, 2D RoPE to spatial dims
        //   replace <image_pad> token embeddings with ViT output projections
        Err(ModelError::Forward("not implemented".into()))
    }
}
