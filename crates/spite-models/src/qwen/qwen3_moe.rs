//! Alibaba Qwen 3 MoE — GGUF arch `qwen3moe`.
//!
//! Variants: Qwen3-30B-A3B (30B total, 3B active), Qwen3-235B-A22B.
//!
//! Same QK-Norm attention as dense Qwen3, but FFN replaced with:
//! - **128 fine-grained routed experts** per layer (top-8 active per token).
//! - **No shared expert** (unlike DeepSeek which keeps 1 always-on expert).
//! - **Dense prefix layers**: first N layers are dense FFN before MoE begins.
//!   Read `qwen3moe.moe_start_layer` from GGUF metadata.
//! - Expert FFN is standard SwiGLU (gate·up·down) at a smaller d_ffn.
//! - Load balancing via auxiliary loss during training; no bias routing (inference only).

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

/// Qwen3-MoE specific fields beyond ModelConfig.
pub struct Qwen3MoeConfig {
    pub n_experts:       usize, // 128
    pub n_experts_used:  usize, // 8
    pub moe_start_layer: usize, // first layer index that uses MoE FFN
}

pub struct Qwen3Moe {
    config:     ModelConfig,
    moe_config: Qwen3MoeConfig,
}

impl Qwen3Moe {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            moe_config: Qwen3MoeConfig {
                n_experts:       128,
                n_experts_used:  8,
                moe_start_layer: 1, // TODO: read from GGUF metadata
            },
            config,
        }
    }
}

impl ModelArch for Qwen3Moe {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO per layer:
        //   if layer_idx < moe_config.moe_start_layer → dense SwiGLU FFN
        //   else:
        //     router(x) → scores [n_experts], top-k indices
        //     for each active expert: gate(x)·up(x)·down(x) with SwiGLU
        //     weighted sum of expert outputs by softmax(router_score)
        //   QK-Norm attention same as Qwen3 dense
        Err(ModelError::Forward("not implemented".into()))
    }
}
