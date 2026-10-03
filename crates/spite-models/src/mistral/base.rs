//! Mistral / Mixtral architecture.
//!
//! Differences from LLaMA:
//!   - Sliding window attention (SWA): each layer attends only to the last
//!     `sliding_window` tokens (default 4096). Layers without SWA attend globally.
//!   - GQA with n_kv_heads typically 8
//!   - Mixtral adds MoE (Mixture of Experts) FFN: `num_experts` FFN blocks,
//!     `num_experts_per_tok` selected per token via a router
//!
//! Extra GGUF metadata:
//!   mistral.attention.sliding_window  u32
//!   mistral.expert_count              u32   (Mixtral only)
//!   mistral.expert_used_count         u32   (Mixtral only)

use spite_abi::SpiteCtx;
use crate::{ModelArch, ModelConfig, ModelError};

pub struct Mistral {
    config: ModelConfig,
    pub sliding_window: Option<usize>, // None → global attention like LLaMA
    pub n_experts:      usize,         // 1 for base Mistral, 8 for Mixtral-8x7B
    pub n_experts_used: usize,         // 2 for Mixtral (top-2 routing)
}

impl Mistral {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            sliding_window: Some(4096),
            n_experts:      1,
            n_experts_used: 1,
        }
    }
}

impl ModelArch for Mistral {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: same as Llama3::forward but:
        //   - pass sliding_window mask to attention kernel when Some
        //   - when n_experts > 1: run router → top-k expert selection
        //     → route tokens to selected expert FFNs → weighted sum
        Ok(())
    }
}
