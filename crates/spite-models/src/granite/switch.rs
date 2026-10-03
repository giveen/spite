//! IBM Granite Switch — GGUF arch `graniteswitch`.
//!
//! Granite base with Switch Transformer-style top-1 MoE routing.
//!
//! # Switch routing
//!
//! Switch Transformers (Fedus et al., 2022) use top-1 routing:
//! - Each token is sent to exactly 1 expert per layer (unlike top-k with k>1).
//! - Tokens can overflow capacity if too many are routed to the same expert;
//!   overflow tokens use a residual pass-through (identity + small scale).
//! - Expert capacity buffer: capacity_factor × (seq_len / n_experts).
//! - Very fast routing (argmax instead of topk + reduce).
//!
//! Granite Switch applies this to IBM's Granite dense base.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct GraniteSwitch {
    config: ModelConfig,
}

impl GraniteSwitch {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for GraniteSwitch {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: Granite attention (same as GraniteHybrid attention layers)
        //   MoE FFN: top-1 routing (argmax over router logits)
        //   overflow: tokens exceeding capacity_factor skip expert, get identity residual
        Err(ModelError::Forward("not implemented".into()))
    }
}
