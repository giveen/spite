//! Mamba-2 (SSD) — GGUF arch `mamba2`.
//!
//! Standalone Mamba-2 models (not the hybrid variant inside Falcon-H1/Jamba).
//!
//! # SSD — Structured State Space Dual
//!
//! Mamba-2 reformulates Mamba-1's selective SSM as a special case of
//! multi-head linear attention, enabling more parallelism during training.
//!
//! Block structure:
//!   1. in_proj: x → [z, x̃, B, C, dt]  — single large projection
//!      z:   gate  [n_heads * head_dim]
//!      x̃:   input [n_heads * head_dim]  — after conv1d
//!      B,C: per-head SSM parameters [n_heads * d_state]
//!      dt:  log time-step [n_heads]
//!   2. conv1d(x̃, width=4, depthwise)
//!   3. A = -exp(A_log)  (learnable, one per head — scalar diagonal)
//!      dt = softplus(dt + dt_bias)
//!      Ā = exp(dt * A)   (discretized decay)
//!   4. SSD recurrence (or parallel chunked scan during prefill):
//!      h_t = Ā·h_{t-1} + B_t·x_t
//!      y_t = C_t·h_t
//!   5. out = (y + D·x) * silu(z)
//!   6. out_proj: out → residual
//!
//! # No KV cache
//!
//! Like RWKV, Mamba-2 decode requires only a fixed-size recurrent state
//! (n_layers × n_heads × head_dim × d_state floats). spite-kvcache does not
//! apply; implement a `MambaState` side-channel in spite-executor instead.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Mamba2 {
    config: ModelConfig,
}

impl Mamba2 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Mamba2 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO:
        //   for each Mamba-2 block:
        //     norm(x) → in_proj → [z, x_after_conv, B, C, dt]
        //     conv1d on x channel
        //     SSD step (recurrence or chunked parallel scan)
        //     out = (y + D·x) * silu(z)
        //     out_proj → residual
        Err(ModelError::Forward("not implemented".into()))
    }
}
