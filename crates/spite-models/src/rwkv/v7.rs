//! RWKV-7 (Eagle) — GGUF arch `rwkv7`.
//!
//! Variants: 0.1B, 0.4B, 1.5B, 2.9B, 7.3B+.
//!
//! RWKV-7 is a **pure RNN** — no attention, O(n) time, O(1) per-step memory.
//! It is a token-shifted gated state-space model.
//!
//! # Per-layer recurrence (simplified)
//!
//! State: h ∈ R^{d_state × d_head}  (matrix-valued hidden state)
//!
//! For each token x_t:
//!   receptance r = sigmoid(W_r · lerp(x_{t-1}, x_t))
//!   key        k = W_k · lerp(x_{t-1}, x_t)
//!   value      v = W_v · lerp(x_{t-1}, x_t)
//!   gate       g = silu(W_g · lerp(x_{t-1}, x_t))
//!   decay      w = exp(-exp(W_w · lerp(x_{t-1}, x_t)))  // dynamic!
//!   bonus      u = W_u                                    // per-channel scalar
//!
//!   h_t = diag(w) · h_{t-1} + outer(k, v)       // state update
//!   y_t = h_t · r   (matrix-vector)
//!   out_t = group_norm(y_t) * g + residual
//!
//! The key RWKV-7 innovation is **dynamic per-token decay** (w depends on x_t),
//! plus a **matrix-valued** hidden state (head_dim² elements per head) giving
//! much higher capacity than RWKV-5/6's lower-rank state.
//!
//! # Inference note
//!
//! Prefill can use parallel scan for training speed; decode uses the recurrence
//! directly — no KV cache needed, constant memory regardless of sequence length.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Rwkv7 {
    config: ModelConfig,
}

impl Rwkv7 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Rwkv7 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: for each RWKV-7 block:
        //   time_mixing(x)  — the recurrence described above
        //   channel_mixing(x) — simpler gated linear unit for FFN equivalent
        // State must be passed in ctx or stored in a per-sequence side-channel
        // (not in spite-kvcache — RWKV has no KV cache).
        Err(ModelError::Forward("not implemented".into()))
    }
}
