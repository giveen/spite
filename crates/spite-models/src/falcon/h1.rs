//! TII Falcon-H1 — GGUF arch `falcon-h1`.
//!
//! Variants: 0.5B, 1.5B, 1.8B, 7B, 34B; MoE variants.
//!
//! Falcon-H1 is a **hybrid Mamba-2 + Transformer** architecture where the two
//! block types run **in parallel per layer** (not interleaved serially).
//!
//! # Per-layer structure
//!
//! Each "hybrid block" computes both paths and sums them:
//!   ```text
//!   mamba_out = mamba2_block(rms_norm(x))
//!   attn_out  = attention_block(rms_norm(x))   // standard GQA
//!   ffn_out   = swiglu_ffn(rms_norm(x + mamba_out + attn_out))
//!   x = x + mamba_out + attn_out + ffn_out
//!   ```
//!
//! # Mamba-2 block (SSD — Structured State Space Dual)
//!
//! State dimension H, head_dim D (typically 64), expansion factor E=2.
//!   1. in_proj: x → z, x̃, B, C, dt  (all in one projection)
//!   2. conv1d over x̃ (short causal depthwise conv, width 4)
//!   3. SSD recurrence: h_{t} = A·h_{t-1} + B·x_{t},  y_{t} = C·h_{t}
//!      where A = exp(-exp(dt) * exp(A_log))  (learnable diagonal A)
//!   4. gated output: out = y * silu(z)
//!
//! # MoE variant
//!
//! Same hybrid block, but the SwiGLU FFN becomes a top-k MoE FFN.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct FalconH1 {
    config: ModelConfig,
}

impl FalconH1 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for FalconH1 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: for each hybrid block:
        //   mamba_out = mamba2_forward(norm(x))   // see mamba2 module for SSD
        //   attn_out  = gqa_forward(norm(x))       // standard RoPE+GQA
        //   x = x + mamba_out + attn_out
        //   ffn_out = swiglu_ffn(norm(x))
        //   x = x + ffn_out
        Err(ModelError::Forward("not implemented".into()))
    }
}
