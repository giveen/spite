//! Google Gemma 3 — GGUF arch `gemma3`.
//!
//! Variants: 1B, 4B, 12B, 27B (March 2025).
//!
//! Key differences from Llama 3:
//! - **Local + Global attention** in a 5:1 ratio: 5 sliding-window local
//!   attention layers for every 1 full global attention layer.
//!   Local layers use `sliding_window = 1024` (configurable per model size).
//! - **GeGLU FFN** activation: `GELU(gate) ⊙ up` instead of SwiGLU.
//!   `gelu_pytorch_tanh` approximation is standard.
//! - **Pre- and post-norm** around both attn and FFN sublayers (4 norm calls
//!   per layer vs 2 in Llama). Weights: `pre_feedforward_layernorm`,
//!   `post_feedforward_layernorm`, `pre_attn_layernorm`, `post_attn_layernorm`.
//! - GQA with n_kv_heads = 4 (small models) or 8 (27B).
//! - RoPE theta = 10 000 (inherited from Gemma 2).
//! - Logit soft-capping: `tanh(logits / 30.0) * 30.0` before sampling.
//!   Prevents logit explosion; cap value is `final_logit_softcapping` in GGUF.
//! - Multimodal: SigLIP vision encoder + image token injection (4B+).
//! - Tied input/output embedding weights.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct Gemma3Config {
    pub sliding_window:          usize,  // local attention window (e.g. 1024)
    pub global_attn_every_n:     usize,  // 1 global layer per N total (default 6)
    pub final_logit_softcap:     f32,    // typically 30.0, 0 = disabled
    pub attn_logit_softcap:      f32,    // per-layer attn cap, typically 50.0
}

pub struct Gemma3 {
    config:     ModelConfig,
    g3_config:  Gemma3Config,
}

impl Gemma3 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            g3_config: Gemma3Config {
                sliding_window:      1024,
                global_attn_every_n: 6,
                final_logit_softcap: 30.0,
                attn_logit_softcap:  50.0,
            },
            config,
        }
    }
}

impl ModelArch for Gemma3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO per layer:
        //   pre_attn_norm(x) → Q,K,V → RoPE(Q), RoPE(K)
        //   if layer_idx % global_attn_every_n != 0:
        //     local attention: mask positions outside sliding_window
        //     attn_softcap: scores = tanh(scores / attn_logit_softcap) * attn_logit_softcap
        //   else:
        //     global full causal attention (same softcap)
        //   post_attn_norm(attn_out) + x
        //   pre_ffn_norm(x) → GeGLU: GELU(gate·W_gate) ⊙ (x·W_up) → W_down
        //   post_ffn_norm(ffn_out) + x
        //
        // Final: logits = tanh(logits / final_logit_softcap) * final_logit_softcap
        Err(ModelError::Forward("not implemented".into()))
    }
}
