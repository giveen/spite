//! DeepSeek-V3 / R1 — GGUF arch `deepseek2`.
//!
//! Variants: DeepSeek-V3 (Dec 2024), DeepSeek-R1 (Jan 2025), R1-0528.
//! Also covers DeepSeek-V2 (May 2024) which shares the arch string.
//!
//! This is the most architecturally novel model family in the 2024-2025 wave.
//!
//! # MLA — Multi-Head Latent Attention
//!
//! Compresses the KV cache from O(n_layers × seq × n_kv_heads × head_dim)
//! to O(n_layers × seq × kv_lora_rank) by projecting into a low-rank latent.
//!
//! Forward:
//!   c_kv = down_proj(x)                            // [seq, kv_lora_rank]
//!   k_rope, v_rope = rope_proj(c_kv)               // positional component
//!   k_nope = k_proj(c_kv)                          // content, no position
//!   K = concat(k_nope, k_rope) along head_dim
//!   // Q also has a low-rank structure: q = up_proj(down_q_proj(x))
//!
//! # Fine-grained MoE
//!
//! - 256 routed experts per layer, top-8 active.
//! - 1 always-on shared expert per layer (unlike Qwen3-MoE).
//! - Auxiliary-loss-free load balancing via learned bias terms on router logits.
//! - No token dropping; all tokens routed to exactly top-8.
//!
//! # FP8 training
//!
//! Weights may be quantized to FP8 E4M3; GGUF stores them as Q8_0 or in
//! the native FP8 block format. Check `general.file_type` in metadata.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct DeepSeekV3Config {
    pub kv_lora_rank:    usize, // typically 512
    pub q_lora_rank:     usize, // typically 1536
    pub qk_rope_head_dim:  usize, // typically 64
    pub qk_nope_head_dim:  usize, // typically 128
    pub v_head_dim:        usize, // typically 128
    pub n_experts:         usize, // 256
    pub n_experts_used:    usize, // 8
    pub n_shared_experts:  usize, // 1
    pub moe_start_layer:   usize, // typically 3
}

pub struct DeepSeekV3 {
    config:     ModelConfig,
    ds_config:  DeepSeekV3Config,
}

impl DeepSeekV3 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            ds_config: DeepSeekV3Config {
                kv_lora_rank:      512,
                q_lora_rank:       1536,
                qk_rope_head_dim:  64,
                qk_nope_head_dim:  128,
                v_head_dim:        128,
                n_experts:         256,
                n_experts_used:    8,
                n_shared_experts:  1,
                moe_start_layer:   3,
            },
            config,
        }
    }
}

impl ModelArch for DeepSeekV3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO — MLA attention per layer:
        //   1. c_kv = W_dkv · rms_norm(x)          // compress to kv_lora_rank
        //   2. k_nope, v = W_ukv · c_kv             // up-project
        //   3. k_rope = W_kr · c_kv; apply RoPE     // positional component
        //   4. K = concat(k_nope, k_rope); use V directly
        //   5. Q: c_q = W_dq · x; Q = W_uq · c_q   // Q also low-rank
        //   6. q_rope portion: apply RoPE; concat with q_nope
        //   7. attention as normal (n_heads = 128 for V3)
        //
        // TODO — MoE FFN per layer (after moe_start_layer):
        //   shared_out = shared_expert(x)
        //   router_scores = topk(router(x), k=n_experts_used)
        //   routed_out = sum over top-k: score_i * expert_i(x)
        //   ffn_out = shared_out + routed_out
        Err(ModelError::Forward("not implemented".into()))
    }
}
