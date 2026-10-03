//! LLaMA 3 / LLaMA 2 architecture.
//!
//! Weight names (GGUF convention):
//!   token_embd.weight               [vocab_size, d_model]
//!   blk.{i}.attn_norm.weight        [d_model]
//!   blk.{i}.attn_q.weight           [d_model, n_heads * head_dim]
//!   blk.{i}.attn_k.weight           [d_model, n_kv_heads * head_dim]
//!   blk.{i}.attn_v.weight           [d_model, n_kv_heads * head_dim]
//!   blk.{i}.attn_output.weight      [n_heads * head_dim, d_model]
//!   blk.{i}.ffn_norm.weight         [d_model]
//!   blk.{i}.ffn_gate.weight         [d_model, d_ffn]
//!   blk.{i}.ffn_up.weight           [d_model, d_ffn]
//!   blk.{i}.ffn_down.weight         [d_ffn, d_model]
//!   output_norm.weight              [d_model]
//!   output.weight                   [d_model, vocab_size]
//!
//! Activation: SwiGLU (gate × silu(up), projected by down)
//! Norm:       RMS norm (no bias)
//! Attention:  GQA (n_kv_heads ≤ n_heads), RoPE with theta=500000 for Llama3

use spite_abi::SpiteCtx;
use crate::{ModelArch, ModelConfig, ModelError};

pub struct Llama3 {
    config: ModelConfig,
    // TODO: weights: HashMap<String, SpiteTensor>  (populated by spite-loader)
}

impl Llama3 {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for Llama3 {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        let _cfg = &self.config;
        // TODO:
        // x = spite_compute::embed_tokens(token_embd, tokens)           [seq, d_model]
        // for i in 0..cfg.n_layers:
        //   h  = rms_norm(x, attn_norm[i])
        //   q  = h @ attn_q[i].T                                         [seq, n_heads*hd]
        //   k  = h @ attn_k[i].T                                         [seq, n_kv_heads*hd]
        //   v  = h @ attn_v[i].T                                         [seq, n_kv_heads*hd]
        //   spite_rope::apply_rope(q, positions, rope_cfg)
        //   spite_rope::apply_rope(k, positions, rope_cfg)
        //   attn_out = attention(q, k, v, kvcache[i])                    [seq, n_heads*hd]
        //   x = x + attn_out @ attn_output[i].T
        //   h  = rms_norm(x, ffn_norm[i])
        //   gate_proj = h @ ffn_gate[i].T                                [seq, d_ffn]
        //   up_proj   = h @ ffn_up[i].T                                  [seq, d_ffn]
        //   ffn_out   = (gate_proj * silu(up_proj)) @ ffn_down[i].T     [seq, d_model]
        //   x = x + ffn_out
        // x = rms_norm(x, output_norm)
        // logits_out = x @ output.T                                      [seq, vocab]
        Ok(())
    }
}
