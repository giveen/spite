//! Parse model hyperparameters from GGUF metadata.
//!
//! GGUF stores architecture configuration under arch-prefixed keys:
//!   llama.block_count                          u32  → n_layers
//!   llama.embedding_length                     u32  → d_model
//!   llama.attention.head_count                 u32  → n_heads
//!   llama.attention.head_count_kv              u32  → n_kv_heads
//!   llama.feed_forward_length                  u32  → d_ffn
//!   llama.context_length                       u32  → max_seq_len
//!   llama.rope.freq_base                       f32  → rope_theta
//!   llama.attention.layer_norm_rms_epsilon     f32  → norm_eps
//!   {arch}.attention.sliding_window            u32  → sliding_window
//!   {arch}.ssm.conv_kernel                     u32  → ssm_d_conv
//!   {arch}.ssm.inner_size                      u32  → ssm_d_inner
//!   {arch}.ssm.state_size                      u32  → ssm_d_state
//!   {arch}.ssm.time_step_rank                  u32  → ssm_dt_rank
//!   {arch}.ssm.group_count                     u32  → ssm_n_group
//!   {arch}.rope.dimension_sections             arr  → rope_sections
//!   {arch}.attention.recurrent_layers          arr  → recurrent_layers
//!   {arch}.full_attention_interval             u32  → interval fallback
//!   general.architecture                       str  → arch
//!   tokenizer.ggml.token_count                 u32  → vocab_size
//!
//! The arch prefix changes per model family (llama4, mistral4, …) but the
//! suffix conventions are shared. `ModelHyperparams::from_meta` handles
//! the lookup transparently.

use std::collections::HashMap;

/// Raw hyperparameters extracted from GGUF metadata.
///
/// These are the numbers the model architectures need to build their
/// weight-name maps and layer configurations.
#[derive(Debug, Clone)]
pub struct ModelHyperparams {
    pub arch: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub d_model: u32,
    pub d_ffn: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// Sliding-window span, if the arch uses local attention (0 = full).
    pub sliding_window: u32,
    /// Gated-delta-net geometry (0 = not a hybrid arch).
    pub ssm_d_conv: u32,
    pub ssm_d_inner: u32,
    pub ssm_d_state: u32,
    pub ssm_dt_rank: u32,
    pub ssm_n_group: u32,
    /// iRoPE dimension sections (Qwen3.5-style interleaved rope).
    pub rope_sections: [u32; 4],
    /// Per-layer recurrent flags for hybrid archs. Empty = resolve from
    /// `full_attention_interval` (every Nth layer is full attention).
    pub recurrent_layers: Vec<bool>,
    pub full_attention_interval: u32,
}

impl ModelHyperparams {
    /// Extract hyperparameters from a raw GGUF metadata map.
    ///
    /// `arch` is the value of `general.architecture` (e.g. `"llama4"`).
    /// `meta` is the full key→value map from `GgufModel`.
    ///
    /// Returns a struct with zero/default values for missing keys —
    /// callers should validate critical fields (n_layers, d_model) before use.
    pub fn from_meta(arch: &str, meta: &HashMap<String, crate::MetaValue>) -> Self {
        let u = |suffix: &str| -> u32 {
            let key = format!("{arch}.{suffix}");
            match meta.get(&key) {
                Some(crate::MetaValue::U32(v)) => *v,
                Some(crate::MetaValue::I32(v)) => *v as u32,
                Some(crate::MetaValue::U64(v)) => *v as u32,
                _ => 0,
            }
        };
        let f = |suffix: &str| -> f32 {
            let key = format!("{arch}.{suffix}");
            match meta.get(&key) {
                Some(crate::MetaValue::F32(v)) => *v,
                Some(crate::MetaValue::F64(v)) => *v as f32,
                _ => 0.0,
            }
        };
        let vocab = u("vocab_size").max(match meta.get("tokenizer.ggml.token_count") {
            Some(crate::MetaValue::U32(v)) => *v,
            _ => 0,
        });

        let rope_theta = {
            let v = f("rope.freq_base");
            if v == 0.0 { 10_000.0 } else { v }
        };
        let norm_eps = {
            let v = f("attention.layer_norm_rms_epsilon");
            if v == 0.0 { 1e-5 } else { v }
        };

        let arr4 = |suffix: &str| -> [u32; 4] {
            let key = format!("{arch}.{suffix}");
            let mut out = [0u32; 4];
            if let Some(crate::MetaValue::Array(items)) = meta.get(&key) {
                for (i, v) in items.iter().take(4).enumerate() {
                    out[i] = match v {
                        crate::MetaValue::U32(x) => *x,
                        crate::MetaValue::I32(x) => *x as u32,
                        _ => 0,
                    };
                }
            }
            out
        };
        // Recurrent-layer flags: explicit array wins, else derive from the
        // full-attention interval (every Nth layer is full attention).
        let n_layers = u("block_count");
        let recurrent_layers = {
            let key = format!("{arch}.attention.recurrent_layers");
            match meta.get(&key) {
                Some(crate::MetaValue::Array(items)) => items
                    .iter()
                    .map(|v| match v {
                        crate::MetaValue::Bool(b) => *b,
                        crate::MetaValue::U32(x) => *x != 0,
                        crate::MetaValue::I32(x) => *x != 0,
                        _ => false,
                    })
                    .collect(),
                _ => {
                    let interval = u("full_attention_interval").max(1);
                    (0..n_layers).map(|i| (i + 1) % interval != 0).collect()
                }
            }
        };

        Self {
            arch: arch.to_owned(),
            n_layers,
            n_heads: u("attention.head_count"),
            n_kv_heads: u("attention.head_count_kv"),
            d_model: u("embedding_length"),
            d_ffn: u("feed_forward_length"),
            vocab_size: vocab,
            max_seq_len: u("context_length"),
            rope_theta,
            norm_eps,
            sliding_window: u("attention.sliding_window"),
            ssm_d_conv: u("ssm.conv_kernel"),
            ssm_d_inner: u("ssm.inner_size"),
            ssm_d_state: u("ssm.state_size"),
            ssm_dt_rank: u("ssm.time_step_rank"),
            ssm_n_group: u("ssm.group_count"),
            rope_sections: arr4("rope.dimension_sections"),
            recurrent_layers,
            full_attention_interval: u("full_attention_interval"),
        }
    }

    /// Convenience: extract hyperparams directly from an open `GgufModel`.
    pub fn from_gguf(model: &crate::GgufModel) -> Self {
        Self::from_meta(model.arch(), &model.meta)
    }
}
