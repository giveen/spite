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
    pub arch:        String,
    pub n_layers:    u32,
    pub n_heads:     u32,
    pub n_kv_heads:  u32,
    pub d_model:     u32,
    pub d_ffn:       u32,
    pub vocab_size:  u32,
    pub max_seq_len: u32,
    pub rope_theta:  f32,
    pub norm_eps:    f32,
}

impl ModelHyperparams {
    /// Extract hyperparameters from a raw GGUF metadata map.
    ///
    /// `arch` is the value of `general.architecture` (e.g. `"llama"`).
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
        let vocab = u("vocab_size")
            .max(match meta.get("tokenizer.ggml.token_count") {
                Some(crate::MetaValue::U32(v)) => *v,
                _ => 0,
            });

        let rope_theta = { let v = f("rope.freq_base"); if v == 0.0 { 10_000.0 } else { v } };
        let norm_eps   = { let v = f("attention.layer_norm_rms_epsilon"); if v == 0.0 { 1e-5 } else { v } };

        Self {
            arch:        arch.to_owned(),
            n_layers:    u("block_count"),
            n_heads:     u("attention.head_count"),
            n_kv_heads:  u("attention.head_count_kv"),
            d_model:     u("embedding_length"),
            d_ffn:       u("feed_forward_length"),
            vocab_size:  vocab,
            max_seq_len: u("context_length"),
            rope_theta,
            norm_eps,
        }
    }

    /// Convenience: extract hyperparams directly from an open `GgufModel`.
    pub fn from_gguf(model: &crate::GgufModel) -> Self {
        Self::from_meta(model.arch(), &model.meta)
    }
}
