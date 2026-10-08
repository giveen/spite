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
    /// MoE geometry (0 experts = dense arch).
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub moe_layer_step: u32,
    pub expert_weights_scale: f32,
    /// Per-layer SWA flags (llama4-style chunked pattern fallback).
    pub swa_layers: Vec<bool>,
    /// Per-layer SwiGLU clamp limits (0 = no clamp).
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,
    /// MLA / KDA geometry (0 = unused).
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub rope_dim_count: u32,
    pub kda_head_dim: u32,
    pub kda_gate_lower_bound: f32,
    pub n_layer_dense_lead: u32,
    pub n_expert_latent: u32,
    pub expert_weights_norm: bool,
    pub expert_gating_func: u32,
    pub attn_res_block_size: u32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    /// Per-layer kv-head counts when stored as an array (kimi-style
    /// recurrent marking: 0 kv heads = recurrent layer).
    pub head_count_kv_arr: Vec<u32>,
    /// DeepSeek-V4 hyper-connections (0 = unused).
    pub hc_mult: u32,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: u32,
    /// Grouped MLA output projection.
    pub o_group_count: u32,
    pub o_lora_rank: u32,
    /// Per-layer sparse compression ratios (0 = dense raw attention).
    pub compress_ratios: Vec<u32>,
    pub compress_rope_base: f32,
    /// Shared experts (deepseek4-style fine-grained MoE).
    pub n_expert_shared: u32,
    /// Indexer geometry for sparse attention (0 = unused).
    pub indexer_n_head: u32,
    pub indexer_head_size: u32,
    pub indexer_top_k: u32,
    /// Per-layer full-indexer flags (empty = all full).
    pub indexer_types: Vec<bool>,
    pub key_length_mla: u32,
    pub value_length_mla: u32,
    pub key_length_swa: u32,
    pub value_length_swa: u32,
    pub rope_freq_base_swa: f32,
    pub final_logit_softcapping: f32,
    /// Multi-Token Prediction (MTP / NextN) draft blocks count.
    pub n_nextn_predict_layers: u32,
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

        let f32_arr = |key: &String| -> Vec<f32> {
            match meta.get(key) {
                Some(crate::MetaValue::Array(items)) => items
                    .iter()
                    .map(|v| match v {
                        crate::MetaValue::F32(x) => *x,
                        crate::MetaValue::F64(x) => *x as f32,
                        _ => 0.0,
                    })
                    .collect(),
                _ => Vec::new(),
            }
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
        // NextN/MTP blocks are appended after the trunk (llama.cpp: n_layer excludes them);
        // they are not part of the main forward pass.
        let n_layers = u("block_count").saturating_sub(u("nextn_predict_layers"));
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

        let head_count_kv_arr: Vec<u32> = match meta.get(&format!("{arch}.attention.head_count_kv"))
        {
            Some(crate::MetaValue::Array(items)) => items
                .iter()
                .map(|v| match v {
                    crate::MetaValue::U32(x) => *x,
                    crate::MetaValue::I32(x) => *x as u32,
                    _ => 0,
                })
                .collect(),
            _ => Vec::new(),
        };
        let n_kv_heads = {
            let scalar = u("attention.head_count_kv");
            if scalar > 0 {
                scalar
            } else {
                head_count_kv_arr.first().copied().unwrap_or(0)
            }
        };

        Self {
            arch: arch.to_owned(),
            n_layers,
            n_heads: u("attention.head_count"),
            n_kv_heads,
            d_model: u("embedding_length"),
            d_ffn: {
                let ffn = u("feed_forward_length");
                if ffn > 0 {
                    ffn
                } else {
                    u("expert_feed_forward_length")
                }
            },
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
            n_expert: u("expert_count"),
            n_expert_used: u("expert_used_count"),
            moe_layer_step: u("interleave_moe_layer_step"),
            expert_weights_scale: f("expert_weights_scale"),
            swa_layers: {
                let key = format!("{arch}.attention.sliding_window_pattern");
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
                        // Chunked pattern fallback (llama4): 3 SWA + 1 full.
                        let sw = u("attention.sliding_window") > 0;
                        (0..n_layers).map(|i| sw && i % 4 < 3).collect()
                    }
                }
            },
            swiglu_clamp_exp: f32_arr(&format!("{arch}.swiglu_clamp_exp")),
            swiglu_clamp_shexp: f32_arr(&format!("{arch}.swiglu_clamp_shexp")),
            q_lora_rank: u("attention.q_lora_rank"),
            kv_lora_rank: u("attention.kv_lora_rank"),
            key_length: u("attention.key_length"),
            value_length: u("attention.value_length"),
            rope_dim_count: u("rope.dimension_count"),
            kda_head_dim: u("kda.head_dim"),
            kda_gate_lower_bound: {
                let v = f("kda.gate_lower_bound");
                if v == 0.0 { f32::NEG_INFINITY } else { v }
            },
            n_layer_dense_lead: u("leading_dense_block_count"),
            n_expert_latent: u("expert_latent_length"),
            expert_weights_norm: matches!(
                meta.get(&format!("{arch}.expert_weights_norm")),
                Some(crate::MetaValue::Bool(true))
                    | Some(crate::MetaValue::U32(1))
                    | Some(crate::MetaValue::I32(1))
            ),
            expert_gating_func: u("expert_gating_func"),
            attn_res_block_size: u("attn_res.block_size"),
            situ_beta: f("activation.situ_beta"),
            situ_linear_beta: f("activation.situ_linear_beta"),
            head_count_kv_arr,
            hc_mult: u("hyper_connection.count"),
            hc_eps: f("hyper_connection.epsilon"),
            hc_sinkhorn_iters: u("hyper_connection.sinkhorn_iterations"),
            o_group_count: u("attention.output_group_count"),
            o_lora_rank: u("attention.output_lora_rank"),
            compress_ratios: match meta.get(&format!("{arch}.attention.compress_ratios")) {
                Some(crate::MetaValue::Array(items)) => items
                    .iter()
                    .map(|v| match v {
                        crate::MetaValue::U32(x) => *x,
                        crate::MetaValue::I32(x) => *x as u32,
                        _ => 0,
                    })
                    .collect(),
                _ => Vec::new(),
            },
            compress_rope_base: f("attention.compress_rope_freq_base"),
            n_expert_shared: u("expert_shared_count"),
            indexer_n_head: u("attention.indexer_head_count"),
            indexer_head_size: u("attention.indexer_key_length"),
            indexer_top_k: u("attention.indexer_top_k"),
            indexer_types: match meta.get(&format!("{arch}.attention.indexer.types")) {
                Some(crate::MetaValue::Array(items)) => items
                    .iter()
                    .map(|v| match v {
                        crate::MetaValue::Bool(b) => *b,
                        crate::MetaValue::U32(x) => *x != 0,
                        crate::MetaValue::I32(x) => *x != 0,
                        _ => true,
                    })
                    .collect(),
                _ => Vec::new(),
            },
            key_length_mla: u("attention.key_length_mla"),
            value_length_mla: u("attention.value_length_mla"),
            key_length_swa: u("attention.key_length_swa"),
            value_length_swa: u("attention.value_length_swa"),
            rope_freq_base_swa: f("rope.freq_base_swa"),
            final_logit_softcapping: f("final_logit_softcapping"),
            n_nextn_predict_layers: u("nextn_predict_layers"),
        }
    }

    /// Convenience: extract hyperparams directly from an open `GgufModel`.
    ///
    /// Also repairs `vocab_size`: some quantized files omit
    /// `tokenizer.ggml.token_count`, in which case the vocabulary size is
    /// inferred from the embedding / output projection tensors. Without this
    /// the LM head matvec gets a zero-width output and every consumer
    /// (CLI, server, bench) fails with a shape mismatch.
    pub fn from_gguf(model: &crate::GgufModel) -> Self {
        let mut hp = Self::from_meta(model.arch(), &model.meta);
        if hp.vocab_size == 0 {
            hp.vocab_size = model.vocab_size() as u32;
        }
        hp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetaValue;

    /// Metadata of `Qwen3.8-27B-Q6_K` (unsloth), transcribed from the GGUF
    /// (`general.architecture = qwen35`). Every field the hybrid decoder reads
    /// is present, including the ones a hand-written config tends to omit.
    fn qwen38_27b_meta() -> HashMap<String, MetaValue> {
        let mut meta: HashMap<String, MetaValue> = [
            ("block_count", 65u32),
            ("nextn_predict_layers", 1),
            ("embedding_length", 5120),
            ("feed_forward_length", 17408),
            ("attention.head_count", 24),
            ("attention.head_count_kv", 4),
            ("attention.key_length", 256),
            ("attention.value_length", 256),
            ("attention.layer_norm_rms_epsilon", 0),
            ("context_length", 262144),
            ("full_attention_interval", 4),
            ("rope.dimension_count", 64),
            ("ssm.conv_kernel", 4),
            ("ssm.inner_size", 6144),
            ("ssm.state_size", 128),
            ("ssm.time_step_rank", 48),
            ("ssm.group_count", 16),
        ]
        .into_iter()
        .map(|(k, v)| (format!("qwen35.{k}"), MetaValue::U32(v)))
        .collect();
        meta.insert("qwen35.rope.freq_base".into(), MetaValue::F32(1.0e7));
        meta.insert(
            "qwen35.rope.dimension_sections".into(),
            MetaValue::Array(vec![
                MetaValue::U32(11),
                MetaValue::U32(11),
                MetaValue::U32(10),
                MetaValue::U32(0),
            ]),
        );
        meta.insert(
            "general.architecture".into(),
            MetaValue::Str("qwen35".into()),
        );
        meta
    }

    #[test]
    fn qwen38_27b_metadata_parses() {
        let hp = ModelHyperparams::from_meta("qwen35", &qwen38_27b_meta());

        // 65 stored blocks = 64 trunk layers + 1 MTP/NextN block.
        assert_eq!(hp.n_layers, 64);
        assert_eq!(hp.n_nextn_predict_layers, 1);
        assert_eq!((hp.d_model, hp.d_ffn), (5120, 17408));
        assert_eq!((hp.n_heads, hp.n_kv_heads), (24, 4));

        // 256-wide k/v heads, RoPE over the first 64 dims, base 1e7.
        assert_eq!((hp.key_length, hp.value_length), (256, 256));
        assert_eq!(hp.rope_dim_count, 64);
        assert_eq!(hp.rope_theta, 1.0e7);
        assert_eq!(hp.rope_sections, [11, 11, 10, 0]);
        assert_eq!(hp.max_seq_len, 262144);

        // Gated Delta Net geometry.
        assert_eq!(hp.ssm_d_conv, 4);
        assert_eq!(hp.ssm_d_inner, 6144);
        assert_eq!(hp.ssm_d_state, 128);
        assert_eq!(hp.ssm_dt_rank, 48);
        assert_eq!(hp.ssm_n_group, 16);
        assert_eq!(hp.full_attention_interval, 4);
    }

    #[test]
    fn qwen38_recurrent_layers_follow_the_interval() {
        let hp = ModelHyperparams::from_meta("qwen35", &qwen38_27b_meta());
        assert_eq!(hp.recurrent_layers.len(), 64);
        // Every 4th layer is full attention (interval 4): 16 full, 48 GDN.
        assert_eq!(hp.recurrent_layers.iter().filter(|&&r| r).count(), 48);
        assert_eq!(hp.recurrent_layers.iter().filter(|&&r| !r).count(), 16);
        assert!(hp.recurrent_layers[0]);
        assert!(!hp.recurrent_layers[3]);
        assert!(!hp.recurrent_layers[63]);
    }

    /// The numbers the hybrid decoder validates against at load:
    /// `n_vh = ssm_dt_rank` (48), `S = ssm_d_state` (128), `n_kh = group_count`
    /// (16), and `inner == n_vh * S`; plus `head_dim = key_length` and
    /// `rope_dim = rope_dim_count`.
    #[test]
    fn qwen38_hybrid_geometry_is_consistent() {
        let hp = ModelHyperparams::from_meta("qwen35", &qwen38_27b_meta());
        let (n_kh, n_vh, s) = (hp.ssm_n_group, hp.ssm_dt_rank, hp.ssm_d_state);
        assert_eq!(hp.ssm_d_inner, n_vh * s, "inner != dt_rank * state");
        assert_eq!(n_vh % n_kh, 0, "dt_rank not divisible by group_count");
        // Head dim comes from key_length, not d_model / n_heads (5120/24 = 213).
        assert_eq!(hp.key_length, 256);
        assert!(hp.rope_dim_count <= hp.key_length);
    }
}
