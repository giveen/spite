//! ModernBERT — GGUF arch `modern-bert`.
//!
//! Variants: ModernBERT-Base (149M), ModernBERT-Large (395M), EuroBERT (2025).
//! Released December 2024 by Answer.AI / LightOn. Replaces BERT/RoBERTa.
//!
//! # Encoder-only architecture
//!
//! ModernBERT is a **bidirectional encoder** (not a causal decoder). It processes
//! the full input sequence simultaneously — every token attends to every other.
//! Use it for embeddings, classification, retrieval, and NLU tasks, **not** for
//! text generation.
//!
//! Key differences from original BERT:
//! - **RoPE** (replaces learned absolute position embeddings).
//! - **Alternating global + local** attention: even layers are global (full
//!   sequence), odd layers are local (sliding window = 128).
//! - Extended context: 8 192 tokens (vs BERT's 512).
//! - Flash Attention 3 compatible; unpadded inputs (no wasted computation on
//!   padding tokens).
//! - GeGLU FFN activation (same as Gemma 3).
//! - No `[SEP]` / `[CLS]` token type IDs — clean BPE tokenizer only.
//!
//! # Forward pass difference
//!
//! No causal mask; use a padding mask only. `logits_out` is a per-token
//! d_model embedding (not a vocabulary distribution) for the embedding use case,
//! or a classification head output for fine-tuned models.

use crate::{ModelArch, ModelConfig, ModelError};
use spite_abi::SpiteCtx;

pub struct ModernBert {
    config: ModelConfig,
}

impl ModernBert {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArch for ModernBert {
    fn config(&self) -> &ModelConfig { &self.config }

    fn forward(
        &self,
        _tokens:     &[u32],
        _logits_out: &mut [f32],
        _ctx:        &SpiteCtx,
    ) -> Result<(), ModelError> {
        // TODO: bidirectional encoder loop:
        //   embed(tokens) → no positional add (RoPE applied inside attn)
        //   for each layer:
        //     if even → global self-attention (no causal mask, full RoPE)
        //     if odd  → local self-attention (sliding_window=128, RoPE)
        //     GeGLU FFN
        //   pool or return all token hidden states for downstream task
        // NOTE: this arch returns embeddings; logits_out is [seq_len × d_model],
        //       not [seq_len × vocab_size].
        Err(ModelError::Forward("not implemented".into()))
    }
}
