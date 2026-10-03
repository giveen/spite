//! LoRA (Low-Rank Adaptation) adapter loading and application.
//!
//! A LoRA adapter stores delta weights as low-rank factored matrices:
//!   W' = W + (B × A) × (alpha / rank)
//! where A ∈ ℝ^{rank × in_features} and B ∈ ℝ^{out_features × rank}.
//!
//! Adapters are stored as GGUF files with metadata keys:
//!   lora.rank              u32
//!   lora.alpha             f32
//!   lora.target_modules    string array   e.g. ["attn_q", "attn_v"]
//!   lora.base_model_arch   string         must match the loaded base model
//!
//! Weight names follow the same GGUF convention as the base model, with
//! `.lora_a` and `.lora_b` suffixes:
//!   blk.0.attn_q.lora_a   [rank, d_model]
//!   blk.0.attn_q.lora_b   [n_heads*head_dim, rank]

use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoraError {
    #[error("adapter file not found: {0}")]
    FileNotFound(String),
    #[error("rank mismatch: adapter {adapter} != expected {expected}")]
    RankMismatch { adapter: u32, expected: u32 },
    #[error("incompatible base model architecture: adapter expects {adapter}, got {base}")]
    ArchMismatch { adapter: String, base: String },
    #[error("missing adapter weight: {0}")]
    MissingWeight(String),
}

#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub rank:           u32,
    pub alpha:          f32,
    pub target_modules: Vec<String>,
    pub base_arch:      String,
}

impl LoraConfig {
    /// Scaling factor applied after B × A.
    pub fn scale(&self) -> f32 { self.alpha / self.rank as f32 }
}

/// One A/B pair for a single weight matrix in a single transformer layer.
pub struct LoraLayer {
    pub layer_idx: usize,
    pub module:    String,      // e.g. "attn_q"
    pub a:         Vec<f32>,    // [rank, in_features]
    pub b:         Vec<f32>,    // [out_features, rank]
    pub rank:      usize,
    pub in_feat:   usize,
    pub out_feat:  usize,
}

/// A fully loaded LoRA adapter.
pub struct LoraAdapter {
    pub config: LoraConfig,
    pub layers: Vec<LoraLayer>,
}

impl LoraAdapter {
    /// Load from a GGUF file.
    pub fn from_gguf(_path: &Path) -> Result<Self, LoraError> {
        // TODO:
        // 1. Open GGUF with spite-loader
        // 2. Read lora.rank, lora.alpha, lora.target_modules, lora.base_model_arch
        // 3. For each (layer_idx, module) in target_modules × n_layers:
        //    load "blk.{i}.{module}.lora_a" and "blk.{i}.{module}.lora_b"
        Ok(Self {
            config: LoraConfig {
                rank:           16,
                alpha:          16.0,
                target_modules: vec![],
                base_arch:      String::new(),
            },
            layers: vec![],
        })
    }
}

/// Apply a LoRA delta to a weight matrix in-place:
///   weight[out, in] += (b[out, rank] × a[rank, in]) * scale
pub fn apply_lora(
    weight:  &mut [f32],
    a:       &[f32],
    b:       &[f32],
    scale:   f32,
    out_dim: usize,
    in_dim:  usize,
    rank:    usize,
) {
    // Compute B × A and accumulate into weight
    for o in 0..out_dim {
        for i in 0..in_dim {
            let mut delta = 0f32;
            for r in 0..rank {
                delta += b[o * rank + r] * a[r * in_dim + i];
            }
            weight[o * in_dim + i] += delta * scale;
        }
    }
}
