//! LoRA weight merging — fuse adapter delta into the base model permanently.
//!
//! After merging, the model weights already include the LoRA delta and the
//! adapter can be discarded. This is useful when deploying a fine-tuned
//! model: a single merged GGUF requires no adapter loading overhead.
//!
//! Merging is destructive to the base weight. Save the original first if
//! you want to be able to switch adapters.
//!
//! # Formula
//!
//!   W_merged = W_base + (B × A) × (alpha / rank)
//!
//! where A and B are the LoRA adapter matrices loaded from the GGUF.

use crate::{LoraAdapter, LoraLayer, apply_lora};

/// Fuse all LoRA layers in `adapter` into `weights`.
///
/// `weights`: flat weight buffers keyed by tensor name ("blk.0.attn_q.weight").
/// Tensors not covered by the adapter are left untouched.
pub fn merge_adapter(
    weights: &mut std::collections::HashMap<String, Vec<f32>>,
    adapter: &LoraAdapter,
) {
    let scale = adapter.config.scale();
    for layer in &adapter.layers {
        let key = format!("blk.{}.{}.weight", layer.layer_idx, layer.module);
        if let Some(w) = weights.get_mut(&key) {
            apply_lora(w, &layer.a, &layer.b, scale, layer.out_feat, layer.in_feat, layer.rank);
        }
    }
}

/// Check that `layer`'s dimensions are consistent.
pub fn validate_layer(layer: &LoraLayer) -> Result<(), String> {
    if layer.a.len() != layer.rank * layer.in_feat {
        return Err(format!(
            "lora_a shape mismatch: expected {}×{} = {}, got {}",
            layer.rank, layer.in_feat, layer.rank * layer.in_feat, layer.a.len()
        ));
    }
    if layer.b.len() != layer.out_feat * layer.rank {
        return Err(format!(
            "lora_b shape mismatch: expected {}×{} = {}, got {}",
            layer.out_feat, layer.rank, layer.out_feat * layer.rank, layer.b.len()
        ));
    }
    Ok(())
}
