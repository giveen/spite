//! Vision Transformer (ViT) encoder forward pass.
//!
//! Implements the CLIP ViT forward pass used by LLaVA-style multimodal models.
//! The encoder takes patch embeddings and outputs hidden states at each patch
//! position, which are then projected into the text model's embedding space.
//!
//! # Architecture
//!
//! Based on ViT-L/14 (as used in LLaVA-1.5):
//! - d_vision = 1024
//! - n_heads  = 16 (64-dim head)
//! - n_layers = 24
//! - patch_size = 14, image_size = 336 → 576 patches
//!
//! Differences from the text transformer:
//! - Pre-norm (LN before attention/FFN, not after)
//! - GELU activation (not SwiGLU)
//! - Absolute positional embeddings (not RoPE)
//! - CLS token prepended (output at position 0 is the global image embedding)

use crate::VisionConfig;

/// One ViT transformer layer's weights (placeholders).
pub struct ViTLayer {
    pub layer_idx: usize,
    // Attention projections
    // TODO: wq, wk, wv, wo: [d_vision, d_vision]
    // FFN projections (GELU)
    // TODO: w_fc1: [d_ffn, d_vision], w_fc2: [d_vision, d_ffn]
    // Layer norms
    // TODO: ln1_weight, ln1_bias, ln2_weight, ln2_bias: [d_vision]
}

/// Full ViT encoder.
pub struct ViTEncoder {
    pub cfg:    VisionConfig,
    pub layers: Vec<ViTLayer>,
    // TODO: patch_embed:  [d_vision, patch_dim]   (patch projection)
    // TODO: pos_embed:    [1 + n_patches, d_vision] (CLS + positional)
    // TODO: cls_token:    [1, d_vision]
    // TODO: post_ln:      (weight, bias) each [d_vision]
}

impl ViTEncoder {
    /// Construct an empty encoder (weights not yet loaded).
    pub fn new(cfg: VisionConfig) -> Self {
        let layers = (0..cfg.n_vision_layers)
            .map(|i| ViTLayer { layer_idx: i })
            .collect();
        Self { cfg, layers }
    }

    /// Forward pass.
    ///
    /// `patches_flat`: `[n_patches × patch_dim]` F32, output of `extract_patches`
    ///                  after running through the patch linear projection.
    ///
    /// Returns `[n_patches × d_vision]` F32 (the per-patch hidden states,
    /// CLS token excluded).
    pub fn forward(&self, _patches_flat: &[f32]) -> Vec<f32> {
        // TODO:
        // 1. linear(patches_flat, patch_embed) → [n_patches, d_vision]
        // 2. prepend cls_token → [(1+n_patches), d_vision]
        // 3. add pos_embed
        // 4. for each layer: pre_ln → attn → residual → pre_ln → ffn → residual
        // 5. post_ln(out)
        // 6. strip CLS, return [n_patches, d_vision]
        vec![0f32; self.cfg.n_patches * self.cfg.d_vision]
    }
}
