//! Multimodal vision support (LLaVA / LLaVA-1.5 style).
//!
//! Pipeline:
//!   1. Raw image (RGB bytes) → resize to `image_size × image_size`
//!   2. Extract `n_patches = (image_size / patch_size)²` non-overlapping patches
//!   3. ViT transformer forward pass → `[n_patches, d_vision]` embeddings
//!   4. MLP projection → `[n_patches, d_text]` — the "image tokens"
//!   5. Splice image tokens into the text token sequence before the main forward pass
//!
//! Weight layout (two GGUF files or one combined file):
//!   vision_model/patch_embd.weight      [d_vision, patch_h*patch_w*3]
//!   vision_model/blk.{i}.*              ViT transformer layers (same naming as text)
//!   vision_model/post_ln.weight         [d_vision]
//!   mm_projector/fc1.weight             [d_proj, d_vision]
//!   mm_projector/fc1.bias               [d_proj]
//!   mm_projector/fc2.weight             [d_text, d_proj]
//!   mm_projector/fc2.bias               [d_text]

use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VisionError {
    #[error("image decode failed: {0}")]
    Decode(String),
    #[error("image resize failed: {0}")]
    Resize(String),
    #[error("encoder forward pass failed: {0}")]
    Encoder(String),
    #[error("projection failed: {0}")]
    Projection(String),
    #[error("model load failed: {0}")]
    Load(String),
}

#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Input image resolution (e.g. 336 for LLaVA-1.5).
    pub image_size:       usize,
    /// Patch size (e.g. 14 for CLIP ViT-L/14).
    pub patch_size:       usize,
    /// Number of patches: (image_size / patch_size)².
    pub n_patches:        usize,
    /// ViT hidden dimension (e.g. 1024 for ViT-L).
    pub d_vision:         usize,
    /// Text model hidden dimension the projection maps into.
    pub d_text:           usize,
    /// MLP hidden dim between fc1 and fc2 in the projector.
    pub d_proj:           usize,
    pub n_vision_layers:  usize,
    pub n_vision_heads:   usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        let image_size = 336;
        let patch_size = 14;
        Self {
            image_size,
            patch_size,
            n_patches:       (image_size / patch_size) * (image_size / patch_size),
            d_vision:        1024,
            d_text:          4096,
            d_proj:          4096,
            n_vision_layers: 24,
            n_vision_heads:  16,
        }
    }
}

/// Loaded vision encoder + MLP projection layer.
pub struct VisionEncoder {
    pub cfg: VisionConfig,
    // TODO: patch_embd:  Vec<f32>  [d_vision, patch_dim]
    // TODO: vit_layers:  Vec<ViTLayer>
    // TODO: post_norm:   Vec<f32>  [d_vision]
    // TODO: proj_fc1:    (Vec<f32>, Vec<f32>)  (weight, bias)
    // TODO: proj_fc2:    (Vec<f32>, Vec<f32>)
}

impl VisionEncoder {
    /// Load from a GGUF file (combined or vision-only).
    pub fn from_gguf(_path: &Path) -> Result<Self, VisionError> {
        // TODO: open with spite-loader, read VisionConfig from metadata,
        //       load all vision_model/* and mm_projector/* tensors
        Err(VisionError::Load("vision encoder not yet implemented".into()))
    }

    /// Encode a raw image into text-space token embeddings.
    ///
    /// `pixels`: RGB, height × width × 3
    /// Returns `[n_patches × d_text]` F32 — splice before text tokens.
    pub fn encode(
        &self,
        _pixels: &[u8],
        _height: usize,
        _width:  usize,
    ) -> Result<Vec<f32>, VisionError> {
        // TODO:
        // 1. Resize pixels to cfg.image_size × cfg.image_size (bilinear)
        // 2. Normalise: (pixel / 255 - mean) / std  (CLIP stats)
        // 3. Extract patches → [n_patches, patch_h * patch_w * 3]
        // 4. patch_embd_out = patches @ patch_embd.T + cls_token
        // 5. ViT forward pass (n_vision_layers blocks)
        // 6. post_norm(vit_out)
        // 7. proj = gelu(vit_out @ fc1.T + fc1.bias) @ fc2.T + fc2.bias
        Err(VisionError::Encoder("not implemented".into()))
    }
}
