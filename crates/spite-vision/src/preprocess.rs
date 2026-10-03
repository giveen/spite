//! Image preprocessing for vision encoders.
//!
//! Pipeline (CLIP convention):
//!   1. Decode raw bytes → RGB u8 pixels
//!   2. Resize to `target_size × target_size` (bilinear)
//!   3. Normalize per channel: (pixel/255 − mean) / std
//!   4. Extract non-overlapping `patch_size × patch_size` patches
//!      → `[n_patches, patch_size*patch_size*3]` F32

/// CLIP normalization statistics (mean and std per RGB channel).
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275,  0.40821073];
pub const CLIP_STD:  [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Resize `src` (RGB, `src_h × src_w × 3`) to `dst` (`tgt × tgt × 3`) using
/// bilinear interpolation.
pub fn resize_bilinear(
    src:   &[u8],
    src_h: usize,
    src_w: usize,
    tgt:   usize,
    dst:   &mut [f32],
) {
    assert_eq!(src.len(),  src_h * src_w * 3);
    assert_eq!(dst.len(),  tgt   * tgt   * 3);

    let scale_h = src_h as f32 / tgt as f32;
    let scale_w = src_w as f32 / tgt as f32;

    for dy in 0..tgt {
        let sy_f = (dy as f32 + 0.5) * scale_h - 0.5;
        let sy0  = (sy_f.floor() as isize).clamp(0, src_h as isize - 1) as usize;
        let sy1  = (sy0 + 1).min(src_h - 1);
        let ty   = sy_f - sy_f.floor();

        for dx in 0..tgt {
            let sx_f = (dx as f32 + 0.5) * scale_w - 0.5;
            let sx0  = (sx_f.floor() as isize).clamp(0, src_w as isize - 1) as usize;
            let sx1  = (sx0 + 1).min(src_w - 1);
            let tx   = sx_f - sx_f.floor();

            let i00 = (sy0 * src_w + sx0) * 3;
            let i01 = (sy0 * src_w + sx1) * 3;
            let i10 = (sy1 * src_w + sx0) * 3;
            let i11 = (sy1 * src_w + sx1) * 3;

            let out_base = (dy * tgt + dx) * 3;
            for c in 0..3usize {
                let p00 = src[i00 + c] as f32;
                let p01 = src[i01 + c] as f32;
                let p10 = src[i10 + c] as f32;
                let p11 = src[i11 + c] as f32;
                dst[out_base + c] = (1.0 - ty) * ((1.0 - tx) * p00 + tx * p01)
                                  +        ty  * ((1.0 - tx) * p10 + tx * p11);
            }
        }
    }
}

/// Normalize float pixels in-place using per-channel mean and std.
/// `pixels`: `[H × W × 3]` in range `[0, 255]`.
pub fn normalize_clip(pixels: &mut [f32]) {
    let n = pixels.len() / 3;
    for i in 0..n {
        for c in 0..3usize {
            pixels[i * 3 + c] = (pixels[i * 3 + c] / 255.0 - CLIP_MEAN[c]) / CLIP_STD[c];
        }
    }
}

/// Extract non-overlapping patches from a normalized image tensor.
///
/// `img`:        `[image_size × image_size × 3]` F32 (row-major)
/// `image_size`: total resolution (must be divisible by `patch_size`)
/// `patch_size`: e.g. 14
///
/// Returns `[n_patches × (patch_size*patch_size*3)]` F32.
/// Patches are in raster order: left-to-right, top-to-bottom.
pub fn extract_patches(img: &[f32], image_size: usize, patch_size: usize) -> Vec<f32> {
    let n_side   = image_size / patch_size;
    let patch_dim = patch_size * patch_size * 3;
    let n_patches = n_side * n_side;
    let mut out = vec![0f32; n_patches * patch_dim];

    for py in 0..n_side {
        for px in 0..n_side {
            let patch_idx = py * n_side + px;
            let out_base  = patch_idx * patch_dim;
            let mut k = 0usize;
            for dy in 0..patch_size {
                for dx in 0..patch_size {
                    let src_base = ((py * patch_size + dy) * image_size + (px * patch_size + dx)) * 3;
                    for c in 0..3usize {
                        out[out_base + k] = img[src_base + c];
                        k += 1;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_count() {
        let img = vec![0f32; 28 * 28 * 3];
        let patches = extract_patches(&img, 28, 14);
        // 28/14 = 2 → 4 patches
        assert_eq!(patches.len(), 4 * 14 * 14 * 3);
    }
}
