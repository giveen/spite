//! LongRoPE: non-uniform per-dimension RoPE rescaling for very long contexts.
//!
//! Standard RoPE and YaRN use the same scale factor for all frequency dimensions.
//! LongRoPE (Ding et al. 2024) instead searches for optimal per-dimension
//! rescale factors λ_i that minimize perplexity at the target context length.
//! These factors are stored in GGUF as `llama.rope.long_factor` (float array).
//!
//! # GGUF keys
//!
//!   llama.rope.long_factor    f32[head_dim/2]  per-dimension λ values (>1 context)
//!   llama.rope.short_factor   f32[head_dim/2]  per-dimension λ values (≤ original ctx)
//!   llama.rope.scaling.type   string           "longrope"
//!   llama.rope.scaling.factor f32              overall scale (unused; per-dim wins)

/// Per-dimension LongRoPE rescaling: apply `long_factors` (or `short_factors`)
/// to the frequency table before computing angles.
///
/// For position `pos > original_ctx`, use `long_factors`.
/// For `pos ≤ original_ctx`, use `short_factors` (or ones if absent).
///
/// `qk`:           `[n_heads × head_dim]` F32, modified in-place
/// `pos`:          token position
/// `theta`:        base RoPE theta
/// `factors`:      per-dimension scale factors `[head_dim/2]`
///                 frequencies are divided by these values
pub fn apply_longrope(qk: &mut [f32], pos: u32, theta: f32, factors: &[f32]) {
    let head_dim = factors.len() * 2;
    let n_heads = qk.len() / head_dim;

    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..head_dim / 2 {
            let lambda = factors[i];
            let freq = 1.0 / (theta.powf(2.0 * i as f32 / head_dim as f32) * lambda);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = qk[base + i];
            let x1 = qk[base + i + head_dim / 2];
            qk[base + i] = x0 * cos - x1 * sin;
            qk[base + i + head_dim / 2] = x0 * sin + x1 * cos;
        }
    }
}

/// Build a LongRoPE factor array that blends short and long factors based
/// on whether the current position exceeds `original_ctx`.
pub fn select_factors<'a>(
    short_factors: &'a [f32],
    long_factors: &'a [f32],
    pos: u32,
    original_ctx: usize,
) -> &'a [f32] {
    if pos as usize > original_ctx {
        long_factors
    } else {
        short_factors
    }
}
