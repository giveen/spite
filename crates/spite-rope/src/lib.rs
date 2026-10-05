//! Rotary positional embeddings (RoPE) and variants.
//!
//! Applied in-place to Q and K tensors before the attention score computation.
//!
//! Variants
//! --------
//! Default   — standard RoPE (LLaMA, Mistral, Gemma)
//! Linear    — multiply frequencies by a constant scale (modest extension)
//! YaRN      — NTK-aware interpolation; extends context with minimal quality loss
//! ALiBi     — attention with linear biases (BLOOM, MPT); no rotation
//! LongRoPE  — non-uniform per-dimension rescaling for very long contexts

pub mod longrope;
pub mod mrope;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RopeError {
    #[error("head_dim must be even, got {0}")]
    OddHeadDim(usize),
    #[error("position {pos} exceeds max context {max_ctx}")]
    PositionOverflow { pos: u32, max_ctx: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeVariant {
    Default,
    Linear { scale: f32 },
    Yarn { scale: f32, original_ctx: usize },
    Alibi,
    LongRope,
}

#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub head_dim: usize,
    pub theta: f32, // default 10 000.0; llama3 uses 500 000.0
    pub variant: RopeVariant,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            theta: 10_000.0,
            variant: RopeVariant::Default,
        }
    }
}

/// Apply RoPE in-place to a single query or key tensor.
///
/// `qk`:  flat F32 buffer `[n_heads, head_dim]` for one sequence position
/// `pos`: the absolute token position
/// `cfg`: rope configuration
pub fn apply_rope(qk: &mut [f32], pos: u32, cfg: &RopeConfig) -> Result<(), RopeError> {
    let d = cfg.head_dim;
    if !d.is_multiple_of(2) {
        return Err(RopeError::OddHeadDim(d));
    }
    let theta_scale = match cfg.variant {
        RopeVariant::Linear { scale } => scale,
        RopeVariant::Yarn { scale, .. } => scale,
        _ => 1.0,
    };
    // Rotate every head: `qk` holds `n_heads` contiguous `head_dim` slices.
    for head in qk.chunks_exact_mut(d) {
        rope_range(head, pos, 0, d, cfg.theta * theta_scale, false);
    }
    Ok(())
}

/// Rotate pairs in `[offset, offset+n_dims)` by `pos`-dependent angles
/// (neox half-split layout: pairs are `(i, i+n_dims/2)`); pairs outside
/// the range are untouched. `invert` rotates back (llama.cpp
/// `rope_ext_back`, used to derope MLA outputs).
pub fn rope_range(x: &mut [f32], pos: u32, offset: usize, n_dims: usize, theta: f32, invert: bool) {
    let half = n_dims / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / n_dims as f32);
        let mut angle = pos as f32 * freq;
        if invert {
            angle = -angle;
        }
        let (sin, cos) = angle.sin_cos();
        let x0 = x[offset + i];
        let x1 = x[offset + i + half];
        x[offset + i] = x0 * cos - x1 * sin;
        x[offset + i + half] = x0 * sin + x1 * cos;
    }
}

/// Interleaved RoPE (llama.cpp `IMROPE`, Qwen3.5/Qwen4-style).
///
/// Head dim is split into 4 sections; pair `p` uses the position of its
/// sector, interleaved 3-way across sections. With all four positions equal
/// (text-only inference) this reduces to standard neox RoPE over `n_dims`.
///
/// `x`: flat F32 `[n_heads, head_dim]` for one position; only the first
/// `n_dims` of each head rotate.
pub fn apply_irope(
    x: &mut [f32],
    pos: [u32; 4],
    sections: [u32; 4],
    n_dims: usize,
    head_dim: usize,
    theta: f32,
) {
    let sect = (sections[0] + sections[1] + sections[2] + sections[3]) as usize;
    if sect == 0 || n_dims == 0 {
        return;
    }
    let s1 = sections[1] as usize;
    let s2 = sections[2] as usize;
    let s0 = sections[0] as usize;
    let n_heads = x.len() / head_dim;
    let half = n_dims / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for p in 0..half {
            let sector = p % sect;
            // Interleaved sector → position mapping (ggml mrope cache init).
            let position = if sector % 3 == 1 && sector < 3 * s1 {
                pos[1]
            } else if sector % 3 == 2 && sector < 3 * s2 {
                pos[2]
            } else if sector.is_multiple_of(3) && sector < 3 * s0 {
                pos[0]
            } else {
                pos[3]
            };
            let freq = 1.0 / theta.powf(2.0 * p as f32 / n_dims as f32);
            let (sin, cos) = (position as f32 * freq).sin_cos();
            let x0 = x[base + p];
            let x1 = x[base + p + half];
            x[base + p] = x0 * cos - x1 * sin;
            x[base + p + half] = x0 * sin + x1 * cos;
        }
    }
}

/// ALiBi: return the slope for head `h` of `n_heads` total.
///
/// This slope is subtracted from attention scores: score -= slope * distance.
pub fn alibi_slope(h: usize, n_heads: usize) -> f32 {
    let m = n_heads as f32;
    2f32.powf(-8.0 * (h as f32 + 1.0) / m)
}

/// Build a YaRN correction map for each dimension pair.
///
/// Returns a vec of per-pair scale factors for the frequency interpolation.
/// See https://arxiv.org/abs/2309.00071 for the algorithm.
pub fn yarn_correction_dims(
    head_dim: usize,
    theta: f32,
    original_ctx: usize,
    target_ctx: usize,
) -> Vec<f32> {
    let scale = target_ctx as f32 / original_ctx as f32;
    (0..head_dim / 2)
        .map(|i| {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let wavelength = 2.0 * std::f32::consts::PI / freq;
            // Interpolate between linear and NTK scaling based on wavelength
            let alpha = 1.0f32; // TODO: derive from context length ratio
            let beta = 32.0f32; // TODO: tune per model
            let ramp =
                ((wavelength / original_ctx as f32 - alpha) / (beta - alpha)).clamp(0.0, 1.0);
            1.0 / (ramp / scale + (1.0 - ramp)) // TODO: per-dim correction
        })
        .collect()
}
