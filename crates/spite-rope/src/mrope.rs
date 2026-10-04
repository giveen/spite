//! M-RoPE: Multimodal Rotary Position Embedding.
//!
//! Used by Qwen-VL, DeepSeek-VL2, and other vision-language models.
//! Each token has three independent position coordinates:
//!   - temporal (for video: frame index; for text/image: constant 0)
//!   - height   (image patch row; text: constant 0)
//!   - width    (image patch column; text: seq position)
//!
//! The head dimension is split into three equal sections, each getting
//! its own position encoding. Text tokens use the same value for all
//! three coordinates (so the result equals standard RoPE for text).
//!
//! # DeepSeek-V3 MLA variant
//!
//! DeepSeek-V3 uses a subset of dimensions for RoPE (the "RoPE head dims")
//! and leaves the rest unrotated (stored in a separate compressed KV cache).
//! The `mrope_sections` parameter selects which dimensions get each coordinate.

/// Apply M-RoPE to one Q or K tensor.
///
/// `qk`:       `[n_heads × head_dim]` F32, modified in-place
/// `pos_t`:    temporal position  (video frame or 0)
/// `pos_h`:    height position    (image row or 0)
/// `pos_w`:    width position     (text seq pos, or image column)
/// `theta`:    base RoPE theta
/// `sections`: `[t_end, h_end, w_end]` — where each coord's dimensions end.
///             The head_dim is split as: [0, t_end) temporal, [t_end, h_end) height,
///             [h_end, w_end) width. Remaining dimensions are unrotated.
///             Typical for Qwen-VL: sections = [head_dim/6, head_dim/3, head_dim/2].
pub fn apply_mrope(
    qk: &mut [f32],
    pos_t: u32,
    pos_h: u32,
    pos_w: u32,
    theta: f32,
    sections: [usize; 3],
) {
    let head_dim = qk.len(); // for single head; caller loops over heads if needed
    let [t_end, h_end, w_end] = sections;

    let mut apply_section = |start: usize, end: usize, pos: u32| {
        let half = (end - start) / 2;
        for i in 0..half {
            let dim_idx = start + i;
            let freq = 1.0 / theta.powf(2.0 * dim_idx as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = qk[start + i];
            let x1 = qk[start + i + half];
            qk[start + i] = x0 * cos - x1 * sin;
            qk[start + i + half] = x0 * sin + x1 * cos;
        }
    };

    apply_section(0, t_end, pos_t);
    apply_section(t_end, h_end, pos_h);
    apply_section(h_end, w_end, pos_w);
    // Dimensions [w_end, head_dim) are left unrotated.
}

/// Compute M-RoPE section boundaries for a model with `head_dim` that splits
/// the rope dimensions evenly across the three coordinates.
///
/// If `rope_dims` < `head_dim`, only the first `rope_dims` dimensions get RoPE
/// (DeepSeek MLA style — the rest go through the compressed KV path).
pub fn uniform_sections(head_dim: usize, rope_dims: usize) -> [usize; 3] {
    let third = rope_dims / 3;
    [third, 2 * third, rope_dims.min(head_dim)]
}
