//! Low-level tensor compute ops.
//!
//! Everything the model architectures need that isn't inside a GPU kernel:
//! matmul dispatch, block dequantization, softmax, embedding lookup.
//! These call into the DispatchTable when a GPU kernel is available,
//! and fall back to pure-Rust scalar implementations otherwise.

pub mod matmul;
pub mod dequant;
pub mod flash_attn;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ComputeError {
    #[error("unsupported dtype for this op")]
    UnsupportedDtype,
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("out of device memory")]
    OutOfMemory,
    #[error("index out of bounds: token {token} >= vocab {vocab}")]
    OutOfBounds { token: usize, vocab: usize },
}

/// Embed a batch of token ids by indexing rows from a weight matrix.
///
/// `weight`: `[vocab_size, d_model]` F32
/// `ids`:    `[seq_len]` token ids
/// `out`:    `[seq_len * d_model]` F32, caller-allocated
pub fn embed_tokens(
    weight:    &[f32], // flat [vocab_size, d_model]
    vocab_size: usize,
    d_model:   usize,
    ids:       &[u32],
    out:       &mut [f32],
) -> Result<(), ComputeError> {
    for (i, &id) in ids.iter().enumerate() {
        let t = id as usize;
        if t >= vocab_size {
            return Err(ComputeError::OutOfBounds { token: t, vocab: vocab_size });
        }
        let src = &weight[t * d_model..(t + 1) * d_model];
        let dst = &mut out[i * d_model..(i + 1) * d_model];
        dst.copy_from_slice(src);
    }
    Ok(())
}

/// Softmax in-place over a logit slice (numerical stability via max subtraction).
pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() { return; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in logits.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    let inv = 1.0 / sum.max(1e-30);
    for x in logits.iter_mut() { *x *= inv; }
}

/// Convert a half-precision f16 bit pattern (stored as u16) to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp  = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let f = if exp == 0 {
        // subnormal
        f32::from_bits(sign | ((mant as u32) << 13))
    } else if exp == 0x1F {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    };
    f
}
