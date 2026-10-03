//! Matrix multiplication dispatch.
//!
//! All shapes follow row-major convention: A[M, K] × B[K, N] → C[M, N].
//! GPU path dispatches via spite-dispatch; CPU path uses a scalar fallback.

use crate::ComputeError;

/// General matrix multiply: C = A × B (F32 scalars, O(MNK) reference impl).
///
/// `a`: `[m, k]` row-major
/// `b`: `[k, n]` row-major
/// `c`: `[m, n]` row-major, zeroed by caller
pub fn gemm_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, k: usize, n: usize,
) -> Result<(), ComputeError> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(ComputeError::ShapeMismatch(
            format!("gemm: A[{m},{k}] × B[{k},{n}] → C[{m},{n}]")
        ));
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    Ok(())
}

/// Quantized matmul: dequantize `b` on the fly then gemm.
///
/// `a`:  `[m, k]` F32 activations
/// `b`:  quantized weight `[k, n]` (any supported SpiteType)
/// `c`:  `[m, n]` F32 output
pub fn gemm_quantized(
    _a: &[f32],
    _b_raw: &[u8],
    _b_kind: spite_abi::SpiteType,
    _c: &mut [f32],
    _m: usize, _k: usize, _n: usize,
) -> Result<(), ComputeError> {
    // TODO: call dequant::dequant_to_f32(b_raw, b_kind) → temp_f32
    //       then gemm_f32(a, &temp_f32, c, m, k, n)
    Err(ComputeError::UnsupportedDtype)
}
