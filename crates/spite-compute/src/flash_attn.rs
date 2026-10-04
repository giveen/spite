//! Flash Attention 2 / 3 host-side dispatch stub.
//!
//! Flash Attention avoids materializing the full N×N attention matrix by
//! tiling the computation so that only O(block_size) rows are live in SRAM
//! at once. This allows O(N²/block_size) memory traffic instead of O(N²).
//!
//! In spite, the GPU-side implementation lives in a kernel .so loaded via
//! spite-dispatch. This module:
//!   1. Validates tensor shapes
//!   2. Selects the kernel variant (causal / non-causal, F16/BF16, page-table)
//!   3. Calls the kernel via the DispatchTable's `attention` slot
//!
//! When no Flash Attention kernel is available (e.g. CPU-only fallback),
//! this falls back to the O(N²) scalar implementation below.
//!
//! # Variants dispatched
//!
//! - FlashAttn2: standard causal/non-causal, O(N√N) SRAM
//! - FlashAttn3: Hopper H100/H200 specialisation (warp-specialised SM90+)
//! - PagedFlashAttn: paged block table support for serving

use crate::ComputeError;

/// Configuration for one attention call.
#[derive(Debug, Clone)]
pub struct FlashAttnConfig {
    /// Query length (tokens in current batch).
    pub n_q: usize,
    /// Key/value sequence length (context window).
    pub n_kv: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Softmax scaling factor (1/√head_dim unless overridden).
    pub scale: f32,
    /// Causal mask (lower-triangular): each query attends only to earlier KV.
    pub causal: bool,
    /// Per-head attention sinks (llama.cpp `attn_sinks`): an extra logit
    /// with zero value mass. None = no sinks.
    pub sinks: Option<Vec<f32>>,
}

impl FlashAttnConfig {
    pub fn new(
        n_q: usize,
        n_kv: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: true,
            sinks: None,
        }
    }
}

/// Scalar O(N²) fallback attention (no SRAM tiling).
/// Used when no GPU kernel is available.
///
/// `q`: `[n_q,  n_heads,    head_dim]` F32
/// `k`: `[n_kv, n_kv_heads, head_dim]` F32 (GQA: n_kv_heads ≤ n_heads)
/// `v`: `[n_kv, n_kv_heads, head_dim]` F32
/// `o`: `[n_q,  n_heads,    head_dim]` F32  ← output, pre-allocated
pub fn scalar_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    cfg: &FlashAttnConfig,
) -> Result<(), ComputeError> {
    let &FlashAttnConfig {
        n_q,
        n_kv,
        n_heads,
        n_kv_heads,
        head_dim,
        scale,
        causal,
        ..
    } = cfg;

    let kv_groups = n_heads / n_kv_heads; // GQA group size
    if n_heads % n_kv_heads != 0 {
        return Err(ComputeError::ShapeMismatch(format!(
            "n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}"
        )));
    }
    let expected_q = n_q * n_heads * head_dim;
    let expected_k = n_kv * n_kv_heads * head_dim;
    if q.len() != expected_q || k.len() != expected_k || v.len() != expected_k {
        return Err(ComputeError::ShapeMismatch("q/k/v length mismatch".into()));
    }
    if o.len() != expected_q {
        return Err(ComputeError::ShapeMismatch("output length mismatch".into()));
    }

    let mut scores = vec![0f32; n_kv];

    for qi in 0..n_q {
        for h in 0..n_heads {
            let kvh = h / kv_groups;

            // Compute raw attention scores: Q[qi,h] · K[j,kvh] for all j.
            for (j, s) in scores.iter_mut().enumerate() {
                if causal && j > qi {
                    *s = f32::NEG_INFINITY;
                    continue;
                }
                let q_base = (qi * n_heads + h) * head_dim;
                let k_base = (j * n_kv_heads + kvh) * head_dim;
                let dot: f32 = (0..head_dim).map(|d| q[q_base + d] * k[k_base + d]).sum();
                *s = dot * scale;
            }

            // Softmax over scores (+ sink logit with zero value mass).
            let sink = cfg
                .sinks
                .as_ref()
                .map(|s| s[h])
                .unwrap_or(f32::NEG_INFINITY);
            let max = scores.iter().cloned().fold(sink, f32::max);
            let mut sum = (sink - max).exp();
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let inv = 1.0 / sum.max(1e-30);
            for s in scores.iter_mut() {
                *s *= inv;
            }

            // Weighted sum over V.
            let o_base = (qi * n_heads + h) * head_dim;
            for d in 0..head_dim {
                let mut acc = 0f32;
                for (j, &sj) in scores.iter().enumerate() {
                    let v_base = (j * n_kv_heads + kvh) * head_dim;
                    acc += sj * v[v_base + d];
                }
                o[o_base + d] = acc;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_token_self_attn() {
        let cfg = FlashAttnConfig::new(1, 1, 2, 2, 4);
        let q = vec![1f32; 8];
        let k = vec![1f32; 8];
        let v = vec![1f32; 8];
        let mut o = vec![0f32; 8];
        scalar_attention(&q, &k, &v, &mut o, &cfg).unwrap();
        // With Q=K=V=ones, output should equal V (one token, attn weight = 1).
        assert!(o.iter().all(|&x| (x - 1.0).abs() < 1e-5), "{:?}", o);
    }
}
