//! Gated Delta Net linear attention (scalar CPU port).
//!
//! Exact math from llama.cpp `llm_build_delta_net_base::
//! build_delta_net_autoregressive` (decode path, one token). Used by hybrid
//! archs (Qwen3.5) for their recurrent layers.
//!
//! Per value-head (S = head dim), with decay `g`, mixing `beta`, and state
//! `M` (S×S, row-major, persistent across positions):
//! ```text
//! M   *= exp(g)                                   (forget)
//! sk    = Mᵀ k          (sk[s] = Σ_r M[r][s]·k[r])
//! d     = (v − sk) · beta                         (delta)
//! M    += k ⊗ d       (M[r][s] += k[r]·d[s])       (update)
//! o     = Mᵀ q          (o[s] = Σ_r M[r][s]·q[r])
//! ```
//! `q` is pre-scaled by 1/√S by the caller.

use crate::ComputeError;

/// One GDN step for a single value-head.
///
/// `state` is S×S row-major, updated in place. `g`/`beta` are scalars.
/// `q` must already be scaled by 1/√S.
pub fn gdn_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    out: &mut [f32],
) -> Result<(), ComputeError> {
    let s = q.len();
    if k.len() != s || v.len() != s || out.len() != s || state.len() != s * s {
        return Err(ComputeError::ShapeMismatch(
            "gdn_step shape mismatch".into(),
        ));
    }
    let decay = g.exp();
    for m in state.iter_mut() {
        *m *= decay;
    }
    // sk = Mᵀk; d = (v − sk)·beta
    let mut d = vec![0f32; s];
    for (ss, d_s) in d.iter_mut().enumerate() {
        let mut sk = 0f32;
        for r in 0..s {
            sk += state[r * s + ss] * k[r];
        }
        *d_s = (v[ss] - sk) * beta;
    }
    // M += k⊗d
    for r in 0..s {
        for (ss, &d_s) in d.iter().enumerate() {
            state[r * s + ss] += k[r] * d_s;
        }
    }
    // o = Mᵀq
    for (ss, o_s) in out.iter_mut().enumerate() {
        let mut acc = 0f32;
        for r in 0..s {
            acc += state[r * s + ss] * q[r];
        }
        *o_s = acc;
    }
    Ok(())
}

/// Depthwise 1D conv step over the trailing `hist` + current input.
///
/// `hist` holds the previous K−1 input vectors (oldest first); `kernel` is
/// [K, C] row-major. Returns the convolved current vector.
pub fn ssm_conv_step(hist: &[Vec<f32>], input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let c = input.len();
    let k = hist.len() + 1;
    let mut out = vec![0f32; c];
    for (cc, o) in out.iter_mut().enumerate() {
        let mut acc = 0f32;
        for (i, h) in hist.iter().enumerate() {
            acc += h[cc] * kernel[i * c + cc];
        }
        acc += input[cc] * kernel[(k - 1) * c + cc];
        *o = acc;
    }
    out
}

/// GDN L2 norm: `rms_norm(x, eps/n) / sqrt(n)` (llama.cpp `build_gdn_l2_norm`).
pub fn gdn_l2_norm(x: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len().max(1) as f32;
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / n;
    let scale = 1.0 / ((mean_sq + eps / n).sqrt() * n.sqrt());
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_matches_naive_reference() {
        // Two steps with decay 1 (g=0): state accumulates k⊗(v−Mᵀk),
        // output reads out through q. Cross-check against a direct
        // re-computation of the same update equations.
        let s = 4;
        let mut state = vec![0f32; s * s];
        let q = vec![0.5, -0.25, 0.125, 1.0];
        let k = vec![1.0, 0.5, -0.5, 0.25];
        let v = vec![0.25, 0.75, -0.125, 0.5];
        let mut out = vec![0f32; s];
        gdn_step(&mut state, &q, &k, &v, 0.0, 1.0, &mut out).unwrap();

        // Naive: M was zero, so sk=0, d=v, M=k⊗v, o=Mᵀq.
        let dot_qk: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
        for (ss, o_s) in out.iter().enumerate() {
            assert!((o_s - v[ss] * dot_qk).abs() < 1e-5, "{out:?}");
        }
        // Second step keeps state across calls (recurrence).
        let mut out2 = vec![0f32; s];
        gdn_step(&mut state, &q, &k, &v, 0.0, 1.0, &mut out2).unwrap();
        assert!(out2.iter().all(|x| x.is_finite()));
        assert_ne!(out, out2);
    }

    #[test]
    fn conv_step_slides_window() {
        // K=2 kernel averaging past+current per channel.
        let hist = vec![vec![1.0, 2.0]];
        let kernel = vec![0.5, 0.5, 0.5, 0.5];
        assert_eq!(ssm_conv_step(&hist, &[3.0, 4.0], &kernel), vec![2.0, 3.0]);
    }
}
