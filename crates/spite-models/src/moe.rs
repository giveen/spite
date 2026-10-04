//! Shared MoE feed-forward (ported from llama.cpp `build_moe_ffn`).
//!
//! Covers the gating modes encountered so far; add variants here when new
//! archs need them — never fork per-arch copies.

use crate::ModelError;
use crate::dense::{Weight, matvec};

/// Router gating modes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MoeGating {
    /// LLaMA-4: select top-k by raw logits, sigmoid weights, scale the
    /// input by the weight *before* the expert FFN, no post-weighting.
    #[default]
    SigmoidPrescaled,
    /// Standard: gate the logits, select top-k by probability, weight the
    /// expert outputs (optionally normalized).
    Standard {
        func: GatingFunc,
        norm_weights: bool,
    },
}

/// Probability function applied to router logits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GatingFunc {
    #[default]
    Sigmoid,
    Softmax,
    /// `sqrt(softplus(x))` — DeepSeek V4.
    SqrtSoftplus,
}

/// Expert FFN activation.
#[derive(Debug, Clone, Copy)]
pub enum MoeActivation {
    /// SwiGLU, optional symmetric clamp on `up` and ceiling on silu(gate).
    SwiGlu { clamp: f32 },
    /// SiTU: `β·tanh(g/β)·σ(g) · (λ·tanh(u/λ) or u)`.
    SiTu { beta: f32, linear_beta: f32 },
}

impl Default for MoeActivation {
    fn default() -> Self {
        Self::SwiGlu { clamp: 0.0 }
    }
}

/// Sigmoid-gated top-k MoE with SwiGLU experts (llama4 shorthand).
///
/// Kept for the llama4 call shape; forwards to [`moe_forward`].
#[allow(clippy::too_many_arguments)]
pub fn moe_swiglu(
    x: &[f32],
    gate_inp: &Weight,
    up_exps: &Weight,
    gate_exps: &Weight,
    down_exps: &Weight,
    n_expert: usize,
    n_used: usize,
    ff_exp: usize,
    w_scale: f32,
    clamp: f32,
) -> Result<Vec<f32>, ModelError> {
    let mut logits = vec![0f32; n_expert];
    matvec(gate_inp, x, &mut logits)?;
    moe_forward(
        x,
        &logits,
        None,
        up_exps,
        gate_exps,
        down_exps,
        n_used,
        ff_exp,
        w_scale,
        MoeGating::SigmoidPrescaled,
        MoeActivation::SwiGlu { clamp },
    )
}

/// Core MoE: route `logits` (precomputed by the caller), run the selected
/// experts over `x`, return the weighted sum.
///
/// `probs_bias` adds to the logits before gating (Kimi-style bias tuning).
#[allow(clippy::too_many_arguments)]
pub fn moe_forward(
    x: &[f32],
    logits: &[f32],
    probs_bias: Option<&[f32]>,
    up_exps: &Weight,
    gate_exps: &Weight,
    down_exps: &Weight,
    n_used: usize,
    ff_exp: usize,
    w_scale: f32,
    gating: MoeGating,
    activation: MoeActivation,
) -> Result<Vec<f32>, ModelError> {
    let d = x.len();
    let n_expert = logits.len();
    let biased: Vec<f32> = match probs_bias {
        Some(b) => logits.iter().zip(b.iter()).map(|(l, bb)| l + bb).collect(),
        None => logits.to_vec(),
    };

    // Selection scores and output weights per gating mode.
    let (order, weights): (Vec<usize>, Vec<f32>) = match gating {
        MoeGating::SigmoidPrescaled => {
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|&a, &b| {
                biased[b]
                    .partial_cmp(&biased[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(n_used.min(n_expert));
            let ws: Vec<f32> = idx
                .iter()
                .map(|&e| {
                    let mut w = 1.0 / (1.0 + (-biased[e]).exp());
                    if w_scale != 0.0 && w_scale != 1.0 {
                        w *= w_scale;
                    }
                    w
                })
                .collect();
            (idx, ws)
        }
        MoeGating::Standard { func, norm_weights } => {
            let probs: Vec<f32> = match func {
                GatingFunc::Sigmoid => biased.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect(),
                GatingFunc::Softmax => {
                    let max = biased.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = biased.iter().map(|&l| (l - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    exps.iter().map(|&e| e / sum.max(1e-30)).collect()
                }
                GatingFunc::SqrtSoftplus => {
                    biased.iter().map(|&l| l.ln_1p().exp().sqrt()).collect()
                }
            };
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(n_used.min(n_expert));
            let mut ws: Vec<f32> = idx.iter().map(|&e| probs[e]).collect();
            if norm_weights {
                let sum: f32 = ws.iter().sum();
                // Smallest positive F16: exact clamp from llama.cpp, keep full precision.
                #[allow(clippy::excessive_precision)]
                let sum = sum.max(6.103515625e-5);
                for w in ws.iter_mut() {
                    *w /= sum;
                }
            }
            if w_scale != 0.0 && w_scale != 1.0 {
                for w in ws.iter_mut() {
                    *w *= w_scale;
                }
            }
            (idx, ws)
        }
    };

    let prescaled = matches!(gating, MoeGating::SigmoidPrescaled);
    let mut out = vec![0f32; d];
    for (k, &e) in order.iter().enumerate() {
        let w = weights[k];
        let xw: Vec<f32> = if prescaled {
            x.iter().map(|&v| v * w).collect()
        } else {
            x.to_vec()
        };
        let up = expert_view(up_exps, e, d, ff_exp);
        let gate = expert_view(gate_exps, e, d, ff_exp);
        let down = expert_view(down_exps, e, ff_exp, d);
        let mut up_out = vec![0f32; ff_exp];
        let mut gate_out = vec![0f32; ff_exp];
        matvec_raw(&up, ff_exp, d, &xw, &mut up_out)?;
        matvec_raw(&gate, ff_exp, d, &xw, &mut gate_out)?;
        apply_activation(&mut up_out, &gate_out, activation);
        let mut down_out = vec![0f32; d];
        matvec_raw(&down, d, ff_exp, &up_out, &mut down_out)?;
        // Standard mode weights the expert outputs; prescaled already did.
        let scale = if prescaled { 1.0 } else { w };
        for (o, &dd) in out.iter_mut().zip(down_out.iter()) {
            *o += dd * scale;
        }
    }
    Ok(out)
}

fn apply_activation(up: &mut [f32], gate: &[f32], activation: MoeActivation) {
    match activation {
        MoeActivation::SwiGlu { clamp } => {
            for (u, &g) in up.iter_mut().zip(gate.iter()) {
                if clamp > 1e-6 {
                    *u = u.clamp(-clamp, clamp);
                }
                let mut act = g / (1.0 + (-g).exp());
                if clamp > 1e-6 {
                    act = act.min(clamp);
                }
                *u *= act;
            }
        }
        MoeActivation::SiTu { beta, linear_beta } => {
            for (u, &g) in up.iter_mut().zip(gate.iter()) {
                let a = beta * (g / beta).tanh() / (1.0 + (-g).exp());
                let uu = if linear_beta > 0.0 {
                    linear_beta * (*u / linear_beta).tanh()
                } else {
                    *u
                };
                *u = a * uu;
            }
        }
    }
}

/// Row-major view of expert `e` from a stacked [cols, rows, E] tensor.
fn expert_view(w: &Weight, e: usize, cols: usize, rows: usize) -> Vec<f32> {
    let stride = cols * rows;
    w.data[e * stride..(e + 1) * stride].to_vec()
}

fn matvec_raw(
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), ModelError> {
    if x.len() != cols || out.len() != rows || w.len() != rows * cols {
        return Err(ModelError::Forward("moe matvec shape mismatch".into()));
    }
    for (r, o) in out.iter_mut().enumerate() {
        let mut acc = 0f32;
        for (c, &xv) in x.iter().enumerate() {
            acc += w[c * rows + r] * xv;
        }
        *o = acc;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight(data: Vec<f32>, cols: usize, rows: usize) -> Weight {
        Weight {
            data,
            ne: [cols as u32, rows as u32, 1, 1],
        }
    }

    fn stacked(experts: Vec<Vec<f32>>, cols: usize, rows: usize) -> Weight {
        assert!(experts.iter().all(|e| e.len() == cols * rows));
        Weight {
            data: experts.concat(),
            ne: [cols as u32, rows as u32, 1, 1],
        }
    }

    #[test]
    fn moe_selects_topk_and_weights() {
        // 2 experts, top-1. Router strongly prefers expert 1.
        let d = 2;
        let ff = 2;
        let gate_inp = weight(vec![0.0, 10.0, 0.0, 10.0], d, 2);
        let ident = vec![1.0, 0.0, 0.0, 1.0];
        let zeros = vec![0.0; 4];
        let up = stacked(vec![zeros.clone(), ident.clone()], d, ff);
        let gate = stacked(vec![zeros.clone(), vec![10.0; 4]], d, ff);
        let down = stacked(vec![zeros, ident], ff, d);
        let x = vec![1.0, 2.0];
        let out = moe_swiglu(&x, &gate_inp, &up, &gate, &down, 2, 1, ff, 0.0, 0.0).unwrap();
        // Expert 1 selected with weight≈1: out ≈ down(silu(gate·x)·(up·x)).
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out[0].abs() + out[1].abs() > 0.5, "{out:?}");
    }
}
