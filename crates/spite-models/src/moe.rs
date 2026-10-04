//! Shared MoE feed-forward (ported from llama.cpp `build_moe_ffn`).
//!
//! LLaMA-4 semantics used here: select top-k experts by raw router logits,
//! weight by sigmoid, scale the input by the weight *before* the expert FFN
//! (`weight_before_ffn`), no post-weighting, no norm. Other gating modes
//! (softmax weights, norm, group selection) belong here when their archs
//! are ported — add variants, don't fork this file.

use crate::ModelError;
use crate::dense::{Weight, matvec};

/// Sigmoid-gated top-k MoE with SwiGLU experts.
///
/// `gate_inp`: [d → n_expert] router. Expert tensors are [d, ff, E] /
/// [ff, d, E] stacked (see `expert_view`). `w_scale` multiplies the
/// weights when nonzero and not one. `clamp` bounds up/silu when > 1e-6.
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
    let d = x.len();
    let mut logits = vec![0f32; n_expert];
    matvec(gate_inp, x, &mut logits)?;

    // Top-k by raw logit.
    let mut idx: Vec<usize> = (0..n_expert).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(n_used.min(n_expert));

    let mut out = vec![0f32; d];
    for &e in &idx {
        let mut w = 1.0 / (1.0 + (-logits[e]).exp());
        if w_scale != 0.0 && w_scale != 1.0 {
            w *= w_scale;
        }
        // Input scaled by weight before the expert FFN.
        let xw: Vec<f32> = x.iter().map(|&v| v * w).collect();
        let up = expert_view(up_exps, e, d, ff_exp);
        let gate = expert_view(gate_exps, e, d, ff_exp);
        let down = expert_view(down_exps, e, ff_exp, d);
        let mut up_out = vec![0f32; ff_exp];
        let mut gate_out = vec![0f32; ff_exp];
        matvec_raw(&up, ff_exp, d, &xw, &mut up_out)?;
        matvec_raw(&gate, ff_exp, d, &xw, &mut gate_out)?;
        for (u, &g) in up_out.iter_mut().zip(gate_out.iter()) {
            if clamp > 1e-6 {
                *u = u.clamp(-clamp, clamp);
            }
            let mut act = g / (1.0 + (-g).exp());
            if clamp > 1e-6 {
                act = act.min(clamp);
            }
            *u *= act;
        }
        let mut down_out = vec![0f32; d];
        matvec_raw(&down, d, ff_exp, &up_out, &mut down_out)?;
        for (o, &dd) in out.iter_mut().zip(down_out.iter()) {
            *o += dd;
        }
    }
    Ok(out)
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
