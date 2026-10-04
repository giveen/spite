//! DeepSeek V4 — GGUF arch `deepseek4`.
//!
//! Hyper-connection transformer (ported from llama.cpp `models/deepseek4.cpp`):
//! each token carries `hc` residual streams mixed per layer by learned
//! pre/post/comb weights (Sinkhorn-normalized), MLA attention with grouped
//! low-rank output, and fine-grained sqrt-softplus MoE.
//!
//! Sparse layers (nonzero compress ratios: indexer/CSA/HCA states) are NOT
//! ported; they fail loudly naming the missing piece.

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_loader::GgufModel;
use spite_rope::rope_range;

use crate::dense::{DenseWeights, Weight, matvec, rmsnorm};
use crate::moe::{GatingFunc, MoeActivation, MoeGating, moe_forward};
use crate::{ModelArch, ModelConfig, ModelError};

pub struct DeepSeekV4 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    kv: RwLock<Vec<Vec<Vec<f32>>>>,
}

impl DeepSeekV4 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv: RwLock::new(Vec::new()),
        }
    }
}

impl ModelArch for DeepSeekV4 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        for name in model.tensor_names() {
            if name.contains("tid2eid") {
                return Err(ModelError::Forward(
                    "hash expert selection (tid2eid) is not yet supported".into(),
                ));
            }
        }
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    fn reset_cache(&self) {
        if let Ok(mut kv) = self.kv.write() {
            kv.clear();
        }
    }

    fn forward(
        &self,
        tokens: &[u32],
        logits_out: &mut [f32],
        ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let Some(w) = &self.weights else {
            return Err(ModelError::Forward("load_weights not called".into()));
        };
        let cfg = &self.config;
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let head_dim = d / n_heads.max(1);
        let vocab = cfg.vocab_size;
        if logits_out.len() != tokens.len() * vocab {
            return Err(ModelError::Forward("logits_out shape mismatch".into()));
        }
        let hc = cfg.hc_mult.max(1);
        // ponytail: O(ctx²) scalar CPU path; GPU kernels own speed.
        let n_rot = cfg.rope_dim_count.max(1).min(head_dim);
        let nope = head_dim - n_rot;

        let embd = w.get("token_embd.weight")?;
        let out_norm = w.get("output_norm.weight")?;
        let out_w = w.get("output.weight")?;
        let hc_head_fn = w.get("output_hc_fn.weight")?;
        let hc_head_base = w.get("output_hc_base.weight")?;
        let hc_head_scale = w.get("output_hc_scale.weight")?;

        let mut kv = self
            .kv
            .write()
            .map_err(|_| ModelError::Forward("kv lock".into()))?;
        while kv.len() < cfg.n_layers {
            kv.push(Vec::new());
        }

        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            let e = &embd.data[tok as usize * d..(tok as usize + 1) * d];
            let mut streams = vec![e.to_vec(); hc];

            for layer in 0..cfg.n_layers {
                let b = format!("blk.{layer}");
                if cfg.compress_ratios.get(layer).copied().unwrap_or(0) != 0 {
                    return Err(ModelError::Forward(format!(
                        "sparse compressed layer {layer} (indexer/CSA/HCA) is not yet supported"
                    )));
                }
                let w_attn_norm = w.get(&format!("{b}.attn_norm.weight"))?;
                let w_ffn_norm = w.get(&format!("{b}.ffn_norm.weight"))?;
                let w_hc_attn_fn = w.get(&format!("{b}.hc_attn_fn.weight"))?;
                let w_hc_attn_base = w.get(&format!("{b}.hc_attn_base.weight"))?;
                let w_hc_attn_scale = w.get(&format!("{b}.hc_attn_scale.weight"))?;
                let w_hc_ffn_fn = w.get(&format!("{b}.hc_ffn_fn.weight"))?;
                let w_hc_ffn_base = w.get(&format!("{b}.hc_ffn_base.weight"))?;
                let w_hc_ffn_scale = w.get(&format!("{b}.hc_ffn_scale.weight"))?;

                // Attention block.
                let (pre, post, comb) = hc_mixes(
                    w_hc_attn_fn,
                    w_hc_attn_scale,
                    w_hc_attn_base,
                    &streams,
                    cfg,
                    hc,
                )?;
                let mut attn_in = hc_pre(&streams, &pre);
                let mut n = vec![0f32; d];
                rmsnorm(&attn_in, &w_attn_norm.data, cfg.norm_eps, &mut n);
                attn_in = mla_attn(w, &b, &n, pos, layer, cfg, head_dim, nope, n_rot, &mut kv)?;
                streams = hc_post(&attn_in, &streams, &post, &comb, d, hc);

                // FFN block.
                let (pre, post, comb) = hc_mixes(
                    w_hc_ffn_fn,
                    w_hc_ffn_scale,
                    w_hc_ffn_base,
                    &streams,
                    cfg,
                    hc,
                )?;
                let mut ffn_in = hc_pre(&streams, &pre);
                let mut fn_ = vec![0f32; d];
                rmsnorm(&ffn_in, &w_ffn_norm.data, cfg.norm_eps, &mut fn_);
                ffn_in = moe_ffn(w, &b, &fn_, cfg)?;
                streams = hc_post(&ffn_in, &streams, &post, &comb, d, hc);
            }

            // Head: mix streams, norm, logits.
            let flat: Vec<f32> = streams.concat();
            let mixes = hc_matvec(hc_head_fn, &flat)?;
            let pre =
                hc_affine_sigmoid(&mixes, hc_head_scale, hc_head_base, 0, hc, 1.0, cfg.hc_eps);
            let mut head = hc_pre(&streams, &pre);
            let mut hn = vec![0f32; d];
            rmsnorm(&head, &out_norm.data, cfg.norm_eps, &mut hn);
            head = hn;
            matvec(out_w, &head, &mut logits_out[ti * vocab..(ti + 1) * vocab])?;
        }
        Ok(())
    }
}

/// Hyper-connection mixes: rmsnorm flat streams → fn proj → split
/// into pre/post/comb with affine+sigmoid(+Sinkhorn for comb).
type HcMixes = (Vec<f32>, Vec<f32>, Vec<f32>);

fn hc_mixes(
    w_fn: &Weight,
    w_scale: &Weight,
    w_base: &Weight,
    streams: &[Vec<f32>],
    cfg: &ModelConfig,
    hc: usize,
) -> Result<HcMixes, ModelError> {
    let flat: Vec<f32> = streams.concat();
    let mut normed = vec![0f32; flat.len()];
    rmsnorm_flat(&flat, cfg.norm_eps, &mut normed);
    let mixes = hc_matvec(w_fn, &normed)?;
    let (pre, post, comb) = (
        hc_affine_sigmoid(&mixes, w_scale, w_base, 0, hc, 1.0, cfg.hc_eps),
        hc_affine_sigmoid(&mixes, w_scale, w_base, hc, hc, 2.0, 0.0),
        {
            let raw = hc_affine(&mixes, w_scale, w_base, 2 * hc, hc * hc);
            sinkhorn(&raw, hc, cfg.hc_sinkhorn_iters.max(1), cfg.hc_eps)
        },
    );
    Ok((pre, post, comb))
}

fn hc_matvec(w: &Weight, x: &[f32]) -> Result<Vec<f32>, ModelError> {
    let rows = w.ne[1].max(1) as usize;
    let cols = w.ne[0].max(1) as usize;
    if x.len() != cols || w.data.len() != rows * cols {
        return Err(ModelError::Forward("hc matvec shape mismatch".into()));
    }
    let mut out = vec![0f32; rows];
    for (r, o) in out.iter_mut().enumerate() {
        let mut acc = 0f32;
        for (c, &xv) in x.iter().enumerate() {
            acc += w.data[c * rows + r] * xv;
        }
        *o = acc;
    }
    Ok(out)
}

fn rmsnorm_flat(x: &[f32], eps: f32, out: &mut [f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v * scale;
    }
}

/// Affine slice + sigmoid (+ optional scale/eps shift).
fn hc_affine_sigmoid(
    mixes: &[f32],
    scale: &Weight,
    base: &Weight,
    off: usize,
    n: usize,
    mul: f32,
    eps: f32,
) -> Vec<f32> {
    mixes[off..off + n]
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let s = scale.data.get(i).copied().unwrap_or(1.0);
            let b = base.data.get(off + i).copied().unwrap_or(0.0);
            (m * s + b).sigmoid() * mul + eps
        })
        .collect()
}

trait Sigmoid {
    fn sigmoid(self) -> f32;
}
impl Sigmoid for f32 {
    fn sigmoid(self) -> f32 {
        1.0 / (1.0 + (-self).exp())
    }
}

fn hc_affine(mixes: &[f32], scale: &Weight, base: &Weight, off: usize, n: usize) -> Vec<f32> {
    mixes[off..off + n]
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let s = scale.data.get(2).copied().unwrap_or(1.0);
            let b = base.data.get(off + i).copied().unwrap_or(0.0);
            m * s + b
        })
        .collect()
}

/// Weighted stream sum: out = Σ_h streams[h]·pre[h].
fn hc_pre(streams: &[Vec<f32>], pre: &[f32]) -> Vec<f32> {
    let d = streams[0].len();
    let mut out = vec![0f32; d];
    for (s, &p) in streams.iter().zip(pre.iter()) {
        for (o, &v) in out.iter_mut().zip(s.iter()) {
            *o += v * p;
        }
    }
    out
}

/// out[dst] = x·post[dst] + Σ_src residual[src]·comb[dst·hc+src].
fn hc_post(
    x: &[f32],
    residual: &[Vec<f32>],
    post: &[f32],
    comb: &[f32],
    d: usize,
    hc: usize,
) -> Vec<Vec<f32>> {
    let _ = d;
    (0..hc)
        .map(|dst| {
            let mut cur: Vec<f32> = x.iter().map(|&v| v * post[dst]).collect();
            for src in 0..hc {
                let c = comb[dst * hc + src];
                for (o, &v) in cur.iter_mut().zip(residual[src].iter()) {
                    *o += v * c;
                }
            }
            cur
        })
        .collect()
}

/// Sinkhorn: row softmax, +eps, alternate col/row normalization.
fn sinkhorn(raw: &[f32], hc: usize, iters: usize, eps: f32) -> Vec<f32> {
    let mut m = vec![0f32; hc * hc];
    for r in 0..hc {
        let max = raw[r * hc..(r + 1) * hc]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for c in 0..hc {
            let e = (raw[r * hc + c] - max).exp() + eps;
            m[r * hc + c] = e;
            sum += e;
        }
        for c in 0..hc {
            m[r * hc + c] /= sum;
        }
    }
    for _ in 0..iters.saturating_sub(1) {
        // Rows.
        for r in 0..hc {
            let sum: f32 = m[r * hc..(r + 1) * hc].iter().sum::<f32>() + eps;
            for c in 0..hc {
                m[r * hc + c] /= sum;
            }
        }
        // Cols.
        for c in 0..hc {
            let sum: f32 = (0..hc).map(|r| m[r * hc + c]).sum::<f32>() + eps;
            for r in 0..hc {
                m[r * hc + c] /= sum;
            }
        }
    }
    // Final column normalization (llama.cpp does cols first, then iters-1 pairs).
    for c in 0..hc {
        let sum: f32 = (0..hc).map(|r| m[r * hc + c]).sum::<f32>() + eps;
        for r in 0..hc {
            m[r * hc + c] /= sum;
        }
    }
    m
}

/// Uncompressed MLA trunk: low-rank Q + shared compressed KV, rope on the
/// first `n_rot` dims, derope after attention, grouped low-rank output.
#[allow(clippy::too_many_arguments)]
fn mla_attn(
    w: &DenseWeights,
    b: &str,
    n: &[f32],
    pos: usize,
    layer: usize,
    cfg: &ModelConfig,
    head_dim: usize,
    nope: usize,
    n_rot: usize,
    kv: &mut [Vec<Vec<f32>>],
) -> Result<Vec<f32>, ModelError> {
    let d = cfg.d_model;
    let n_heads = cfg.n_heads;

    let wqa = w.get(&format!("{b}.attn_q_a.weight"))?;
    let wqan = w.get(&format!("{b}.attn_q_a_norm.weight"))?;
    let wqb = w.get(&format!("{b}.attn_q_b.weight"))?;
    let mut qr = vec![0f32; wqa.ne[1] as usize];
    matvec(wqa, n, &mut qr)?;
    let mut qrn = vec![0f32; qr.len()];
    rmsnorm(&qr, &wqan.data, cfg.norm_eps, &mut qrn);
    let mut q = vec![0f32; n_heads * head_dim];
    matvec(wqb, &qrn, &mut q)?;
    // Plain per-head RMS (deepseek4 applies rms after up-proj).
    let ones = vec![1f32; head_dim];
    for h in 0..n_heads {
        let mut nq = vec![0f32; head_dim];
        rmsnorm(
            &q[h * head_dim..(h + 1) * head_dim],
            &ones,
            cfg.norm_eps,
            &mut nq,
        );
        q[h * head_dim..(h + 1) * head_dim].copy_from_slice(&nq);
    }
    rope_range(&mut q, pos as u32, nope, n_rot, cfg.rope_theta, false);

    let wkv = w.get(&format!("{b}.attn_kv.weight"))?;
    let wkn = w.get(&format!("{b}.attn_kv_norm.weight"))?;
    let mut kv_raw = vec![0f32; wkv.ne[1] as usize];
    matvec(wkv, n, &mut kv_raw)?;
    let mut kvn = vec![0f32; head_dim];
    rmsnorm(&kv_raw, &wkn.data, cfg.norm_eps, &mut kvn);
    rope_range(&mut kvn, pos as u32, nope, n_rot, cfg.rope_theta, false);
    kv[layer].push(kvn);

    let n_prev = kv[layer].len();
    let mut kk = Vec::with_capacity(n_prev * n_heads * head_dim);
    let mut vv = Vec::with_capacity(n_prev * n_heads * head_dim);
    for row in kv[layer].iter() {
        for _ in 0..n_heads {
            kk.extend_from_slice(row);
            vv.extend_from_slice(row);
        }
    }
    let mut qq = Vec::with_capacity(n_heads * head_dim);
    for h in 0..n_heads {
        qq.extend_from_slice(&q[h * head_dim..(h + 1) * head_dim]);
    }
    let sinks = w
        .get(&format!("{b}.attn_sinks.weight"))
        .ok()
        .map(|t| t.data.clone());
    let mut attn_cfg = FlashAttnConfig::new(1, n_prev, n_heads, n_heads, head_dim);
    attn_cfg.sinks = sinks;
    let mut attn_out = vec![0f32; n_heads * head_dim];
    scalar_attention(&qq, &kk, &vv, &mut attn_out, &attn_cfg)
        .map_err(|e| ModelError::Forward(format!("mla attn: {e}")))?;
    // Derope the output (rope_ext_back over the rope section).
    rope_range(&mut attn_out, pos as u32, nope, n_rot, cfg.rope_theta, true);

    // Grouped low-rank output projection.
    let groups = cfg.o_group_count.max(1);
    let o_rank = cfg.o_lora_rank.max(1);
    let wo_a = w.get(&format!("{b}.attn_out_a.weight"))?;
    let wo_b = w.get(&format!("{b}.attn_out_b.weight"))?;
    let group_dim = n_heads * head_dim / groups;
    let mut oa = vec![0f32; o_rank * groups];
    for g in 0..groups {
        // wo_a is [group_dim, o_rank, groups]: slice group g.
        let stride = group_dim * o_rank;
        let slice = &wo_a.data[g * stride..(g + 1) * stride];
        let head = &attn_out[g * group_dim..(g + 1) * group_dim];
        let mut og = vec![0f32; o_rank];
        for (r, o) in og.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (c, &xv) in head.iter().enumerate() {
                acc += slice[c * o_rank + r] * xv;
            }
            *o = acc;
        }
        oa[g * o_rank..(g + 1) * o_rank].copy_from_slice(&og);
    }
    let mut proj = vec![0f32; d];
    matvec(wo_b, &oa, &mut proj)?;
    Ok(proj)
}

/// Fine-grained MoE with sqrt-softplus gating + shared experts.
fn moe_ffn(
    w: &DenseWeights,
    b: &str,
    n: &[f32],
    cfg: &ModelConfig,
) -> Result<Vec<f32>, ModelError> {
    let w_inp = w.get(&format!("{b}.ffn_gate_inp.weight"))?;
    let w_up = w.get(&format!("{b}.ffn_up_exps.weight"))?;
    let w_gate = w.get(&format!("{b}.ffn_gate_exps.weight"))?;
    let w_down = w.get(&format!("{b}.ffn_down_exps.weight"))?;
    let mut logits = vec![0f32; cfg.n_expert.max(1)];
    matvec(w_inp, n, &mut logits)?;
    let bias = w
        .get(&format!("{b}.ffn_exp_probs_b"))
        .ok()
        .map(|t| t.data.clone());
    let ff_exp = w_up.ne[1] as usize;
    let gating = match cfg.expert_gating_func {
        3 => MoeGating::Standard {
            func: GatingFunc::Softmax,
            norm_weights: cfg.expert_weights_norm,
        },
        4 | 0 => MoeGating::Standard {
            func: GatingFunc::SqrtSoftplus,
            norm_weights: cfg.expert_weights_norm,
        },
        _ => MoeGating::Standard {
            func: GatingFunc::Sigmoid,
            norm_weights: cfg.expert_weights_norm,
        },
    };
    let mut out = moe_forward(
        n,
        &logits,
        bias.as_deref(),
        w_up,
        w_gate,
        w_down,
        cfg.n_expert_used,
        ff_exp,
        cfg.expert_weights_scale,
        gating,
        MoeActivation::SwiGlu { clamp: 0.0 },
    )?;
    // Shared experts (plain SwiGLU at full width).
    if cfg.n_expert_shared > 0
        && let (Ok(wsu), Ok(wsg), Ok(wsd)) = (
            w.get(&format!("{b}.ffn_up_shexp.weight")),
            w.get(&format!("{b}.ffn_gate_shexp.weight")),
            w.get(&format!("{b}.ffn_down_shexp.weight")),
        )
    {
        let se = wsu.ne[1] as usize;
        let mut g = vec![0f32; se];
        let mut u = vec![0f32; se];
        matvec(wsg, n, &mut g)?;
        matvec(wsu, n, &mut u)?;
        for (gg, &uu) in g.iter_mut().zip(u.iter()) {
            *gg = *gg / (1.0 + (-*gg).exp()) * uu;
        }
        let mut sh = vec![0f32; out.len()];
        matvec(wsd, &g, &mut sh)?;
        for (o, &s) in out.iter_mut().zip(sh.iter()) {
            *o += s;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny trunk: hc=2, 1 layer, MLA (rank 4, rope 2), 2 experts top-1.
    fn tiny_weights() -> (ModelConfig, DenseWeights) {
        let d = 8usize;
        let hc = 2usize;
        let nh = 2usize;
        let hd = 4usize;
        let rot = 2usize;
        let ff = 8usize;
        let v = 8usize;
        let ne = 2usize;
        let qrank = 4usize;
        let orank = 2usize;

        let mut map: HashMap<String, (Vec<f32>, [u32; 4])> = HashMap::new();
        let w = |rows: usize, cols: usize, fill: f32| {
            (vec![fill; rows * cols], [cols as u32, rows as u32, 1, 1])
        };
        let ones = |n: usize| (vec![1.0; n], [n as u32, 1, 1, 1]);
        map.insert("token_embd.weight".into(), w(v, d, 0.1));
        map.insert("output_norm.weight".into(), ones(d));
        map.insert("output.weight".into(), w(v, d, 0.1));
        map.insert("output_hc_fn.weight".into(), w((2 + hc) * hc, hc * d, 0.05));
        map.insert(
            "output_hc_scale.weight".into(),
            (vec![1.0; 3], [3, 1, 1, 1]),
        );
        map.insert(
            "output_hc_base.weight".into(),
            (
                vec![0.0; 2 * hc + hc * hc],
                [(2 * hc + hc * hc) as u32, 1, 1, 1],
            ),
        );

        let b = "blk.0";
        for (k, val) in [
            ("attn_norm", ones(d)),
            ("ffn_norm", ones(d)),
            ("attn_q_a", w(qrank, d, 0.05)),
            ("attn_q_a_norm", ones(qrank)),
            ("attn_q_b", w(nh * hd, qrank, 0.05)),
            ("attn_kv", w(hd, d, 0.05)),
            ("attn_kv_norm", ones(hd)),
            ("attn_sinks", (vec![0.0; nh], [nh as u32, 1, 1, 1])),
            (
                "attn_out_a",
                (
                    vec![0.05; nh * hd * orank],
                    [(nh * hd) as u32, orank as u32, 1, 1],
                ),
            ),
            ("attn_out_b", w(d, orank, 0.05)),
            ("hc_attn_fn", w((2 + hc) * hc, hc * d, 0.05)),
            ("hc_attn_scale", (vec![1.0; 3], [3, 1, 1, 1])),
            (
                "hc_attn_base",
                (
                    vec![0.0; 2 * hc + hc * hc],
                    [(2 * hc + hc * hc) as u32, 1, 1, 1],
                ),
            ),
            ("hc_ffn_fn", w((2 + hc) * hc, hc * d, 0.05)),
            ("hc_ffn_scale", (vec![1.0; 3], [3, 1, 1, 1])),
            (
                "hc_ffn_base",
                (
                    vec![0.0; 2 * hc + hc * hc],
                    [(2 * hc + hc * hc) as u32, 1, 1, 1],
                ),
            ),
            ("ffn_gate_inp", w(ne, d, 0.5)),
            ("ffn_exp_probs_b", (vec![0.0; ne], [ne as u32, 1, 1, 1])),
        ] {
            map.insert(format!("{b}.{k}.weight"), val);
        }
        let stack = |fill: f32| (vec![fill; ne * d * ff], [d as u32, ff as u32, ne as u32, 1]);
        map.insert("blk.0.ffn_up_exps.weight".into(), stack(0.05));
        map.insert("blk.0.ffn_gate_exps.weight".into(), stack(0.05));
        map.insert(
            "blk.0.ffn_down_exps.weight".into(),
            (vec![0.05; ne * ff * d], [ff as u32, d as u32, ne as u32, 1]),
        );
        map.insert("blk.0.ffn_up_shexp.weight".into(), w(ff, d, 0.05));
        map.insert("blk.0.ffn_gate_shexp.weight".into(), w(ff, d, 0.05));
        map.insert("blk.0.ffn_down_shexp.weight".into(), w(d, ff, 0.05));

        let cfg = ModelConfig {
            arch: "deepseek4".into(),
            n_layers: 1,
            n_heads: nh,
            n_kv_heads: 1,
            d_model: d,
            d_ffn: ff,
            vocab_size: v,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            hc_mult: hc,
            hc_eps: 1e-6,
            hc_sinkhorn_iters: 3,
            o_group_count: 1,
            o_lora_rank: orank,
            rope_dim_count: rot,
            n_expert: ne,
            n_expert_used: 1,
            n_expert_shared: 1,
            expert_gating_func: 4,
            ..Default::default()
        };
        (cfg, DenseWeights::from_map(map))
    }

    #[test]
    fn trunk_forward_finite_and_incremental() {
        let (cfg, weights) = tiny_weights();
        let mut model = DeepSeekV4::new(cfg.clone());
        model.weights = Some(weights);
        let ctx = SpiteCtx {
            n_ctx: 64,
            n_batch: 1,
            n_threads: 1,
            pos: 0,
            n_heads: 2,
            n_kv_heads: 1,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: std::ptr::null_mut(),
            scratchpad_bytes: 0,
        };
        let mut full = vec![0f32; 2 * cfg.vocab_size];
        model.forward(&[1, 2], &mut full, &ctx).unwrap();
        assert!(full.iter().all(|x| x.is_finite()));

        model.reset_cache();
        let mut pre = vec![0f32; cfg.vocab_size];
        let mut step = vec![0f32; cfg.vocab_size];
        model.forward(&[1], &mut pre, &ctx).unwrap();
        let mut ctx1 = ctx;
        ctx1.pos = 1;
        model.forward(&[2], &mut step, &ctx1).unwrap();
        assert!(full[..cfg.vocab_size] == pre[..]);
        assert!(full[cfg.vocab_size..] == step[..]);
    }
}
