//! Zhipu GLM with Dynamic Sparse Attention — GGUF arch `glm-dsa`.
//!
//! MLA (absorbed, DeepSeek-style) + lightning indexer: each full indexer
//! layer scores all cached positions (ReLU-gated, Hadamard-rotated),
//! attends only the top-k; shared layers reuse the previous top-k.
//! Ported from llama.cpp `models/glm-dsa.cpp` (unfused paths).

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;
use spite_rope::{RopeConfig, apply_rope, rope_range};

use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::moe::{GatingFunc, MoeActivation, MoeGating, moe_forward};
use crate::{ModelArch, ModelConfig, ModelError};

pub struct GlmDsa {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    state: RwLock<GlmState>,
}

#[derive(Default)]
struct GlmState {
    /// MLA cache per layer per pos: [kv_lora | k_pe].
    mla: Vec<Vec<Vec<f32>>>,
    /// Indexer keys per layer per pos.
    idx: Vec<Vec<Vec<f32>>>,
}

impl GlmDsa {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            state: RwLock::new(GlmState::default()),
        }
    }

    fn is_full_indexer(&self, layer: usize) -> bool {
        self.config
            .indexer_types
            .get(layer)
            .copied()
            .unwrap_or(true)
    }
}

impl ModelArch for GlmDsa {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    fn reset_cache(&self) {
        if let Ok(mut s) = self.state.write() {
            s.mla.clear();
            s.idx.clear();
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
        let d_ffn = cfg.d_ffn;
        let vocab = cfg.vocab_size;
        if logits_out.len() != tokens.len() * vocab {
            return Err(ModelError::Forward("logits_out shape mismatch".into()));
        }
        // ponytail: O(ctx²) scalar CPU path; GPU kernels own speed.
        let rank = cfg.kv_lora_rank.max(1);
        let head_k = cfg.key_length_mla.max(1);
        let head_v = cfg.value_length_mla.max(1);
        let n_rot = cfg.rope_dim_count.min(head_k).max(1);
        let nope = head_k - n_rot;
        let ihd = cfg.indexer_head_size.max(1);
        let ih = cfg.indexer_n_head.max(1);
        let top_k = cfg.indexer_top_k.max(1);
        let hadamard = hadamard(ihd)?;

        let embd = w.get("token_embd.weight")?;
        let out_norm = w.get("output_norm.weight")?;
        let out_w = w.get("output.weight")?;

        let mut state = self
            .state
            .write()
            .map_err(|_| ModelError::Forward("state lock".into()))?;
        while state.mla.len() < cfg.n_layers {
            state.mla.push(Vec::new());
            state.idx.push(Vec::new());
        }

        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            let mut h = vec![0f32; d];
            h.copy_from_slice(&embd.data[tok as usize * d..(tok as usize + 1) * d]);
            let mut prev_top_k: Option<Vec<usize>> = None;

            for layer in 0..cfg.n_layers {
                let b = format!("blk.{layer}");
                let w_norm = w.get(&format!("{b}.attn_norm.weight"))?;
                let w_ffn_norm = w.get(&format!("{b}.ffn_norm.weight"))?;
                let mut n = vec![0f32; d];
                rmsnorm(&h, &w_norm.data, cfg.norm_eps, &mut n);

                // Query low-rank + norm (shared by MLA and indexer).
                let wqa = w.get(&format!("{b}.attn_q_a.weight"))?;
                let wqan = w.get(&format!("{b}.attn_q_a_norm.weight"))?;
                let mut qr = vec![0f32; wqa.ne[1] as usize];
                matvec(wqa, &n, &mut qr)?;
                let mut qrn = vec![0f32; qr.len()];
                rmsnorm(&qr, &wqan.data, cfg.norm_eps, &mut qrn);

                // Indexer top-k (full layers compute, shared reuse).
                let topk: Vec<usize> = if self.is_full_indexer(layer) {
                    let tk = indexer_top_k(
                        w,
                        &b,
                        &qrn,
                        &n,
                        pos,
                        &mut state,
                        layer,
                        ihd,
                        ih,
                        top_k,
                        &hadamard,
                        cfg.rope_theta,
                    )?;
                    prev_top_k = Some(tk.clone());
                    tk
                } else if let Some(tk) = prev_top_k.clone() {
                    tk
                } else {
                    return Err(ModelError::Forward(
                        "shared indexer layer without previous top-k".into(),
                    ));
                };

                // Absorbed MLA with sparse mask.
                let wqb = w.get(&format!("{b}.attn_q_b.weight"))?;
                let mut q = vec![0f32; n_heads * head_k];
                matvec(wqb, &qrn, &mut q)?;
                let mut q_nope = vec![0f32; n_heads * nope];
                let mut q_pe = vec![0f32; n_heads * n_rot];
                for hh in 0..n_heads {
                    q_nope[hh * nope..(hh + 1) * nope]
                        .copy_from_slice(&q[hh * head_k..hh * head_k + nope]);
                    q_pe[hh * n_rot..(hh + 1) * n_rot]
                        .copy_from_slice(&q[hh * head_k + nope..(hh + 1) * head_k]);
                }
                let rope = RopeConfig {
                    head_dim: n_rot,
                    theta: cfg.rope_theta,
                    ..Default::default()
                };
                apply_rope(&mut q_pe, pos as u32, &rope)
                    .map_err(|e| ModelError::Forward(format!("rope: {e}")))?;

                let wkv_a = w.get(&format!("{b}.attn_kv_a_mqa.weight"))?;
                let kv_rows = wkv_a.ne[1] as usize;
                let mut kv_full = vec![0f32; kv_rows];
                matvec(wkv_a, &n, &mut kv_full)?;
                let (kv_raw, k_pe_raw) = kv_full.split_at(rank.min(kv_rows));
                let wkn = w.get(&format!("{b}.attn_kv_a_norm.weight"))?;
                let mut kvn = vec![0f32; kv_raw.len()];
                rmsnorm(kv_raw, &wkn.data, cfg.norm_eps, &mut kvn);
                let mut kpe = k_pe_raw.to_vec();
                let kpe_len = kpe.len();
                rope_range(
                    &mut kpe,
                    pos as u32,
                    0,
                    n_rot.min(kpe_len),
                    cfg.rope_theta,
                    false,
                );
                let mut row = Vec::with_capacity(kvn.len() + kpe.len());
                row.extend_from_slice(&kvn);
                row.extend_from_slice(&kpe);
                state.mla[layer].push(row);

                // Absorb q_nope into rank space via wk_b, then attend.
                let wkb = w.get(&format!("{b}.attn_k_b.weight"))?;
                let mut q_abs = vec![0f32; n_heads * rank];
                for hh in 0..n_heads {
                    // wk_b is [nope, rank, n_heads]: per-head slice.
                    let kh = &wkb.data[hh * nope * rank..(hh + 1) * nope * rank];
                    let mut out = vec![0f32; rank];
                    for (r, o) in out.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for (c, &xv) in q_nope[hh * nope..(hh + 1) * nope].iter().enumerate() {
                            acc += kh[c * rank + r] * xv;
                        }
                        *o = acc;
                    }
                    q_abs[hh * rank..(hh + 1) * rank].copy_from_slice(&out);
                }
                // Q rows: [q_abs (rank) | q_pe (rope)] per head.
                let dim = rank + n_rot;
                let mut qq = Vec::with_capacity(n_heads * dim);
                for hh in 0..n_heads {
                    qq.extend_from_slice(&q_abs[hh * rank..(hh + 1) * rank]);
                    qq.extend_from_slice(&q_pe[hh * n_rot..(hh + 1) * n_rot]);
                }
                // K/V rows: [kv (rank) | k_pe (rope)] per cached pos.
                let n_prev = state.mla[layer].len();
                let dim = rank + n_rot;
                let mut kk = Vec::with_capacity(n_prev * dim);
                let mut vv = Vec::with_capacity(n_prev * rank);
                for c in state.mla[layer].iter() {
                    let (ckv, cpe) = c.split_at(rank.min(c.len()));
                    kk.extend_from_slice(ckv);
                    kk.extend_from_slice(&cpe[..n_rot.min(cpe.len())]);
                    vv.extend_from_slice(ckv);
                }
                // Allowed positions: top-k ∩ causal.
                let allowed: Vec<usize> = topk.into_iter().filter(|&p| p <= pos).collect();
                // Absorbed scores over [rank|rope], values over [rank].
                let scale = 1.0 / (dim as f32).sqrt();
                let mut attn_rank = vec![0f32; n_heads * rank];
                for hh in 0..n_heads {
                    let mut scores = Vec::with_capacity(allowed.len());
                    for &p in &allowed {
                        let kr = &kk[p * dim..(p + 1) * dim];
                        scores.push(
                            qq[hh * dim..(hh + 1) * dim]
                                .iter()
                                .zip(kr.iter())
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                * scale,
                        );
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for (r, o) in attn_rank[hh * rank..(hh + 1) * rank].iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for (j, &p) in allowed.iter().enumerate() {
                            acc += scores[j] / sum * vv[p * rank + r];
                        }
                        *o = acc;
                    }
                }

                // V rows carry kv mass; project via wv_b to head_v, then wo.
                let wvb = w.get(&format!("{b}.attn_v_b.weight"))?;
                let mut v_heads = vec![0f32; n_heads * head_v];
                for hh in 0..n_heads {
                    let vh = &wvb.data[hh * rank * head_v..(hh + 1) * rank * head_v];
                    let mut out = vec![0f32; head_v];
                    for (r, o) in out.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for (c, &xv) in attn_rank[hh * rank..(hh + 1) * rank].iter().enumerate() {
                            acc += vh[c * head_v + r] * xv;
                        }
                        *o = acc;
                    }
                    v_heads[hh * head_v..(hh + 1) * head_v].copy_from_slice(&out);
                }
                let wo = w.get(&format!("{b}.attn_output.weight"))?;
                let mut proj = vec![0f32; d];
                matvec(wo, &v_heads, &mut proj)?;
                for (h_i, &p) in h.iter_mut().zip(proj.iter()) {
                    *h_i += p;
                }

                // FFN: dense lead or MoE.
                rmsnorm(&h, &w_ffn_norm.data, cfg.norm_eps, &mut n);
                let ffn_out = if layer < cfg.n_layer_dense_lead
                    && let (Ok(wg), Ok(wu), Ok(wd)) = (
                        w.get(&format!("{b}.ffn_gate.weight")),
                        w.get(&format!("{b}.ffn_up.weight")),
                        w.get(&format!("{b}.ffn_down.weight")),
                    ) {
                    let mut gate = vec![0f32; d_ffn];
                    let mut up = vec![0f32; d_ffn];
                    matvec(wg, &n, &mut gate)?;
                    matvec(wu, &n, &mut up)?;
                    for (g, &u) in gate.iter_mut().zip(up.iter()) {
                        *g = *g / (1.0 + (-*g).exp()) * u;
                    }
                    let mut down = vec![0f32; d];
                    matvec(wd, &gate, &mut down)?;
                    down
                } else {
                    moe_ffn(w, &b, &n, cfg)?
                };
                for (h_i, &f) in h.iter_mut().zip(ffn_out.iter()) {
                    *h_i += f;
                }
            }

            let mut n = vec![0f32; d];
            rmsnorm(&h, &out_norm.data, cfg.norm_eps, &mut n);
            matvec(out_w, &n, &mut logits_out[ti * vocab..(ti + 1) * vocab])?;
        }
        Ok(())
    }
}

/// Indexer top-k terminals: rope q/k, Hadamard rotate, ReLU-gated scores.
#[allow(clippy::too_many_arguments)]
fn indexer_top_k(
    w: &DenseWeights,
    b: &str,
    qrn: &[f32],
    n: &[f32],
    pos: usize,
    state: &mut GlmState,
    layer: usize,
    ihd: usize,
    ih: usize,
    top_k: usize,
    hadamard: &[f32],
    theta: f32,
) -> Result<Vec<usize>, ModelError> {
    let wqb = w.get(&format!("{b}.indexer_attn_q_b.weight"))?;
    let wk = w.get(&format!("{b}.indexer_attn_k.weight"))?;
    let wkn = w.get(&format!("{b}.indexer_k_norm.weight"))?;
    let wkn_b = w.get(&format!("{b}.indexer_k_norm_b.weight"))?;
    let wproj = w.get(&format!("{b}.indexer_proj.weight"))?;

    let mut iq = vec![0f32; ih * ihd];
    matvec(wqb, qrn, &mut iq)?;
    let rope = RopeConfig {
        head_dim: ihd,
        theta,
        ..Default::default()
    };
    apply_rope(&mut iq, pos as u32, &rope)
        .map_err(|e| ModelError::Forward(format!("indexer rope: {e}")))?;
    let mut ik = vec![0f32; ihd];
    matvec(wk, n, &mut ik)?;
    // LayerNorm with weight + bias.
    let mean: f32 = ik.iter().sum::<f32>() / ik.len() as f32;
    let var: f32 = ik.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / ik.len() as f32;
    for (i, v) in ik.iter_mut().enumerate() {
        *v = (*v - mean) / (var + 1e-5).sqrt() * wkn.data[i] + wkn_b.data[i];
    }
    apply_rope(&mut ik, pos as u32, &rope)
        .map_err(|e| ModelError::Forward(format!("indexer rope: {e}")))?;
    state.idx[layer].push(ik);

    // Hadamard-rotate q and all cached k.
    let rot = |v: &[f32]| {
        let mut o = vec![0f32; ihd];
        for (r, o_r) in o.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (c, &xv) in v.iter().enumerate() {
                acc += hadamard[c * ihd + r] * xv;
            }
            *o_r = acc;
        }
        o
    };
    let mut wts = vec![0f32; ih];
    matvec(wproj, n, &mut wts)?;
    let scale = 1.0 / ((ihd * ih) as f32).sqrt();
    for wv in wts.iter_mut() {
        *wv *= scale;
    }

    let n_prev = state.idx[layer].len();
    let mut scores = vec![0f32; n_prev];
    for (k_pos, cached) in state.idx[layer].iter().enumerate() {
        if k_pos > pos {
            scores[k_pos] = f32::NEG_INFINITY;
            continue;
        }
        let mut s = 0f32;
        for hh in 0..ih {
            let qh = rot(&iq[hh * ihd..(hh + 1) * ihd]);
            let kh = rot(cached);
            let dot: f32 = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum();
            s += dot.max(0.0) * wts[hh];
        }
        scores[k_pos] = s;
    }
    let mut idx: Vec<usize> = (0..n_prev).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(top_k.min(n_prev));
    Ok(idx)
}

/// Normalize Sylvester Hadamard matrix (llama.cpp `ggml_gen_hadamard`).
/// Head dim must be an exact power of two ≥ 1.
fn hadamard(n: usize) -> Result<Vec<f32>, ModelError> {
    if n == 0 || !n.is_power_of_two() {
        return Err(ModelError::Forward(
            "hadamard dim must be a power of two".into(),
        ));
    }
    let mut m = vec![0f32; n * n];
    m[0] = 1.0 / (n as f32).sqrt();
    let mut s = 1usize;
    while s < n {
        for i in 0..s {
            for j in 0..s {
                let v = m[i * n + j];
                m[(i + s) * n + j] = v;
                m[i * n + j + s] = v;
                m[(i + s) * n + j + s] = -v;
            }
        }
        s *= 2;
    }
    Ok(m)
}

/// MoE branch (sigmoid default, norm/scale/bias from config).
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
        1 => MoeGating::Standard {
            func: GatingFunc::Softmax,
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

    /// Tiny 1-layer model: MLA (split up-proj) + full indexer + dense FFN.
    fn tiny_weights() -> (ModelConfig, DenseWeights) {
        let d = 8usize;
        let nh = 2usize;
        let rank = 4usize;
        let rope = 2usize;
        let nope = 2usize;
        let hv = 4usize;
        let ff = 16usize;
        let v = 8usize;
        let ihd = 4usize;
        let ih = 1usize;

        let mut map: HashMap<String, (Vec<f32>, [u32; 4])> = HashMap::new();
        let w = |rows: usize, cols: usize, fill: f32| {
            (vec![fill; rows * cols], [cols as u32, rows as u32, 1, 1])
        };
        let ones = |n: usize| (vec![1.0; n], [n as u32, 1, 1, 1]);
        map.insert("token_embd.weight".into(), w(v, d, 0.1));
        map.insert("output_norm.weight".into(), ones(d));
        map.insert("output.weight".into(), w(v, d, 0.1));

        let b = "blk.0";
        for (k, val) in [
            ("attn_norm", ones(d)),
            ("ffn_norm", ones(d)),
            ("attn_q_a", w(rank, d, 0.05)),
            ("attn_q_a_norm", ones(rank)),
            ("attn_q_b", w(nh * (nope + rope), rank, 0.05)),
            ("attn_kv_a_mqa", w(rank + rope, d, 0.05)),
            ("attn_kv_a_norm", ones(rank)),
            (
                "attn_k_b",
                (
                    vec![0.05; nh * nope * rank],
                    [rank as u32, nope as u32, nh as u32, 1],
                ),
            ),
            (
                "attn_v_b",
                (
                    vec![0.05; nh * rank * hv],
                    [rank as u32, hv as u32, nh as u32, 1],
                ),
            ),
            ("attn_gate", w(nh * hv, d, 0.05)),
            ("attn_output", w(d, nh * hv, 0.05)),
            ("ffn_gate", w(ff, d, 0.05)),
            ("ffn_up", w(ff, d, 0.05)),
            ("ffn_down", w(d, ff, 0.05)),
            ("indexer_attn_q_b", w(ih * ihd, rank, 0.05)),
            ("indexer_attn_k", w(ihd, d, 0.05)),
            ("indexer_k_norm", ones(ihd)),
            ("indexer_k_norm_b", (vec![0.0; ihd], [ihd as u32, 1, 1, 1])),
            ("indexer_proj", w(ih, d, 0.05)),
        ] {
            map.insert(format!("{b}.{k}.weight"), val);
        }

        let cfg = ModelConfig {
            arch: "glm-dsa".into(),
            n_layers: 1,
            n_heads: nh,
            n_kv_heads: 1,
            d_model: d,
            d_ffn: ff,
            vocab_size: v,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            kv_lora_rank: rank,
            key_length_mla: nope + rope,
            value_length_mla: hv,
            rope_dim_count: rope,
            indexer_n_head: ih,
            indexer_head_size: ihd,
            indexer_top_k: 2,
            n_layer_dense_lead: 1,
            ..Default::default()
        };
        (cfg, DenseWeights::from_map(map))
    }

    #[test]
    fn sparse_forward_finite_and_incremental() {
        let (cfg, weights) = tiny_weights();
        let mut model = GlmDsa::new(cfg.clone());
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
