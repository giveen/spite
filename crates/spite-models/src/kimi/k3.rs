//! Moonshot AI Kimi K3 — GGUF arch `kimi-k3`.
//!
//! Hybrid MLA + KDA (gated-delta) layers with cross-layer residual
//! banking, SiTU FFNs, and latent MoE. Ported from llama.cpp
//! `models/kimi-k3.cpp`. Kimi MLA is nope-only (no RoPE anywhere).

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_compute::flash_attn::{FlashAttnConfig, scalar_attention};
use spite_compute::linear_attn::{gdn_l2_norm, gdn_step, ssm_conv_step};
use spite_loader::GgufModel;
use spite_rope::RopeConfig;

use crate::dense::{DenseWeights, matvec, rmsnorm};
use crate::moe::{GatingFunc, MoeActivation, MoeGating, moe_forward};
use crate::{ModelArch, ModelConfig, ModelError};

/// Per-layer state: MLA cache, GDN state, conv histories.
enum LayerState {
    Mla {
        cache: Vec<Vec<f32>>,
    },
    Kda {
        gdn: Vec<f32>,
        conv_q: Vec<Vec<f32>>,
        conv_k: Vec<Vec<f32>>,
        conv_v: Vec<Vec<f32>>,
    },
}

pub struct KimiK3 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    state: RwLock<KimiState>,
}

#[derive(Default)]
struct KimiState {
    layers: Vec<LayerState>,
    /// Banked residual checkpoints per position (oldest first).
    res_stacks: Vec<Vec<Vec<f32>>>,
}

impl KimiK3 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            state: RwLock::new(KimiState::default()),
        }
    }
}

impl ModelArch for KimiK3 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    fn reset_cache(&self) {
        if let Ok(mut s) = self.state.write() {
            s.layers.clear();
            s.res_stacks.clear();
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
        let vocab = cfg.vocab_size;
        if logits_out.len() != tokens.len() * vocab {
            return Err(ModelError::Forward("logits_out shape mismatch".into()));
        }
        // ponytail: O(ctx²)/O(ctx·S²) scalar CPU path; GPU kernels own speed.
        let res_bs = cfg.attn_res_block_size;
        let use_res = res_bs > 0;
        let rope = RopeConfig {
            head_dim: d / n_heads.max(1),
            theta: cfg.rope_theta,
            ..Default::default()
        };
        let _ = rope;

        let embd = w.get("token_embd.weight")?;
        let out_norm = w.get("output_norm.weight")?;
        let out_w = w.get("output.weight")?;
        let out_res = w.get("output_res_score").ok();

        let mut state = self
            .state
            .write()
            .map_err(|_| ModelError::Forward("state lock".into()))?;
        while state.layers.len() < cfg.n_layers {
            let layer = state.layers.len();
            state.layers.push(if kimi_is_recurrent(cfg, layer) {
                LayerState::Kda {
                    gdn: Vec::new(),
                    conv_q: Vec::new(),
                    conv_k: Vec::new(),
                    conv_v: Vec::new(),
                }
            } else {
                LayerState::Mla { cache: Vec::new() }
            });
        }

        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            while state.res_stacks.len() <= pos {
                state.res_stacks.push(Vec::new());
            }
            let mut h = vec![0f32; d];
            h.copy_from_slice(&embd.data[tok as usize * d..(tok as usize + 1) * d]);
            let mut prefix = h.clone();

            for layer in 0..cfg.n_layers {
                let b = format!("blk.{layer}");
                let w_norm = w.get(&format!("{b}.attn_norm.weight"))?;
                let w_ffn_norm = w.get(&format!("{b}.ffn_norm.weight"))?;
                let w_attn_res = w.get(&format!("{b}.attn_res_score")).ok();
                let w_ffn_res = w.get(&format!("{b}.ffn_res_score")).ok();

                let mut cur = if use_res {
                    res_mix(&state.res_stacks[pos], &prefix, w_attn_res, cfg.norm_eps)?
                } else {
                    prefix.clone()
                };
                let banked = use_res && layer % res_bs == 0;
                if banked {
                    state.res_stacks[pos].push(prefix.clone());
                }
                let mut n = vec![0f32; d];
                rmsnorm(&cur, &w_norm.data, cfg.norm_eps, &mut n);
                cur = if !kimi_is_recurrent(cfg, layer) {
                    mla_attn(w, &b, &n, cfg, &mut state, layer)?
                } else {
                    kda_attn(w, &b, &n, cfg, &mut state, layer)?
                };
                prefix = if banked { cur } else { add(&prefix, &cur) };

                cur = if use_res {
                    res_mix(&state.res_stacks[pos], &prefix, w_ffn_res, cfg.norm_eps)?
                } else {
                    prefix.clone()
                };
                rmsnorm(&cur, &w_ffn_norm.data, cfg.norm_eps, &mut n);
                cur = ffn(w, &b, &n, cfg, layer)?;
                prefix = add(&prefix, &cur);
            }

            if use_res {
                prefix = res_mix(&state.res_stacks[pos], &prefix, out_res, cfg.norm_eps)?;
            }
            let mut n = vec![0f32; d];
            rmsnorm(&prefix, &out_norm.data, cfg.norm_eps, &mut n);
            matvec(out_w, &n, &mut logits_out[ti * vocab..(ti + 1) * vocab])?;
        }
        Ok(())
    }
}

/// Kimi recurrent rule: per-layer kv-head array marks KDA layers with 0.
fn kimi_is_recurrent(cfg: &ModelConfig, layer: usize) -> bool {
    if let Some(&kv) = cfg.head_count_kv_arr.get(layer) {
        return kv == 0;
    }
    cfg.recurrent_layers.get(layer).copied().unwrap_or(false)
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Cross-layer residual mix: softmax over checkpoint scores + current.
fn res_mix(
    stack: &[Vec<f32>],
    cur: &[f32],
    score_w: Option<&crate::dense::Weight>,
    eps: f32,
) -> Result<Vec<f32>, ModelError> {
    let Some(sw) = score_w else {
        return Ok(cur.to_vec());
    };
    if stack.is_empty() {
        return Ok(cur.to_vec());
    }
    let d = cur.len();
    let mut scores = Vec::with_capacity(stack.len() + 1);
    for ckpt in stack {
        scores.push(scored(ckpt, &sw.data, eps, d));
    }
    scores.push(scored(cur, &sw.data, eps, d));
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    let mut out = vec![0f32; d];
    for (ckpt, &p) in stack.iter().zip(scores.iter()) {
        let w = p / sum;
        for (o, &c) in out.iter_mut().zip(ckpt.iter()) {
            *o += w * c;
        }
    }
    let w_cur = scores[stack.len()] / sum;
    for (o, &c) in out.iter_mut().zip(cur.iter()) {
        *o += w_cur * c;
    }
    Ok(out)
}

fn scored(x: &[f32], w: &[f32], eps: f32, d: usize) -> f32 {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / d as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(&v, &ww)| v * scale * ww).sum()
}

/// MLA layer (nope-only): compressed KV cache, fused or split up-projection,
/// sigmoid output gate, output proj.
#[allow(clippy::too_many_arguments)]
fn mla_attn(
    w: &DenseWeights,
    b: &str,
    n: &[f32],
    cfg: &ModelConfig,
    state: &mut KimiState,
    layer: usize,
) -> Result<Vec<f32>, ModelError> {
    let d = cfg.d_model;
    let n_heads = cfg.n_heads;
    let rank = cfg.kv_lora_rank.max(1);
    let head_k = cfg.key_length.max(1);
    let head_v = cfg.value_length.max(1);
    let rope_dim = cfg.rope_dim_count.min(head_k);
    let nope_dim = head_k - rope_dim;

    // Q (possibly low-rank).
    let q_full = if let (Ok(wqa), Ok(wqb)) = (
        w.get(&format!("{b}.attn_q_a.weight")),
        w.get(&format!("{b}.attn_q_b.weight")),
    ) {
        let wqn = w.get(&format!("{b}.attn_q_a_norm.weight"))?;
        let mut qr = vec![0f32; rank.min(wqa.ne[1] as usize)];
        matvec(wqa, n, &mut qr)?;
        let mut qrn = vec![0f32; qr.len()];
        rmsnorm(&qr, &wqn.data, cfg.norm_eps, &mut qrn);
        let mut q = vec![0f32; n_heads * head_k];
        matvec(wqb, &qrn, &mut q)?;
        q
    } else {
        let wq = w.get(&format!("{b}.attn_q.weight"))?;
        let mut q = vec![0f32; n_heads * head_k];
        matvec(wq, n, &mut q)?;
        q
    };

    // Compressed KV + shared rope part (never roped: nope-only).
    let wkv_a = w.get(&format!("{b}.attn_kv_a_mqa.weight"))?;
    let kv_rows = wkv_a.ne[1] as usize;
    let mut kv_full = vec![0f32; kv_rows];
    matvec(wkv_a, n, &mut kv_full)?;
    let (kv_raw, k_pe) = kv_full.split_at(rank.min(kv_rows));
    let mut kvn = vec![0f32; kv_raw.len()];
    let wkn = w.get(&format!("{b}.attn_kv_a_norm.weight"))?;
    rmsnorm(kv_raw, &wkn.data, cfg.norm_eps, &mut kvn);

    let LayerState::Mla { cache } = &mut state.layers[layer] else {
        return Err(ModelError::Forward("layer state kind mismatch".into()));
    };
    let mut row = Vec::with_capacity(kvn.len() + k_pe.len());
    row.extend_from_slice(&kvn);
    row.extend_from_slice(k_pe);
    cache.push(row);

    // Up-project the whole cache: K = [up(nope) | k_pe], V = up(v).
    let mut kk = Vec::new();
    let mut vv = Vec::new();
    if let Ok(wkv_b) = w.get(&format!("{b}.attn_kv_b.weight")) {
        for c in cache.iter() {
            let (ckv, cpe) = c.split_at(kvn.len());
            let mut up = vec![0f32; n_heads * (nope_dim + head_v)];
            matvec(wkv_b, ckv, &mut up)?;
            for h in 0..n_heads {
                let base = h * (nope_dim + head_v);
                kk.extend_from_slice(&up[base..base + nope_dim]);
                kk.extend_from_slice(cpe);
                vv.extend_from_slice(&up[base + nope_dim..base + nope_dim + head_v]);
            }
        }
    } else {
        let wkb = w.get(&format!("{b}.attn_k_b.weight"))?;
        let wvb = w.get(&format!("{b}.attn_v_b.weight"))?;
        for c in cache.iter() {
            let (ckv, cpe) = c.split_at(kvn.len());
            // wk_b is [nope, rank, n_heads]: per-head slice.
            for h in 0..n_heads {
                let kh = &wkb.data[h * nope_dim * rank..(h + 1) * nope_dim * rank];
                let mut kh_out = vec![0f32; nope_dim];
                matvec_raw(kh, nope_dim, rank, ckv, &mut kh_out)?;
                kk.extend_from_slice(&kh_out);
                kk.extend_from_slice(cpe);
                let vh = &wvb.data[h * rank * head_v..(h + 1) * rank * head_v];
                let mut vh_out = vec![0f32; head_v];
                matvec_raw(vh, head_v, rank, ckv, &mut vh_out)?;
                vv.extend_from_slice(&vh_out);
            }
        }
    }
    // Q split into nope/rope parts (rope part unused numerically: nope-only).
    let n_prev = cache.len();
    let mut qq = Vec::with_capacity(n_heads * (nope_dim + rope_dim));
    for h in 0..n_heads {
        let base = h * head_k;
        qq.extend_from_slice(&q_full[base..base + nope_dim]);
        qq.extend_from_slice(&q_full[base + nope_dim..base + nope_dim + rope_dim]);
    }
    let attn_cfg = FlashAttnConfig::new(1, n_prev, n_heads, n_heads, nope_dim + rope_dim);
    // Repeat k_pe per head is already expanded in kk; heads match 1:1.
    let mut attn_out = vec![0f32; n_heads * (nope_dim + rope_dim)];
    // NOTE: kk/vv are laid out head-major ([h][pos][dim]); scalar_attention
    // expects position-major. Reorder below.
    let (kk_pm, vv_pm) = reorder_head_major(&kk, &vv, n_heads, n_prev, nope_dim + rope_dim);
    scalar_attention(&qq, &kk_pm, &vv_pm, &mut attn_out, &attn_cfg)
        .map_err(|e| ModelError::Forward(format!("mla attn: {e}")))?;

    // K3 output gate on the normed input, then out proj over v-dim only.
    let mut gated = vec![0f32; n_heads * head_v];
    for h in 0..n_heads {
        gated[h * head_v..(h + 1) * head_v].copy_from_slice(
            &attn_out[h * (nope_dim + rope_dim)..h * (nope_dim + rope_dim) + head_v],
        );
    }
    if let Ok(wg) = w.get(&format!("{b}.attn_gate.weight")) {
        let mut g = vec![0f32; n_heads * head_v];
        // Gate proj sized to v-dim; fall back to full if mismatched.
        if wg.ne[1] as usize == n_heads * head_v {
            matvec(wg, n, &mut g)?;
            for (o, &gg) in gated.iter_mut().zip(g.iter()) {
                *o *= 1.0 / (1.0 + (-gg).exp());
            }
        }
    }
    let wo = w.get(&format!("{b}.attn_output.weight"))?;
    let mut proj = vec![0f32; d];
    matvec(wo, &gated, &mut proj)?;
    Ok(proj)
}

/// Row-major raw matvec over a borrowed slice: out[r] = Σ_c w[c*rows+r]·x[c].
fn matvec_raw(
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), ModelError> {
    if x.len() != cols || out.len() != rows || w.len() != rows * cols {
        return Err(ModelError::Forward("matvec shape mismatch".into()));
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

/// Reorder head-major [h][pos][dim] to position-major [pos][h][dim].
fn reorder_head_major(
    kk: &[f32],
    vv: &[f32],
    n_heads: usize,
    n_pos: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut ko = vec![0f32; kk.len()];
    let mut vo = vec![0f32; vv.len()];
    for h in 0..n_heads {
        for p in 0..n_pos {
            let src = h * n_pos * dim + p * dim;
            let dst = p * n_heads * dim + h * dim;
            ko[dst..dst + dim].copy_from_slice(&kk[src..src + dim]);
            vo[dst..dst + dim].copy_from_slice(&vv[src..src + dim]);
        }
    }
    (ko, vo)
}

/// KDA linear layer: per-qkv convs, GDN recurrence, output gate.
fn kda_attn(
    w: &DenseWeights,
    b: &str,
    n: &[f32],
    cfg: &ModelConfig,
    state: &mut KimiState,
    layer: usize,
) -> Result<Vec<f32>, ModelError> {
    let d = cfg.d_model;
    let n_heads = cfg.n_heads;
    let hd = cfg.kda_head_dim.max(1);
    let d_inner = hd * n_heads;
    let k_size = cfg.ssm_d_conv.max(1);

    // QKV projs (fused or split), then one conv per third.
    let (qq, kk, vv) = if let Ok(wqkv) = w.get(&format!("{b}.attn_qkv.weight")) {
        let mut all = vec![0f32; 3 * d_inner];
        matvec(wqkv, n, &mut all)?;
        let (q, rest) = all.split_at(d_inner);
        let (k, v) = rest.split_at(d_inner);
        (q.to_vec(), k.to_vec(), v.to_vec())
    } else {
        let wq = w.get(&format!("{b}.attn_q.weight"))?;
        let wk = w.get(&format!("{b}.attn_k.weight"))?;
        let wv = w.get(&format!("{b}.attn_v.weight"))?;
        let (mut q, mut k, mut v) = (
            vec![0f32; d_inner],
            vec![0f32; d_inner],
            vec![0f32; d_inner],
        );
        matvec(wq, n, &mut q)?;
        matvec(wk, n, &mut k)?;
        matvec(wv, n, &mut v)?;
        (q, k, v)
    };
    let wqc = w.get(&format!("{b}.ssm_q_conv.weight"))?;
    let wkc = w.get(&format!("{b}.ssm_k_conv.weight"))?;
    let wvc = w.get(&format!("{b}.ssm_v_conv.weight"))?;

    let LayerState::Kda {
        gdn,
        conv_q,
        conv_k,
        conv_v,
    } = &mut state.layers[layer]
    else {
        return Err(ModelError::Forward("layer state kind mismatch".into()));
    };
    let step = |hist: &mut Vec<Vec<f32>>, x: &[f32], kw: &crate::dense::Weight| {
        while hist.len() + 1 < k_size {
            hist.insert(0, vec![0f32; d_inner]);
        }
        // Conv kernels may be stored 4D [K,1,C,1]; first K*C values apply.
        let need = k_size * d_inner;
        let ker = if kw.data.len() >= need {
            &kw.data[..need]
        } else {
            &kw.data[..]
        };
        let mut k_full = vec![0f32; need];
        k_full[..ker.len().min(need)].copy_from_slice(&ker[..ker.len().min(need)]);
        let out = ssm_conv_step(hist, x, &k_full);
        hist.push(x.to_vec());
        while hist.len() + 1 > k_size {
            hist.remove(0);
        }
        out.iter()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect::<Vec<f32>>()
    };
    let qc = step(conv_q, &qq, wqc);
    let kc = step(conv_k, &kk, wkc);
    let vc = step(conv_v, &vv, wvc);

    // Decay gate: bounded form when lower bound set, else softplus form.
    let wfa = w.get(&format!("{b}.ssm_f_a.weight"))?;
    let wfb = w.get(&format!("{b}.ssm_f_b.weight"))?;
    let wdt = w
        .get(&format!("{b}.ssm_dt"))
        .or_else(|_| w.get(&format!("{b}.ssm_dt.weight")))?;
    let wa = w
        .get(&format!("{b}.ssm_a"))
        .or_else(|_| w.get(&format!("{b}.ssm_a.weight")))?;
    let mut fa = vec![0f32; hd];
    matvec(wfa, n, &mut fa)?;
    let mut g1 = vec![0f32; d_inner];
    matvec(wfb, &fa, &mut g1)?;
    for (g, &dt) in g1.iter_mut().zip(wdt.data.iter()) {
        *g += dt;
    }
    // g1 is [hd, nh] in (dim, head) order already (row-major d_inner).
    let lower = cfg.kda_gate_lower_bound;
    let mut decay = vec![0f32; d_inner];
    if lower.is_finite() {
        for h in 0..n_heads {
            let a = wa.data[h.min(wa.data.len() - 1)];
            for s in 0..hd {
                let v = g1[h * hd + s] * a;
                decay[h * hd + s] = lower * (1.0 / (1.0 + (-v).exp()));
            }
        }
    } else {
        for h in 0..n_heads {
            let a = wa.data[h.min(wa.data.len() - 1)];
            for s in 0..hd {
                let v = g1[h * hd + s];
                decay[h * hd + s] = (v.ln_1p().exp()) * a;
            }
        }
    }

    let wbeta = w.get(&format!("{b}.ssm_beta.weight"))?;
    let mut beta_raw = vec![0f32; n_heads];
    matvec(wbeta, n, &mut beta_raw)?;

    if gdn.len() < n_heads * hd * hd {
        gdn.resize(n_heads * hd * hd, 0.0);
    }
    // L2-normalize q/k per head.
    let mut qn = vec![0f32; d_inner];
    let mut kn = vec![0f32; d_inner];
    for h in 0..n_heads {
        gdn_l2_norm(
            &qc[h * hd..(h + 1) * hd],
            cfg.norm_eps,
            &mut qn[h * hd..(h + 1) * hd],
        );
        gdn_l2_norm(
            &kc[h * hd..(h + 1) * hd],
            cfg.norm_eps,
            &mut kn[h * hd..(h + 1) * hd],
        );
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let mut scan = vec![0f32; d_inner];
    for h in 0..n_heads {
        let qs: Vec<f32> = qn[h * hd..(h + 1) * hd]
            .iter()
            .map(|&x| x * scale)
            .collect();
        let beta = 1.0 / (1.0 + (-beta_raw[h]).exp());
        let st = &mut gdn[h * hd * hd..(h + 1) * hd * hd];
        let mut o = vec![0f32; hd];
        gdn_step(
            st,
            &qs,
            &kn[h * hd..(h + 1) * hd],
            &vc[h * hd..(h + 1) * hd],
            &decay[h * hd..(h + 1) * hd],
            beta,
            &mut o,
        )
        .map_err(|e| ModelError::Forward(format!("kda gdn: {e}")))?;
        scan[h * hd..(h + 1) * hd].copy_from_slice(&o);
    }

    // Output gate: rmsnorm(scan) * sigmoid(ssm_g·x), then wo.
    let wo_norm = w.get(&format!("{b}.ssm_o_norm.weight"))?;
    let wg = w.get(&format!("{b}.ssm_g.weight"))?;
    let wo = w.get(&format!("{b}.attn_output.weight"))?;
    let mut normed = vec![0f32; d_inner];
    // ssm_o_norm is per-head-dim; apply per head slice.
    for h in 0..n_heads {
        rmsnorm(
            &scan[h * hd..(h + 1) * hd],
            &wo_norm.data,
            cfg.norm_eps,
            &mut normed[h * hd..(h + 1) * hd],
        );
    }
    let mut g2 = vec![0f32; d_inner];
    matvec(wg, n, &mut g2)?;
    for (y, &gg) in normed.iter_mut().zip(g2.iter()) {
        *y *= 1.0 / (1.0 + (-gg).exp());
    }
    let mut proj = vec![0f32; d];
    matvec(wo, &normed, &mut proj)?;
    Ok(proj)
}

/// FFN: dense SiTU lead layers (tensor presence), else latent MoE.
fn ffn(
    w: &DenseWeights,
    b: &str,
    n: &[f32],
    cfg: &ModelConfig,
    layer: usize,
) -> Result<Vec<f32>, ModelError> {
    let d = cfg.d_model;
    let d_ffn = cfg.d_ffn;
    if layer < cfg.n_layer_dense_lead
        && let (Ok(w_gate), Ok(w_up), Ok(w_down)) = (
            w.get(&format!("{b}.ffn_gate.weight")),
            w.get(&format!("{b}.ffn_up.weight")),
            w.get(&format!("{b}.ffn_down.weight")),
        )
    {
        let mut gate = vec![0f32; d_ffn];
        let mut up = vec![0f32; d_ffn];
        matvec(w_gate, n, &mut gate)?;
        matvec(w_up, n, &mut up)?;
        let mut out = vec![0f32; d_ffn];
        for ((o, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
            *o = situ(g, u, cfg.situ_beta, cfg.situ_linear_beta);
        }
        let mut down = vec![0f32; d];
        matvec(w_down, &out, &mut down)?;
        return Ok(down);
    }
    // Latent MoE.
    let latent = cfg.n_expert_latent.max(1);
    let routed_in = if let Ok(wd) = w.get(&format!("{b}.ffn_routed_down.weight")) {
        let mut r = vec![0f32; latent];
        matvec(wd, n, &mut r)?;
        r
    } else {
        n.to_vec()
    };
    let w_inp = w.get(&format!("{b}.ffn_gate_inp.weight"))?;
    let mut logits = vec![0f32; cfg.n_expert.max(1)];
    matvec(w_inp, n, &mut logits)?;
    let bias = w
        .get(&format!("{b}.ffn_exp_probs_b"))
        .ok()
        .map(|t| t.data.clone());
    let w_up = w.get(&format!("{b}.ffn_up_exps.weight"))?;
    let w_gate = w.get(&format!("{b}.ffn_gate_exps.weight"))?;
    let w_down = w.get(&format!("{b}.ffn_down_exps.weight"))?;
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
        &routed_in,
        &logits,
        bias.as_deref(),
        w_up,
        w_gate,
        w_down,
        cfg.n_expert_used,
        ff_exp,
        cfg.expert_weights_scale,
        gating,
        MoeActivation::SiTu {
            beta: cfg.situ_beta,
            linear_beta: cfg.situ_linear_beta,
        },
    )?;
    if let Ok(wn) = w.get(&format!("{b}.ffn_routed_norm.weight")) {
        let mut normed = vec![0f32; out.len()];
        rmsnorm(&out, &wn.data, cfg.norm_eps, &mut normed);
        out = normed;
    }
    if let Ok(wu) = w.get(&format!("{b}.ffn_routed_up.weight")) {
        let mut up = vec![0f32; d];
        matvec(wu, &out, &mut up)?;
        out = up;
    }
    if let (Ok(wsg), Ok(wsu), Ok(wsd)) = (
        w.get(&format!("{b}.ffn_gate_shexp.weight")),
        w.get(&format!("{b}.ffn_up_shexp.weight")),
        w.get(&format!("{b}.ffn_down_shexp.weight")),
    ) {
        let se = wsg.ne[1] as usize;
        let mut g = vec![0f32; se];
        let mut u = vec![0f32; se];
        matvec(wsg, n, &mut g)?;
        matvec(wsu, n, &mut u)?;
        let mut sh = vec![0f32; se];
        for ((o, &gg), &uu) in sh.iter_mut().zip(g.iter()).zip(u.iter()) {
            *o = situ(gg, uu, cfg.situ_beta, cfg.situ_linear_beta);
        }
        let mut sh_down = vec![0f32; d];
        matvec(wsd, &sh, &mut sh_down)?;
        for (o, &s) in out.iter_mut().zip(sh_down.iter()) {
            *o += s;
        }
    }
    Ok(out)
}

fn situ(g: f32, u: f32, beta: f32, linear_beta: f32) -> f32 {
    let a = beta * (g / beta).tanh() / (1.0 + (-g).exp());
    let uu = if linear_beta > 0.0 {
        linear_beta * (u / linear_beta).tanh()
    } else {
        u
    };
    a * uu
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny hybrid: layer 0 MLA (fused up-proj, sigmoid gate), layer 1 KDA,
    /// res banking every 2 layers, layer 0 dense SiTU, layer 1 latent MoE.
    fn tiny_weights() -> (ModelConfig, DenseWeights) {
        let d = 8usize;
        let nh = 2usize;
        let rank = 4usize;
        let rope = 2usize;
        let nope = 2usize;
        let hv = 4usize;
        let ff = 16usize;
        let v = 8usize;
        let hd = 4usize;
        let ksize = 2usize;
        let ne = 2usize;

        let mut map: HashMap<String, (Vec<f32>, [u32; 4])> = HashMap::new();
        let w = |rows: usize, cols: usize, fill: f32| {
            (vec![fill; rows * cols], [cols as u32, rows as u32, 1, 1])
        };
        let ones = |n: usize| (vec![1.0; n], [n as u32, 1, 1, 1]);
        map.insert("token_embd.weight".into(), w(v, d, 0.1));
        map.insert("output_norm.weight".into(), ones(d));
        map.insert("output.weight".into(), w(v, d, 0.1));
        map.insert("output_res_score".into(), ones(d));

        // Layer 0: MLA.
        let b = "blk.0";
        for (k, val) in [
            ("attn_norm", ones(d)),
            ("ffn_norm", ones(d)),
            ("attn_res_score", ones(d)),
            ("ffn_res_score", ones(d)),
            ("attn_q", w(nh * (nope + rope), d, 0.05)),
            ("attn_kv_a_mqa", w(rank + rope, d, 0.05)),
            ("attn_kv_a_norm", ones(rank)),
            ("attn_kv_b", w(nh * (nope + hv), rank, 0.05)),
            ("attn_gate", w(nh * hv, d, 0.05)),
            ("attn_output", w(d, nh * hv, 0.05)),
            ("ffn_gate", w(ff, d, 0.05)),
            ("ffn_up", w(ff, d, 0.05)),
            ("ffn_down", w(d, ff, 0.05)),
        ] {
            map.insert(format!("{b}.{k}.weight"), val);
        }

        // Layer 1: KDA + latent MoE.
        let b = "blk.1";
        let di = hd * nh;
        for (k, val) in [
            ("attn_norm", ones(d)),
            ("ffn_norm", ones(d)),
            ("attn_res_score", ones(d)),
            ("ffn_res_score", ones(d)),
            ("attn_qkv", w(3 * di, d, 0.05)),
            (
                "ssm_q_conv",
                (vec![0.25; ksize * di], [di as u32, ksize as u32, 1, 1]),
            ),
            (
                "ssm_k_conv",
                (vec![0.25; ksize * di], [di as u32, ksize as u32, 1, 1]),
            ),
            (
                "ssm_v_conv",
                (vec![0.25; ksize * di], [di as u32, ksize as u32, 1, 1]),
            ),
            ("ssm_f_a", w(hd, d, 0.05)),
            ("ssm_f_b", w(di, hd, 0.05)),
            ("ssm_dt", (vec![0.1; di], [di as u32, 1, 1, 1])),
            ("ssm_a", (vec![-0.5; nh], [nh as u32, 1, 1, 1])),
            ("ssm_beta", w(nh, d, 0.05)),
            ("ssm_g", w(di, d, 0.05)),
            ("ssm_o_norm", ones(hd)),
            ("attn_output", w(d, di, 0.05)),
            ("ffn_gate_inp", w(ne, d, 0.5)),
            ("ffn_exp_probs_b", (vec![0.0; ne], [ne as u32, 1, 1, 1])),
            ("ffn_routed_down", w(d, d, 0.1)),
            ("ffn_routed_up", w(d, d, 0.1)),
            ("ffn_routed_norm", ones(d)),
            ("ffn_gate_shexp", w(ff, d, 0.05)),
            ("ffn_up_shexp", w(ff, d, 0.05)),
            ("ffn_down_shexp", w(d, ff, 0.05)),
        ] {
            map.insert(format!("{b}.{k}.weight"), val);
        }
        let stack = |fill: f32| (vec![fill; ne * d * ff], [d as u32, ff as u32, ne as u32, 1]);
        map.insert("blk.1.ffn_up_exps.weight".into(), stack(0.05));
        map.insert("blk.1.ffn_gate_exps.weight".into(), stack(0.05));
        map.insert(
            "blk.1.ffn_down_exps.weight".into(),
            (vec![0.05; ne * ff * d], [ff as u32, d as u32, ne as u32, 1]),
        );

        let cfg = ModelConfig {
            arch: "kimi-k3".into(),
            n_layers: 2,
            n_heads: nh,
            n_kv_heads: nh,
            d_model: d,
            d_ffn: ff,
            vocab_size: v,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            q_lora_rank: 0,
            kv_lora_rank: rank,
            key_length: nope + rope,
            value_length: hv,
            rope_dim_count: rope,
            kda_head_dim: hd,
            ssm_d_conv: ksize,
            kda_gate_lower_bound: f32::NEG_INFINITY,
            n_layer_dense_lead: 1,
            n_expert: ne,
            n_expert_used: 1,
            n_expert_latent: d,
            expert_weights_norm: false,
            expert_gating_func: 2,
            attn_res_block_size: 2,
            situ_beta: 1.0,
            situ_linear_beta: 0.0,
            head_count_kv_arr: vec![nh, 0],
            ..Default::default()
        };
        (cfg, DenseWeights::from_map(map))
    }

    #[test]
    fn hybrid_forward_finite_and_incremental() {
        let (cfg, weights) = tiny_weights();
        let mut model = KimiK3::new(cfg.clone());
        model.weights = Some(weights);
        let ctx = SpiteCtx {
            n_ctx: 64,
            n_batch: 1,
            n_threads: 1,
            pos: 0,
            n_heads: 2,
            n_kv_heads: 2,
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
