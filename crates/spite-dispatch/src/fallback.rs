//! Pure-Rust generic fallbacks for every kernel op.
//!
//! These are used when no .so kernel is available for a slot — they run on
//! the CPU and are correct but unoptimized. The dispatch builder calls into
//! these rather than leaving a slot as None whenever the generic .so is also
//! absent (e.g. in unit tests or stripped distributions).

use spite_abi::{SpiteCtx, SpiteKvCache, SpiteTensor, SpiteType};

// ── RMS Norm ───────────────────────────────────────────────────────────────

/// Scalar RMS layer normalisation: out[i] = x[i] / rms(x) * weight[i]
///
/// # Safety
///
/// `out`, `input`, and `weight` must be valid, non-null pointers to
/// `SpiteTensor`s with compatible shapes. `input` and `out` must be F32.
pub unsafe fn rms_norm(
    out: *mut SpiteTensor,
    input: *const SpiteTensor,
    weight: *const SpiteTensor,
    eps: f32,
    _ctx: *const SpiteCtx,
) -> i32 {
    let (out, input, weight) = unsafe { (&mut *out, &*input, &*weight) };
    let n = input.ne[0] as usize;
    let xs = unsafe { std::slice::from_raw_parts(input.data as *const f32, n) };
    let ws = unsafe { std::slice::from_raw_parts(weight.data as *const f32, n) };
    let ys = unsafe { std::slice::from_raw_parts_mut(out.data as *mut f32, n) };

    let rms = (xs.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    for i in 0..n {
        ys[i] = xs[i] / rms * ws[i];
    }
    0
}

// ── Attention ──────────────────────────────────────────────────────────────

/// Naive O(n²) scaled dot-product attention (single head, F32 only).
///
/// # Safety
///
/// `out`, `q`, `k`, and `v` must be valid, non-null pointers to `SpiteTensor`s
/// with compatible shapes. All tensors must be F32.
pub unsafe fn attention(
    out: *mut SpiteTensor,
    q: *const SpiteTensor,
    k: *const SpiteTensor,
    v: *const SpiteTensor,
    _kvc: *const SpiteKvCache,
    _ctx: *const SpiteCtx,
) -> i32 {
    let (out, q, k, v) = unsafe { (&mut *out, &*q, &*k, &*v) };
    let seq_q = q.ne[1] as usize;
    let seq_k = k.ne[1] as usize;
    let d_head = q.ne[0] as usize;
    let scale = 1.0 / (d_head as f32).sqrt();

    let qs = unsafe { std::slice::from_raw_parts(q.data as *const f32, seq_q * d_head) };
    let ks = unsafe { std::slice::from_raw_parts(k.data as *const f32, seq_k * d_head) };
    let vs = unsafe { std::slice::from_raw_parts(v.data as *const f32, seq_k * d_head) };
    let os = unsafe { std::slice::from_raw_parts_mut(out.data as *mut f32, seq_q * d_head) };

    let mut scores = vec![0f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            let dot: f32 = (0..d_head)
                .map(|d| qs[i * d_head + d] * ks[j * d_head + d])
                .sum();
            scores[i * seq_k + j] = dot * scale;
        }
        // softmax over row i
        let row = &mut scores[i * seq_k..(i + 1) * seq_k];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in row.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in row.iter_mut() {
            *s /= sum;
        }
    }
    // weighted sum over V
    for i in 0..seq_q {
        for d in 0..d_head {
            os[i * d_head + d] = (0..seq_k)
                .map(|j| scores[i * seq_k + j] * vs[j * d_head + d])
                .sum();
        }
    }
    0
}

// ── FFN ────────────────────────────────────────────────────────────────────

/// Minimal SwiGLU FFN: out = (gate * silu(up)) down, all F32.
///
/// # Safety
///
/// `out`, `input`, `gate`, `up`, and `down` must be valid, non-null pointers
/// to `SpiteTensor`s with compatible shapes. All tensors must be F32.
pub unsafe fn ffn(
    out: *mut SpiteTensor,
    input: *const SpiteTensor,
    gate: *const SpiteTensor,
    up: *const SpiteTensor,
    down: *const SpiteTensor,
    _ctx: *const SpiteCtx,
) -> i32 {
    let (out, input, gate, up, down) = unsafe { (&mut *out, &*input, &*gate, &*up, &*down) };
    let d_model = input.ne[0] as usize;
    let d_ffn = gate.ne[1] as usize;

    let xs = unsafe { std::slice::from_raw_parts(input.data as *const f32, d_model) };
    let gw = unsafe { std::slice::from_raw_parts(gate.data as *const f32, d_model * d_ffn) };
    let uw = unsafe { std::slice::from_raw_parts(up.data as *const f32, d_model * d_ffn) };
    let dw = unsafe { std::slice::from_raw_parts(down.data as *const f32, d_ffn * d_model) };
    let ys = unsafe { std::slice::from_raw_parts_mut(out.data as *mut f32, d_model) };

    let mut g_act = vec![0f32; d_ffn];
    for j in 0..d_ffn {
        let g: f32 = (0..d_model).map(|i| xs[i] * gw[i * d_ffn + j]).sum();
        let u: f32 = (0..d_model).map(|i| xs[i] * uw[i * d_ffn + j]).sum();
        g_act[j] = silu(g) * u;
    }
    for i in 0..d_model {
        ys[i] = (0..d_ffn).map(|j| g_act[j] * dw[j * d_model + i]).sum();
    }
    0
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── Speculative verify ─────────────────────────────────────────────────────

/// CPU fallback for the speculative verify step.
///
/// Fills `accept_mask[i]` using the standard accept/reject rule:
///   accept with prob min(1, p(t_i) / q(t_i))
///
/// Returns 0 on success, non-zero on bad input.
///
/// # Safety
///
/// `accept_mask`, `draft_logits`, and `main_logits` must be valid, non-null
/// pointers. `draft_logits` and `main_logits` must each point to `n_draft`
/// valid `SpiteTensor` values; `accept_mask` must point to at least `n_draft`
/// writable `bool`s. All tensors must be F32.
pub unsafe extern "C" fn spec_verify_fallback(
    accept_mask: *mut bool,
    draft_logits: *const SpiteTensor,
    main_logits: *const SpiteTensor,
    temperature: f32,
    n_draft: u32,
    _ctx: *const SpiteCtx,
) -> i32 {
    if accept_mask.is_null() || draft_logits.is_null() || main_logits.is_null() {
        return -1;
    }
    let n = n_draft as usize;
    let mask = unsafe { std::slice::from_raw_parts_mut(accept_mask, n) };
    let drafts = unsafe { std::slice::from_raw_parts(draft_logits, n) };
    let mains = unsafe { std::slice::from_raw_parts(main_logits, n) };

    let mut rng: u64 = 0xdead_beef_cafe_babe;

    for (m, (dl, ml)) in mask.iter_mut().zip(drafts.iter().zip(mains.iter())) {
        // We expect F32 tensors; bail on anything else.
        if dl.kind != SpiteType::F32 || ml.kind != SpiteType::F32 {
            return -2;
        }
        let vocab = dl.ne[0] as usize;
        let dlogits = unsafe { std::slice::from_raw_parts(dl.data as *const f32, vocab) };
        let mlogits = unsafe { std::slice::from_raw_parts(ml.data as *const f32, vocab) };

        // The draft token is the argmax of draft_logits (greedy draft).
        let t = dlogits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let q = softmax_at(dlogits, t, temperature);
        let p = softmax_at(mlogits, t, temperature);
        let accept_prob = (p / q.max(1e-10)).min(1.0);
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((rng >> 33) as f32) / (u32::MAX as f32);
        *m = u < accept_prob;
    }
    0
}

fn softmax_at(logits: &[f32], idx: usize, temperature: f32) -> f32 {
    if temperature <= 0.0 {
        let best = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        return if best == idx { 1.0 } else { 0.0 };
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let sum: f32 = exp.iter().sum();
    exp.get(idx).copied().unwrap_or(0.0) / sum.max(1e-10)
}

// ── MTP Stem ───────────────────────────────────────────────────────────────

/// Pure-Rust fallback for `mtp_stem`: normalize embedding, normalize hidden state, pack [2*d, T].
///
/// # Safety
///
/// `out`, `embed`, and `hidden` must be valid, non-null pointers to `SpiteTensor`s.
/// If provided, `w_enorm` and `w_hnorm` must point to valid `SpiteTensor`s.
pub unsafe fn mtp_stem(
    out: *mut SpiteTensor,
    embed: *const SpiteTensor,
    hidden: *const SpiteTensor,
    w_enorm: *const SpiteTensor,
    w_hnorm: *const SpiteTensor,
    eps: f32,
    _ctx: *const SpiteCtx,
) -> i32 {
    if out.is_null() || embed.is_null() || hidden.is_null() {
        return -1;
    }
    let (out, embed, hidden) = unsafe { (&mut *out, &*embed, &*hidden) };
    let d = embed.ne[0] as usize;
    let t = embed.ne[1].max(1) as usize;

    if out.ne[0] as usize != 2 * d {
        return -1;
    }

    let e_data = unsafe { std::slice::from_raw_parts(embed.data as *const f32, d * t) };
    let h_data = unsafe { std::slice::from_raw_parts(hidden.data as *const f32, d * t) };
    let out_data = unsafe { std::slice::from_raw_parts_mut(out.data as *mut f32, 2 * d * t) };

    let we_data = if !w_enorm.is_null() && !unsafe { (*w_enorm).data }.is_null() {
        Some(unsafe { std::slice::from_raw_parts((*w_enorm).data as *const f32, d) })
    } else {
        None
    };
    let wh_data = if !w_hnorm.is_null() && !unsafe { (*w_hnorm).data }.is_null() {
        Some(unsafe { std::slice::from_raw_parts((*w_hnorm).data as *const f32, d) })
    } else {
        None
    };

    let emb_scale = (d as f32).sqrt();

    for tok in 0..t {
        let e_slice = &e_data[tok * d..(tok + 1) * d];
        let h_slice = &h_data[tok * d..(tok + 1) * d];
        let out_slice = &mut out_data[tok * 2 * d..(tok + 1) * 2 * d];

        if let (Some(we), Some(wh)) = (we_data, wh_data) {
            let rms_e = (e_slice.iter().map(|&v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
            let rms_h = (h_slice.iter().map(|&v| v * v).sum::<f32>() / d as f32 + eps).sqrt();

            for i in 0..d {
                out_slice[i] = (e_slice[i] / rms_e) * we[i];
                out_slice[d + i] = (h_slice[i] / rms_h) * wh[i];
            }
        } else {
            for i in 0..d {
                let mut ev = e_slice[i] * emb_scale;
                if let Some(we) = we_data {
                    ev *= we[i];
                }
                out_slice[i] = ev;

                let mut hv = h_slice[i];
                if let Some(wh) = wh_data {
                    hv *= wh[i];
                }
                out_slice[d + i] = hv;
            }
        }
    }

    0
}
