/*
 * kernels/generic/generic/ops.c
 *
 * Reference scalar implementations of RMS norm, FFN, and attention.
 * These are ground-truth ops — all GPU kernels must match these numerically.
 *
 * Inputs:
 *   - Activation tensors (x, out) are always F32.
 *   - Weight tensors may be any type spite_dequantize_row() supports (all GGUF
 *     quant types, F32/F16/BF16); ops dequantize on-the-fly.
 *   - Unsupported types make the op return -1 (dispatcher uses Rust fallback).
 *
 * Performance: intentionally ignored. One malloc per op call.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../../core/abi.h"
#include "../../../core/quant.h"
#include "ref_common.h"

/* ── Tensor helpers ───────────────────────────────────────────────────── */

static int tensor_rows(const SpiteTensor *t) { return (int)t->ne[1]; }
static int tensor_cols(const SpiteTensor *t) { return (int)t->ne[0]; }
static int tensor_elems(const SpiteTensor *t) {
    int n = 1;
    for (int i = 0; i < 4; i++) if (t->ne[i]) n *= (int)t->ne[i];
    return n;
}

/*
 * Dequantize `t` into a freshly malloc'd F32 buffer.
 * Caller must free() the result.
 * Returns NULL if the quant type is unsupported.
 */
static float *dequant_to_f32(const SpiteTensor *t) {
    int n = tensor_elems(t);
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return NULL;

    if (spite_dequantize_row((SpiteType)t->kind, t->data, buf, n) != 0) {
        free(buf);
        return NULL; /* unsupported type or ragged length — caller returns -1 */
    }
    return buf;
}

/* ── RMS Norm ─────────────────────────────────────────────────────────── */

/*
 * out[i] = x[i] / sqrt( mean(x^2) + eps ) * weight[i]
 *
 * Applied row-wise: x is [rows, cols], weight is [cols].
 */
int spite_generic_rms_norm(
    SpiteTensor       *out,
    const SpiteTensor *x,
    const SpiteTensor *weight,
    float              eps,
    const SpiteCtx    *ctx
) {
    (void)ctx;
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;

    float *weight_f32 = dequant_to_f32(weight);
    if (!weight_f32) return -1;

    const float *xp  = (const float *)x->data;
    float       *outp = (float *)out->data;
    int rows = tensor_rows(x);
    int cols = tensor_cols(x);
    if (rows == 0) rows = 1; /* treat 1-D as single row */
    if (rows == 0) rows = 1;

    for (int r = 0; r < rows; r++) {
        const float *row_in  = xp   + r * cols;
        float       *row_out = outp + r * cols;

        float sum_sq = 0.0f;
        for (int i = 0; i < cols; i++) sum_sq += row_in[i] * row_in[i];
        float rms_inv = 1.0f / sqrtf(sum_sq / (float)cols + eps);

        for (int i = 0; i < cols; i++) {
            row_out[i] = row_in[i] * rms_inv * weight_f32[i];
        }
    }

    free(weight_f32);
    return 0;
}

/* ── FFN: gate/up projections + SiLU activation + down projection ─────── */

/*
 * out = down_proj( silu(gate_proj(x)) * up_proj(x) )
 *
 * x       : [1, hidden_dim] (single token, F32)
 * w_gate  : [ffn_dim, hidden_dim]
 * w_up    : [ffn_dim, hidden_dim]
 * w_down  : [hidden_dim, ffn_dim]
 * out     : [1, hidden_dim]
 */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void matmul_f32(float *out, const float *a, const float *b,
                        int m, int k, int n) {
    /* out[m,n] = a[m,k] × b[k,n]  (row-major) */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++) {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

int spite_generic_ffn(
    SpiteTensor       *out,
    const SpiteTensor *x,
    const SpiteTensor *w_gate,
    const SpiteTensor *w_up,
    const SpiteTensor *w_down,
    SpiteFfnActivation activation,
    const SpiteCtx    *ctx
) {
    (void)ctx;
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    /* Only SwiGLU is implemented; the dispatcher falls back for the rest. */
    if (activation != SPITE_FFN_SILU_GATE) return -1;

    float *wg = dequant_to_f32(w_gate);
    float *wu = dequant_to_f32(w_up);
    float *wd = dequant_to_f32(w_down);
    if (!wg || !wu || !wd) { free(wg); free(wu); free(wd); return -1; }

    /* Dimensions: single token assumed (batch_size=1) */
    int hidden = tensor_cols(x);
    int ffn    = tensor_rows(w_gate);  /* ffn_dim: output of gate/up proj */

    float *gate = (float *)malloc((size_t)ffn * sizeof(float));
    float *up   = (float *)malloc((size_t)ffn * sizeof(float));
    if (!gate || !up) { free(gate); free(up); free(wg); free(wu); free(wd); return -1; }

    /* gate = w_gate × x  [ffn, hidden] × [hidden, 1] → [ffn, 1] */
    matmul_f32(gate, wg, (const float *)x->data, ffn, hidden, 1);
    /* up   = w_up   × x */
    matmul_f32(up,   wu, (const float *)x->data, ffn, hidden, 1);

    /* fuse: gate = silu(gate) * up */
    for (int i = 0; i < ffn; i++) gate[i] = silu(gate[i]) * up[i];

    /* out += w_down × gate  [hidden, ffn] × [ffn, 1] → [hidden, 1]
     * ABI v4: ffn accumulates into out (residual fused). */
    float *down = (float *)malloc((size_t)hidden * sizeof(float));
    if (!down) { free(gate); free(up); free(wg); free(wu); free(wd); return -1; }
    matmul_f32(down, wd, gate, hidden, ffn, 1);
    float *o = (float *)out->data;
    for (int i = 0; i < hidden; i++) o[i] += down[i];
    free(down);

    free(gate); free(up);
    free(wg); free(wu); free(wd);
    return 0;
}

/* ── Attention ────────────────────────────────────────────────────────── */

/*
 * Reference implementation of the attention ops for one token at ctx->pos.
 * spite_generic_attention_ex is the general op (ABI v7); the ABI v4
 * spite_generic_attention is exactly attention_ex with gated_q = 0 and
 * rope_dim = head_dim.  It mirrors kernels/qwen/qwen3/nvidia/kv_attn.inl step
 * for step, which is what makes it usable as the numeric oracle for GPU
 * attention in tools/verify/verify.py:
 *
 *   qfull = Wq·x                     k = Wk·x      v = Wv·x
 *   per head h: q_h = qfull[h*qs .. +hd], gate_h = qfull[h*qs+hd .. +hd]
 *               (qs = hd, or 2*hd when gated_q: INTERLEAVED per head)
 *   q_h = RoPE_rd(NEOX, pos)( RMSNorm_hd(q_h) * q_norm )   when q_norm != NULL
 *   k_h = RoPE_rd(NEOX, pos)( RMSNorm_hd(k_h) * k_norm )   when k_norm != NULL
 *       RoPE rotates only the first rope_dim dims of a head, pairs (i, i+rope_dim/2),
 *       theta_i = pos * base^(-2i/rope_dim); the other dims pass through unrotated
 *   K[pos] = k,  V[pos] = v
 *   scores[h,t] = (q_h · K[t][h/group]) / sqrt(hd)   for t in [0, pos]
 *   att[h,:]    = softmax_t scores[h,:] · V[t][h/group]
 *   att[h,:]   *= sigmoid(gate_h)                    when gated_q
 *   out += Wo·att                                    (residual fused)
 *
 * Deviation from the GPU path, deliberate and load-bearing: the KV cache is
 * F32 only. This kernel leaves `kv_cache_kinds` NULL, which the host reads as
 * "F32 tiers only", so the contract and the declared capability agree. The
 * block-quantised tiers belong to the GPU kernels; their layout is pinned
 * separately by crates/spite-kvcache/tests/portable_codec_parity.rs.
 *
 * Weights are streamed through ref_gemv() (any SpiteType, float accumulation in
 * column order — the arithmetic of the original attention op, bit for bit).
 * The work area is ctx->scratchpad when the host provides one, else malloc;
 * under verify.py the reference runs with none.
 */

/*
 * dst and src may alias. Per-head RMSNorm over all hd dims, then NEOX RoPE on
 * the first rope_dim of them (pairs (i, i + rope_dim/2), as in GGUF/llama.cpp);
 * dims [rope_dim, hd) are normed but not rotated.
 */
static void qk_norm_rope(float *dst, const float *src, const float *norm_w, float eps, int hd,
                         int rope_dim, int pos, float theta) {
    const int half = rope_dim / 2;
    float scale = 1.0f;
    if (norm_w) {
        float sum_sq = 0.0f;
        for (int i = 0; i < hd; i++) sum_sq += src[i] * src[i];
        scale = 1.0f / sqrtf(sum_sq / (float)hd + eps);
    }
    for (int i = 0; i < half; i++) {
        float x0 = src[i] * scale;
        float x1 = src[i + half] * scale;
        if (norm_w) {
            x0 *= norm_w[i];
            x1 *= norm_w[i + half];
        }
        const float freq = powf(theta, -2.0f * (float)i / (float)rope_dim);
        const float angle = (float)pos * freq;
        const float sn = sinf(angle);
        const float cs = cosf(angle);
        dst[i]        = x0 * cs - x1 * sn;
        dst[i + half] = x0 * sn + x1 * cs;
    }
    for (int i = rope_dim; i < hd; i++) {
        float v = src[i] * scale;
        if (norm_w) v *= norm_w[i];
        dst[i] = v;
    }
}

/* F32 view of a q/k norm weight: the tensor itself when F32, else decoded into `tmp`. */
static const float *norm_weights(const SpiteTensor *t, float *tmp, int hd) {
    if (!t) return NULL;
    if (t->kind == SPITE_TYPE_F32) return (const float *)t->data;
    (void)spite_dequantize_row(t->kind, t->data, tmp, hd);
    return tmp;
}

int spite_generic_attention_ex(
    SpiteTensor           *out,
    const SpiteTensor     *x,
    const SpiteTensor     *wq,
    const SpiteTensor     *wk,
    const SpiteTensor     *wv,
    const SpiteTensor     *wo,
    const SpiteTensor     *q_norm,
    const SpiteTensor     *k_norm,
    float                  norm_eps,
    SpiteKvCache          *kvcache,
    float                  rope_freq_base,
    const SpiteAttnParams *params,
    const SpiteCtx        *ctx
) {
    if (!out || !x || !wq || !wk || !wv || !wo || !kvcache || !params || !ctx) return -1;
    if (ctx->n_heads <= 0 || ctx->n_kv_heads <= 0) return -1;
    if (ctx->n_heads > REF_MAX_DIM || ctx->n_kv_heads > REF_MAX_DIM) return -1;

    const int nh    = ctx->n_heads;
    const int nkv   = ctx->n_kv_heads;
    const int hd    = params->head_dim;
    const int rd    = params->rope_dim;
    const int gated = params->gated_q;
    const int pos   = ctx->pos;

    /* Same acceptance domain as kvattn_run, so a mismatch shows up as a
     * numeric difference rather than a spurious -1 from the reference. */
    if (hd < 1 || hd > REF_MAX_DIM) return -1;
    if (rd < 2 || rd > hd || (rd & 1)) return -1;
    if (gated != 0 && gated != 1) return -1;
    if (nh % nkv) return -1;

    const int kv_stride = nkv * hd;
    const int q_stride  = gated ? 2 * hd : hd;     /* floats per head in qfull */
    const int d_model   = (int)x->ne[0];
    const int d_out     = (int)out->ne[0];
    if (d_model < 1 || d_out < 1) return -1;
    if (!ref_f32_vec(x, d_model) || !ref_f32_vec(out, d_out)) return -1;
    if (!ref_weight_ok(wq, d_model, (int64_t)nh * q_stride) ||
        !ref_weight_ok(wk, d_model, kv_stride) || !ref_weight_ok(wv, d_model, kv_stride) ||
        !ref_weight_ok(wo, (int64_t)nh * hd, d_out))
        return -1;
    if ((q_norm && !ref_weight_ok(q_norm, hd, 1)) || (k_norm && !ref_weight_ok(k_norm, hd, 1)))
        return -1;

    if (!kvcache->k.data || !kvcache->v.data) return -1;
    if ((int)kvcache->k.ne[0] != kv_stride) return -1;
    if ((int)kvcache->v.ne[0] != kv_stride) return -1;
    if (kvcache->k.kind != kvcache->v.kind) return -1;
    if (kvcache->k.kind != SPITE_TYPE_F32) return -1; /* see note above */
    const int n_ctx = (int)(kvcache->k.ne[1] < kvcache->v.ne[1] ? kvcache->k.ne[1]
                                                               : kvcache->v.ne[1]);
    if (pos < 0 || pos >= n_ctx) return -2;

    const int n_tok = pos + 1;

    /* qfull[nh*q_stride] k_stage[kv_stride] att[nh*hd] scores[nh*n_tok] + the
     * decoded q/k norm weights when those are not F32 */
    const uint64_t n_q      = (uint64_t)nh * q_stride;
    const uint64_t n_att    = (uint64_t)nh * hd;
    const uint64_t n_scores = (uint64_t)nh * n_tok;
    const uint64_t n_norm   = (q_norm && q_norm->kind != SPITE_TYPE_F32 ? (uint64_t)hd : 0) +
                              (k_norm && k_norm->kind != SPITE_TYPE_F32 ? (uint64_t)hd : 0);
    int owned = 0, err = -1;
    float *buf = ref_work_buf(ctx, n_q + kv_stride + n_att + n_scores + n_norm, &owned, &err);
    if (!buf) return err;

    float *q       = buf;
    float *k_stage = q + n_q;
    float *att     = k_stage + kv_stride;
    float *scores  = att + n_att;
    float *nq_tmp  = scores + n_scores;
    float *nk_tmp  = nq_tmp + (q_norm && q_norm->kind != SPITE_TYPE_F32 ? hd : 0);
    const float *nw_q = norm_weights(q_norm, nq_tmp, hd);
    const float *nw_k = norm_weights(k_norm, nk_tmp, hd);

    const float *xin  = (const float *)x->data;
    const int    group = nh / nkv;

    /* Row `pos` of the cache, addressed by byte stride exactly as the GPU
     * kernel does — nb[1] is not assumed to be 4 * kv_stride. */
    float *k_row = (float *)((uint8_t *)kvcache->k.data + (size_t)pos * kvcache->k.nb[1]);
    float *v_row = (float *)((uint8_t *)kvcache->v.data + (size_t)pos * kvcache->v.nb[1]);

    /* Projections. V goes straight into its cache row, as on the GPU. */
    ref_gemv(wq, xin, q, 0, 0);
    ref_gemv(wk, xin, k_stage, 0, 0);
    ref_gemv(wv, xin, v_row, 0, 0);

    /* Per-head QK RMSNorm + NEOX RoPE (q part only; the gate half is untouched).
     * K lands in the cache row. */
    for (int h = 0; h < nh; h++)
        qk_norm_rope(q + (size_t)h * q_stride, q + (size_t)h * q_stride, nw_q, norm_eps, hd, rd,
                     pos, rope_freq_base);
    for (int h = 0; h < nkv; h++)
        qk_norm_rope(k_row + (size_t)h * hd, k_stage + (size_t)h * hd, nw_k, norm_eps, hd, rd, pos,
                     rope_freq_base);

    const float scale = 1.0f / sqrtf((float)hd);

    /* scores -> online softmax -> weighted V, per head. Fusing these three
     * passes is a GPU concern; the reference keeps them separate so it reads
     * like the maths. */
    for (int h = 0; h < nh; h++) {
        const float *qh = q + (size_t)h * q_stride;
        const int    kh = h / group;
        float       *sc = scores + (size_t)h * n_tok;

        for (int t = 0; t < n_tok; t++) {
            const float *kt = (const float *)((const uint8_t *)kvcache->k.data +
                                              (size_t)t * kvcache->k.nb[1]) + (size_t)kh * hd;
            float acc = 0.0f;
            for (int i = 0; i < hd; i++) acc += qh[i] * kt[i];
            sc[t] = acc * scale;
        }

        float m = -INFINITY;
        for (int t = 0; t < n_tok; t++) m = fmaxf(m, sc[t]);
        float sum = 0.0f;
        for (int t = 0; t < n_tok; t++) { sc[t] = expf(sc[t] - m); sum += sc[t]; }
        const float inv = 1.0f / sum;
        for (int t = 0; t < n_tok; t++) sc[t] *= inv;

        const int vh = h / group;
        float    *ah = att + (size_t)h * hd;
        for (int i = 0; i < hd; i++) {
            float acc = 0.0f;
            for (int t = 0; t < n_tok; t++) {
                const float *vt = (const float *)((const uint8_t *)kvcache->v.data +
                                                  (size_t)t * kvcache->v.nb[1]) + (size_t)vh * hd;
                acc += sc[t] * vt[i];
            }
            ah[i] = acc;
        }
        if (gated) {
            const float *gh = qh + hd;
            for (int i = 0; i < hd; i++) ah[i] *= 1.0f / (1.0f + expf(-gh[i]));
        }
    }

    /* out += Wo·att (ABI v4 residual fusion). */
    ref_gemv(wo, att, (float *)out->data, 1, 0);

    if (owned) free(buf);
    return 0;
}

/* ABI v4 attention: head_dim comes from wq, every dim is rotated, no gate. */
int spite_generic_attention(
    SpiteTensor       *out,
    const SpiteTensor *x,
    const SpiteTensor *wq,
    const SpiteTensor *wk,
    const SpiteTensor *wv,
    const SpiteTensor *wo,
    const SpiteTensor *q_norm,
    const SpiteTensor *k_norm,
    float              norm_eps,
    SpiteKvCache      *kvcache,
    float              rope_freq_base,
    const SpiteCtx    *ctx
) {
    if (!wq || !ctx || ctx->n_heads <= 0) return -1;
    const int64_t hd = (int64_t)wq->ne[1] / ctx->n_heads;
    if (hd < 1 || hd > REF_MAX_DIM) return -1;
    const SpiteAttnParams p = { .head_dim = (int32_t)hd, .rope_dim = (int32_t)hd, .gated_q = 0 };
    return spite_generic_attention_ex(out, x, wq, wk, wv, wo, q_norm, k_norm, norm_eps, kvcache,
                                      rope_freq_base, &p, ctx);
}

/* ── Matmul ───────────────────────────────────────────────────────────── */

/* out[r] = sum_c w[r, c] * x[c]; single token. Reference for GPU matmul. */
int spite_generic_matmul(
    SpiteTensor       *out,
    const SpiteTensor *x,
    const SpiteTensor *w,
    const SpiteCtx    *ctx
) {
    (void)ctx;
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    float *wf = dequant_to_f32(w);
    if (!wf) return -1;
    matmul_f32((float *)out->data, wf, (const float *)x->data,
               tensor_rows(w), tensor_cols(w), 1);
    free(wf);
    return 0;
}

/* ── MTP Stem ─────────────────────────────────────────────────────────── */

int spite_generic_mtp_stem(
    SpiteTensor       *out_packed,
    const SpiteTensor *embed,
    const SpiteTensor *hidden,
    const SpiteTensor *w_enorm,
    const SpiteTensor *w_hnorm,
    float              eps,
    const SpiteCtx    *ctx
) {
    (void)ctx;
    if (!out_packed || !embed || !hidden) return -1;
    if (embed->kind != SPITE_TYPE_F32 || hidden->kind != SPITE_TYPE_F32 || out_packed->kind != SPITE_TYPE_F32) return -1;

    const int d = tensor_cols(embed);
    int t = tensor_rows(embed);
    if (t < 1) t = 1;
    if (tensor_cols(out_packed) != 2 * d) return -1;

    float *we = w_enorm ? dequant_to_f32(w_enorm) : NULL;
    float *wh = w_hnorm ? dequant_to_f32(w_hnorm) : NULL;
    if ((w_enorm && !we) || (w_hnorm && !wh)) {
        if (we) free(we);
        if (wh) free(wh);
        return -1;
    }

    const float *e_data = (const float *)embed->data;
    const float *h_data = (const float *)hidden->data;
    float *out_data = (float *)out_packed->data;
    const float emb_scale = sqrtf((float)d);

    for (int tok = 0; tok < t; ++tok) {
        const float *e_tok = e_data + (size_t)tok * d;
        const float *h_tok = h_data + (size_t)tok * d;
        float *out_tok = out_data + (size_t)tok * (2 * d);

        if (we && wh) {
            /* Qwen-style MTP stem: RMSNorm(embed) * we, RMSNorm(hidden) * wh */
            float sum_e = 0.0f;
            float sum_h = 0.0f;
            for (int i = 0; i < d; ++i) {
                sum_e += e_tok[i] * e_tok[i];
                sum_h += h_tok[i] * h_tok[i];
            }
            const float rms_e = sqrtf(sum_e / (float)d + eps);
            const float rms_h = sqrtf(sum_h / (float)d + eps);

            for (int i = 0; i < d; ++i) {
                out_tok[i]     = (e_tok[i] / rms_e) * we[i];
                out_tok[d + i] = (h_tok[i] / rms_h) * wh[i];
            }
        } else {
            /* Gemma-style MTP stem: embed * sqrt(d), hidden */
            for (int i = 0; i < d; ++i) {
                float ev = e_tok[i] * emb_scale;
                if (we) ev *= we[i];
                out_tok[i] = ev;

                float hv = h_tok[i];
                if (wh) hv *= wh[i];
                out_tok[d + i] = hv;
            }
        }
    }

    if (we) free(we);
    if (wh) free(wh);
    return 0;
}
