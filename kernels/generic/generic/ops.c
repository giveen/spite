/*
 * kernels/generic/generic/ops.c
 *
 * Reference scalar implementations of RMS norm, FFN, and attention.
 * These are ground-truth ops — all GPU kernels must match these numerically.
 *
 * Inputs:
 *   - Activation tensors (x, out) are always F32.
 *   - Weight tensors may be F32, Q8_0, or Q4_K; ops dequantize on-the-fly.
 *   - For other quant types the op returns -1 (dispatcher uses Rust fallback).
 *
 * Performance: intentionally ignored. One malloc per op call.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../../core/abi.h"
#include "../../../core/quant.h"

/* forward-declare dequant functions from dequant.c */
void dequant_q8_0(float *out, const block_q8_0 *blocks, int n);
void dequant_q4_K(float *out, const block_q4_K *blocks, int n);

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

    switch (t->kind) {
    case SPITE_TYPE_F32:
        memcpy(buf, t->data, (size_t)n * sizeof(float));
        return buf;

    case SPITE_TYPE_Q8_0: {
        int n_blocks = (n + QK8_0 - 1) / QK8_0;
        dequant_q8_0(buf, (const block_q8_0 *)t->data, n_blocks);
        return buf;
    }

    case SPITE_TYPE_Q4_K: {
        int n_blocks = (n + QK_K - 1) / QK_K;
        dequant_q4_K(buf, (const block_q4_K *)t->data, n_blocks);
        return buf;
    }

    default:
        free(buf);
        return NULL; /* unsupported — caller returns -1 */
    }
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
 * Reference implementation of the ABI v4 attention op for one token at
 * ctx->pos. It mirrors kernels/qwen/qwen3/nvidia/kv_attn.inl step for step,
 * which is what makes it usable as the numeric oracle for GPU attention in
 * tools/verify/verify.py:
 *
 *   q = Wq·x                         k = Wk·x      v = Wv·x
 *   q = RoPE(NEOX, pos)( RMSNorm_hd(q) * q_norm )   when q_norm != NULL
 *   k = RoPE(NEOX, pos)( RMSNorm_hd(k) * k_norm )   when k_norm != NULL
 *   K[pos] = k,  V[pos] = v
 *   scores[h,t] = (q_h · K[t][h/group]) / sqrt(hd)   for t in [0, pos]
 *   att[h,:]    = softmax_t scores[h,:] · V[t][h/group]
 *   out += Wo·att                                    (residual fused)
 *
 * Deviation from the GPU path, deliberate and load-bearing: the KV cache is
 * F32 only. This kernel leaves `kv_cache_kinds` NULL, which the host reads as
 * "F32 tiers only", so the contract and the declared capability agree. The
 * block-quantised tiers belong to the GPU kernels; their layout is pinned
 * separately by crates/spite-kvcache/tests/portable_codec_parity.rs.
 *
 * Also deliberately independent of ctx->scratchpad: this runs on the CPU
 * reference path and under verify.py, where ctx carries no device scratchpad.
 */

/* dst and src may alias. NEOX RoPE: pairs (i, i + hd/2), as in GGUF/llama.cpp. */
static void qk_norm_rope(float *dst, const float *src, const float *norm_w,
                         float eps, int hd, int pos, float theta) {
    const int half = hd / 2;
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
        const float freq = powf(theta, -2.0f * (float)i / (float)hd);
        const float angle = (float)pos * freq;
        const float sn = sinf(angle);
        const float cs = cosf(angle);
        dst[i]        = x0 * cs - x1 * sn;
        dst[i + half] = x0 * sn + x1 * cs;
    }
}

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
    if (!out || !x || !wq || !wk || !wv || !wo || !kvcache || !ctx) return -1;
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (ctx->n_heads <= 0 || ctx->n_kv_heads <= 0) return -1;

    const int nh  = ctx->n_heads;
    const int nkv = ctx->n_kv_heads;
    const int hd  = (int)wq->ne[1] / nh;
    const int pos = ctx->pos;
    const int kv_stride = nkv * hd;

    /* Same acceptance domain as kvattn_run, so a mismatch shows up as a
     * numeric difference rather than a spurious -1 from the reference. */
    if (hd <= 0 || (hd & 1) || (nh % nkv)) return -1;
    if ((int)kvcache->k.ne[0] != kv_stride) return -1;
    if ((int)kvcache->v.ne[0] != kv_stride) return -1;
    if (kvcache->k.kind != kvcache->v.kind) return -1;
    if (kvcache->k.kind != SPITE_TYPE_F32) return -1; /* see note above */
    const int n_ctx = (int)kvcache->k.ne[1];
    if (pos < 0 || pos >= n_ctx) return -2;

    float *wf_q = dequant_to_f32(wq);
    float *wf_k = dequant_to_f32(wk);
    float *wf_v = dequant_to_f32(wv);
    float *wf_o = dequant_to_f32(wo);
    float *nw_q = q_norm ? dequant_to_f32(q_norm) : NULL;
    float *nw_k = k_norm ? dequant_to_f32(k_norm) : NULL;
    if (!wf_q || !wf_k || !wf_v || !wf_o || (q_norm && !nw_q) || (k_norm && !nw_k)) {
        free(wf_q); free(wf_k); free(wf_v); free(wf_o); free(nw_q); free(nw_k);
        return -1;
    }

    /* q[nh*hd] k_stage[kv_stride] att[nh*hd] scores[nh*n_ctx] */
    const size_t n_q      = (size_t)nh * hd;
    const size_t n_att    = n_q;
    const size_t n_scores = (size_t)nh * n_ctx;
    float *buf = (float *)malloc((n_q + (size_t)kv_stride + n_att + n_scores) * sizeof(float));
    if (!buf) {
        free(wf_q); free(wf_k); free(wf_v); free(wf_o); free(nw_q); free(nw_k);
        return -1;
    }
    float *q       = buf;
    float *k_stage = q + n_q;
    float *att     = k_stage + kv_stride;
    float *scores  = att + n_att;

    const float *xin     = (const float *)x->data;
    const int    d_model = (int)x->ne[0];
    const int    group   = nh / nkv;
    const int    n_tok   = pos + 1;

    /* Row `pos` of the cache, addressed by byte stride exactly as the GPU
     * kernel does — nb[1] is not assumed to be 4 * kv_stride. */
    float *k_row = (float *)((uint8_t *)kvcache->k.data + (size_t)pos * kvcache->k.nb[1]);
    float *v_row = (float *)((uint8_t *)kvcache->v.data + (size_t)pos * kvcache->v.nb[1]);

    /* Projections. V goes straight into its cache row, as on the GPU. */
    matmul_f32(q, wf_q, xin, (int)n_q, d_model, 1);
    matmul_f32(k_stage, wf_k, xin, kv_stride, d_model, 1);
    matmul_f32(v_row, wf_v, xin, kv_stride, d_model, 1);

    /* Per-head QK RMSNorm + NEOX RoPE. K lands in the cache row. */
    for (int h = 0; h < nh; h++)
        qk_norm_rope(q + (size_t)h * hd, q + (size_t)h * hd, nw_q, norm_eps, hd, pos,
                     rope_freq_base);
    for (int h = 0; h < nkv; h++)
        qk_norm_rope(k_row + (size_t)h * hd, k_stage + (size_t)h * hd, nw_k, norm_eps, hd, pos,
                     rope_freq_base);

    const float scale = 1.0f / sqrtf((float)hd);

    /* scores -> online softmax -> weighted V, per head. Fusing these three
     * passes is a GPU concern; the reference keeps them separate so it reads
     * like the maths. */
    for (int h = 0; h < nh; h++) {
        const float *qh = q + (size_t)h * hd;
        const int    kh = h / group;
        float       *sc = scores + (size_t)h * n_ctx;

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
    }

    /* out += Wo·att (ABI v4 residual fusion). */
    const int o_cols = (int)wo->ne[0];
    const int o_rows = (int)wo->ne[1];
    float *outp = (float *)out->data;
    for (int r = 0; r < o_rows; r++) {
        float acc = 0.0f;
        for (int c = 0; c < o_cols; c++) acc += wf_o[(size_t)r * o_cols + c] * att[c];
        outp[r] += acc;
    }

    free(buf);
    free(wf_q); free(wf_k); free(wf_v); free(wf_o); free(nw_q); free(nw_k);
    return 0;
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
