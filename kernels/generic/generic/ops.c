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
    const SpiteCtx    *ctx
) {
    (void)ctx;
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;

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

    /* out = w_down × gate  [hidden, ffn] × [ffn, 1] → [hidden, 1] */
    matmul_f32((float *)out->data, wd, gate, hidden, ffn, 1);

    free(gate); free(up);
    free(wg); free(wu); free(wd);
    return 0;
}

/* ── Attention ────────────────────────────────────────────────────────── */

/*
 * Return -1: the Rust spite-compute scalar GQA fallback handles this.
 * A correct-but-simple C implementation would be substantial; the Rust
 * version in spite-compute/src/flash_attn.rs is the reference for attention.
 */
int spite_generic_attention(
    SpiteTensor       *out,
    const SpiteTensor *x,
    const SpiteTensor *wq,
    const SpiteTensor *wk,
    const SpiteTensor *wv,
    const SpiteTensor *wo,
    SpiteKvCache      *kvcache,
    int                pos,
    float              rope_freq_base,
    const SpiteCtx    *ctx
) {
    (void)out; (void)x; (void)wq; (void)wk; (void)wv; (void)wo;
    (void)kvcache; (void)pos; (void)rope_freq_base; (void)ctx;
    return -1; /* defer to Rust scalar fallback */
}
