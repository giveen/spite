/*
 * kernels/generic/generic/ref_common.h
 *
 * Helpers shared by the layer-level reference ops (attention_ex in ops.c, the
 * Gated Delta Net layer in linear_attn.c).  Everything here is on the model-data
 * boundary, so nothing asserts or aborts: a tensor or geometry the op cannot
 * serve makes the check fail and the op returns -1 (-2 for a too-small scratch).
 */

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include "../../../core/abi.h"
#include "../../../core/quant.h"

/* Elements dequantized per spite_dequantize_row() call in ref_gemv().  256 is a
 * multiple of every SpiteType block size (1/32/64/128/256), so chunks of a row
 * always start on a block boundary and only the tail is shorter. */
#define REF_CHUNK 256

/* Sanity bound on every geometry parameter (head counts, head_dim, d_conv, ...).
 * Real models are four orders of magnitude below it; it keeps every product of
 * up to three parameters inside int64_t. */
#define REF_MAX_DIM ((int64_t)1 << 20)

/* Number of elements of `t` (ne[i] == 0 counts as 1, as for unused dims). */
static inline uint64_t ref_elems(const SpiteTensor *t) {
    uint64_t n = 1;
    for (int i = 0; i < 4; i++)
        if (t->ne[i]) n *= t->ne[i];
    return n;
}

/* Contiguous F32 tensor of exactly n elements (n == 0: only the struct is needed). */
static inline int ref_f32_vec(const SpiteTensor *t, int64_t n) {
    if (!t || t->kind != SPITE_TYPE_F32) return 0;
    if (n > 0 && !t->data) return 0;
    return ref_elems(t) == (uint64_t)n && spite_tensor_is_contiguous(t);
}

/*
 * Weight matrix [cols, rows] (GGUF layout: row r is contiguous, `cols` along
 * ne[0]) of any type spite_dequantize_row() decodes.  The type is probed by
 * decoding the first block, so ref_gemv() can never fail half way through.
 */
static inline int ref_weight_ok(const SpiteTensor *w, int64_t cols, int64_t rows) {
    if (!w || !w->data || cols < 1 || rows < 1) return 0;
    if ((int64_t)w->ne[0] != cols || (int64_t)w->ne[1] != rows) return 0;
    if (w->ne[2] > 1 || w->ne[3] > 1) return 0;
    const uint32_t be = spite_type_block_elements(w->kind);
    if (be == 0 || cols % be != 0 || !spite_tensor_is_contiguous(w)) return 0;
    float probe[REF_CHUNK];
    return spite_dequantize_row(w->kind, w->data, probe, be) == 0;
}

/*
 * y[r] = w[r,:] . x   (accumulate == 0)   or   y[r] += w[r,:] . x   (accumulate != 0)
 * for every row of a tensor that passed ref_weight_ok().  Rows are dequantized
 * REF_CHUNK elements at a time into a stack buffer, so the cost is O(1) memory
 * whatever the weight size.  dbl != 0 accumulates in double (rounded to float
 * once per row); dbl == 0 accumulates in float in column order.
 */
static inline void ref_gemv(const SpiteTensor *w, const float *x, float *y, int accumulate,
                            int dbl) {
    const int64_t cols = (int64_t)w->ne[0], rows = (int64_t)w->ne[1];
    const uint64_t be = spite_type_block_elements(w->kind);
    const uint64_t bb = spite_type_block_bytes(w->kind);
    const uint64_t row_bytes = bb * ((uint64_t)cols / be);
    float buf[REF_CHUNK];

    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *row = (const uint8_t *)w->data + (uint64_t)r * row_bytes;
        float accf = 0.0f;
        double accd = 0.0;
        for (int64_t c0 = 0; c0 < cols; c0 += REF_CHUNK) {
            const int64_t n = cols - c0 < REF_CHUNK ? cols - c0 : REF_CHUNK;
            (void)spite_dequantize_row(w->kind, row + ((uint64_t)c0 / be) * bb, buf, n);
            if (dbl)
                for (int64_t i = 0; i < n; i++) accd += (double)buf[i] * (double)x[c0 + i];
            else
                for (int64_t i = 0; i < n; i++) accf += buf[i] * x[c0 + i];
        }
        if (dbl)
            y[r] = (float)((accumulate ? (double)y[r] : 0.0) + accd);
        else
            y[r] = accumulate ? y[r] + accf : accf;
    }
}

/*
 * Work area of `n` floats: ctx->scratchpad when the host provides one (-2 if it
 * is too small, -1 if misaligned), otherwise a fresh malloc.  *owned tells the
 * caller whether to free() it.  Returns NULL with *err set on failure.
 */
static inline float *ref_work_buf(const SpiteCtx *ctx, uint64_t n, int *owned, int *err) {
    if (ctx->scratchpad) {
        if (((uintptr_t)ctx->scratchpad & (sizeof(float) - 1)) != 0) { *err = -1; return NULL; }
        if ((uint64_t)ctx->scratchpad_bytes / sizeof(float) < n) { *err = -2; return NULL; }
        *owned = 0;
        return (float *)ctx->scratchpad;
    }
    float *b = (float *)malloc((size_t)n * sizeof(float));
    if (!b) { *err = -1; return NULL; }
    *owned = 1;
    return b;
}
