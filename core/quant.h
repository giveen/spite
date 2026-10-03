/*
 * spite/core/quant.h  —  GGUF quantization block layouts
 *
 * Reference dequantization that kernel authors can use directly or
 * replace with optimized GPU versions.
 *
 * Block layouts match the GGUF spec exactly so kernels can operate
 * directly on mmap'd weight data without repacking.
 */

#pragma once
#include <stdint.h>

/* ── Q8_0 ─────────────────────────────────────────────────────────────
 * 32 int8 values + 1 fp16 scale per block
 */
#define QK8_0 32
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;

/* ── Q4_0 ─────────────────────────────────────────────────────────────
 * 32 4-bit values (packed 2/byte) + 1 fp16 scale per block
 */
#define QK4_0 32
typedef struct { uint16_t d; uint8_t qs[QK4_0 / 2]; } block_q4_0;

/* ── Q4_K ─────────────────────────────────────────────────────────────
 * Superblock of 256 values: 8 sub-blocks of 32, each with its own
 * 6-bit scale. Two fp16 values (d, dmin) scale the sub-block scales.
 *
 * This is the block format for Q4_K_S and Q4_K_M — they differ only
 * in how many bits are used for the sub-block scales (stored outside
 * this struct in the higher-level metadata).
 */
#define QK_K   256
#define K_SCALE_SIZE 12

typedef struct {
    uint16_t d;                  /* super-block scale for quantized scales */
    uint16_t dmin;               /* super-block scale for quantized mins */
    uint8_t  scales[K_SCALE_SIZE]; /* 6-bit scales and mins, packed */
    uint8_t  qs[QK_K / 2];      /* 4-bit quants */
} block_q4_K;

/* ── Q6_K ─────────────────────────────────────────────────────────────
 * 256 values: 6-bit quants + 8-bit scales per 16-value sub-block
 */
typedef struct {
    uint8_t  ql[QK_K / 2];      /* low 4 bits of quants */
    uint8_t  qh[QK_K / 4];      /* high 2 bits of quants */
    int8_t   scales[QK_K / 16]; /* scales, one per 16 quants */
    uint16_t d;
} block_q6_K;

/* ── Reference dequantization (CPU, scalar) ───────────────────────────
 *
 * Kernel authors: implement GPU versions of these that operate on your
 * target architecture's block layout above. The function signatures
 * match what the generic fallback kernel uses — keep them compatible.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* dequantize n blocks into out (fp32). out must hold n * QK8_0 floats. */
void dequant_q8_0 (float *out, const block_q8_0 *blocks, int n);
void dequant_q4_0 (float *out, const block_q4_0 *blocks, int n);
void dequant_q4_K (float *out, const block_q4_K *blocks, int n);
void dequant_q6_K (float *out, const block_q6_K *blocks, int n);

#ifdef __cplusplus
}
#endif
