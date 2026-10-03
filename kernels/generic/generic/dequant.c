/*
 * kernels/generic/generic/dequant.c
 *
 * Reference scalar dequantization — implements the declarations in
 * core/quant.h.  These are the ground-truth implementations that all
 * GPU kernels must produce bit-identical results to (within fp32 ULP).
 *
 * Performance is intentionally ignored: correctness is the only goal.
 * GPU kernel authors should verify their implementations against these.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../../../core/quant.h"

/* ── fp16 → fp32 ─────────────────────────────────────────────────────── */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        /* denormalized */
        if (mant == 0) {
            float z = 0.0f;
            uint32_t bits = sign;
            memcpy(&z, &bits, sizeof z);
            return z;
        }
        float f = (float)mant * (1.0f / 1024.0f);
        f = ldexpf(f, -14);
        return sign ? -f : f;
    }
    if (exp == 31) {
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        float f;
        memcpy(&f, &bits, sizeof f);
        return f;
    }
    uint32_t bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* ── Q8_0 ─────────────────────────────────────────────────────────────── */

void dequant_q8_0(float *out, const block_q8_0 *blocks, int n) {
    for (int b = 0; b < n; b++) {
        const float d = f16_to_f32(blocks[b].d);
        for (int i = 0; i < QK8_0; i++) {
            out[b * QK8_0 + i] = (float)blocks[b].qs[i] * d;
        }
    }
}

/* ── Q4_0 ─────────────────────────────────────────────────────────────── */

void dequant_q4_0(float *out, const block_q4_0 *blocks, int n) {
    for (int b = 0; b < n; b++) {
        const float d = f16_to_f32(blocks[b].d);
        for (int i = 0; i < QK4_0 / 2; i++) {
            uint8_t byte = blocks[b].qs[i];
            /* 4-bit signed: subtract 8 to center around zero */
            int8_t lo = (int8_t)((int)(byte & 0x0F) - 8);
            int8_t hi = (int8_t)((int)(byte >>    4) - 8);
            out[b * QK4_0 + 2 * i    ] = (float)lo * d;
            out[b * QK4_0 + 2 * i + 1] = (float)hi * d;
        }
    }
}

/* ── Q4_K helpers ─────────────────────────────────────────────────────── */

/*
 * Extract the 6-bit scale and min for sub-block `j` from block_q4_K.scales[].
 *
 * Packing (from GGUF spec):
 *   j < 4: scale = scales[j] & 0x3F;  min = scales[j+4] & 0x3F
 *   j >= 4: scale = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
 *            min  = (scales[j+4] >> 4)   | ((scales[j-0] >> 6) << 4)
 */
static void get_scale_min_q4k(int j, const uint8_t *scales,
                               uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = scales[j    ] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (uint8_t)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
        *m  = (uint8_t)((scales[j + 4] >>    4) | ((scales[j - 0] >> 6) << 4));
    }
}

/* ── Q4_K ─────────────────────────────────────────────────────────────── */

void dequant_q4_K(float *out, const block_q4_K *blocks, int n) {
    for (int b = 0; b < n; b++) {
        const float d    = f16_to_f32(blocks[b].d);
        const float dmin = f16_to_f32(blocks[b].dmin);

        const uint8_t *qs     = blocks[b].qs;
        const uint8_t *scales = blocks[b].scales;
        float *y = out + b * QK_K;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_q4k(is,     scales, &sc1, &m1);
            get_scale_min_q4k(is + 1, scales, &sc2, &m2);

            const float d1 = d * (float)sc1;
            const float m1f = dmin * (float)m1;
            const float d2 = d * (float)sc2;
            const float m2f = dmin * (float)m2;

            /* First 32 values use low nibble */
            for (int l = 0; l < 32; l++) {
                y[l] = d1 * (float)(qs[l] & 0x0F) - m1f;
            }
            /* Next 32 values use high nibble */
            for (int l = 0; l < 32; l++) {
                y[32 + l] = d2 * (float)(qs[l] >> 4) - m2f;
            }

            y  += 64;
            qs += 32;
            is += 2;
        }
    }
}

/* ── Q6_K ─────────────────────────────────────────────────────────────── */

void dequant_q6_K(float *out, const block_q6_K *blocks, int n) {
    for (int b = 0; b < n; b++) {
        const float d = f16_to_f32(blocks[b].d);

        const uint8_t *ql     = blocks[b].ql;
        const uint8_t *qh     = blocks[b].qh;
        const int8_t  *sc     = blocks[b].scales;
        float         *y      = out + b * QK_K;

        for (int i = 0; i < QK_K / 16; i++) {
            const float scale = d * (float)sc[i];
            /* Each group of 16 draws from 8 bytes of ql and 4 bytes of qh */
            for (int l = 0; l < 16; l++) {
                int idx = i * 16 + l;
                /* low 4 bits from ql; high 2 bits from qh */
                uint8_t ql_byte = ql[idx / 2];
                uint8_t lo4 = (l & 1) ? (ql_byte >> 4) : (ql_byte & 0x0F);
                uint8_t hi2 = (qh[idx / 4] >> (2 * (idx % 4))) & 0x03;
                int8_t  q6  = (int8_t)((int)(lo4 | (hi2 << 4)) - 32);
                y[idx] = scale * (float)q6;
            }
        }
    }
}
