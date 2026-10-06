/*
 * tools/verify/quant_oracle.c — bit-exact check of spite_dequantize_row()
 * against llama.cpp's ggml type traits (to_float) for every SpiteType.
 *
 * Build (from repo root):
 *   L=/mnt/storage/llama.cpp
 *   gcc -std=c11 -O2 -Wall -Wextra -Icore -I$L/ggml/include \
 *       tools/verify/quant_oracle.c core/quant.c -lm \
 *       -L$L/build/bin -lggml-base -Wl,-rpath,$L/build/bin -o /tmp/quant_oracle
 *   /tmp/quant_oracle
 *
 * Per type: blocks are filled with seeded random bytes; every fp16 scale
 * field is then replaced by a finite value (finite-scale pass, must match
 * bit-exactly). A second pass leaves the random bytes untouched (NaN/Inf
 * scales included); there NaN==NaN (any payload) is accepted, everything
 * else must match bit-exactly. Exit status is non-zero on any failure.
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"
#include "ggml.h"

#define NBLK 64

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

static uint16_t sane_half(void) {
    return (uint16_t)((rnd() & 0x8000) | ((8 + rnd() % 9) << 10) | (rnd() & 0x3FF));
}

static void put16(uint8_t *b, size_t off) { uint16_t h = sane_half(); memcpy(b + off, &h, 2); }

/* Overwrite all scale fields of one block with finite values. */
static void patch(SpiteType t, uint8_t *b) {
    switch (t) {
    case SPITE_TYPE_F32: { float f = (float)(int)(rnd() % 2001 - 1000) / 7.0f; memcpy(b, &f, 4); break; }
    case SPITE_TYPE_F16: put16(b, 0); break;
    case SPITE_TYPE_BF16: b[1] = (uint8_t)((rnd() & 0x80) | (0x30 + rnd() % 0x20)); break;
    case SPITE_TYPE_Q4_0: put16(b, offsetof(block_q4_0, d)); break;
    case SPITE_TYPE_Q4_1: put16(b, offsetof(block_q4_1, d)); put16(b, offsetof(block_q4_1, m)); break;
    case SPITE_TYPE_Q5_0: put16(b, offsetof(block_q5_0, d)); break;
    case SPITE_TYPE_Q5_1: put16(b, offsetof(block_q5_1, d)); put16(b, offsetof(block_q5_1, m)); break;
    case SPITE_TYPE_Q8_0: put16(b, offsetof(block_q8_0, d)); break;
    case SPITE_TYPE_Q1_0: put16(b, offsetof(block_q1_0, d)); break;
    case SPITE_TYPE_Q2_0: put16(b, offsetof(block_q2_0, d)); break;
    case SPITE_TYPE_Q2_K: put16(b, offsetof(block_q2_K, d)); put16(b, offsetof(block_q2_K, dmin)); break;
    case SPITE_TYPE_Q3_K: put16(b, offsetof(block_q3_K, d)); break;
    case SPITE_TYPE_Q4_K: put16(b, offsetof(block_q4_K, d)); put16(b, offsetof(block_q4_K, dmin)); break;
    case SPITE_TYPE_Q5_K: put16(b, offsetof(block_q5_K, d)); put16(b, offsetof(block_q5_K, dmin)); break;
    case SPITE_TYPE_Q6_K: put16(b, offsetof(block_q6_K, d)); break;
    case SPITE_TYPE_IQ2_XXS: put16(b, offsetof(block_iq2_xxs, d)); break;
    case SPITE_TYPE_IQ2_XS: put16(b, offsetof(block_iq2_xs, d)); break;
    case SPITE_TYPE_IQ2_S: put16(b, offsetof(block_iq2_s, d)); break;
    case SPITE_TYPE_IQ3_XXS: put16(b, offsetof(block_iq3_xxs, d)); break;
    case SPITE_TYPE_IQ3_S: put16(b, offsetof(block_iq3_s, d)); break;
    case SPITE_TYPE_IQ1_S: put16(b, offsetof(block_iq1_s, d)); break;
    case SPITE_TYPE_IQ1_M: {
        /* fp16 scale is spread over the top nibble of each of 4 uint16 scales */
        uint16_t h = sane_half(), sc[4];
        memcpy(sc, b + offsetof(block_iq1_m, scales), 8);
        for (int i = 0; i < 4; i++) sc[i] = (uint16_t)((sc[i] & 0x0FFF) | ((h >> (4 * i)) & 0xF) << 12);
        memcpy(b + offsetof(block_iq1_m, scales), sc, 8);
        break;
    }
    case SPITE_TYPE_IQ4_NL: put16(b, offsetof(block_iq4_nl, d)); break;
    case SPITE_TYPE_IQ4_XS: put16(b, offsetof(block_iq4_xs, d)); break;
    case SPITE_TYPE_TQ1_0: put16(b, offsetof(block_tq1_0, d)); break;
    case SPITE_TYPE_TQ2_0: put16(b, offsetof(block_tq2_0, d)); break;
    case SPITE_TYPE_MXFP4:  /* E8M0 / UE4M3 scales: every byte value is finite */
    case SPITE_TYPE_NVFP4: break;
    default: break;
    }
}

static int same(const float *a, const float *b, int64_t n, int allow_nan, int64_t *bad) {
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(a + i, b + i, 4) == 0) continue;
        if (allow_nan && isnan(a[i]) && isnan(b[i])) continue;
        *bad = i;
        return 0;
    }
    return 1;
}

static const SpiteType ALL[] = {
    SPITE_TYPE_F32, SPITE_TYPE_F16, SPITE_TYPE_Q4_0, SPITE_TYPE_Q4_1, SPITE_TYPE_Q5_0,
    SPITE_TYPE_Q5_1, SPITE_TYPE_Q8_0, SPITE_TYPE_Q2_K, SPITE_TYPE_Q3_K, SPITE_TYPE_Q4_K,
    SPITE_TYPE_Q5_K, SPITE_TYPE_Q6_K, SPITE_TYPE_IQ2_XXS, SPITE_TYPE_IQ2_XS,
    SPITE_TYPE_IQ3_XXS, SPITE_TYPE_IQ1_S, SPITE_TYPE_IQ4_NL, SPITE_TYPE_IQ3_S,
    SPITE_TYPE_IQ2_S, SPITE_TYPE_IQ4_XS, SPITE_TYPE_IQ1_M, SPITE_TYPE_BF16,
    SPITE_TYPE_TQ1_0, SPITE_TYPE_TQ2_0, SPITE_TYPE_MXFP4, SPITE_TYPE_NVFP4,
    SPITE_TYPE_Q1_0, SPITE_TYPE_Q2_0,
};

int main(void) {
    int fails = 0, npass = 0;
    for (size_t ti = 0; ti < sizeof ALL / sizeof *ALL; ti++) {
        SpiteType t = ALL[ti];
        const char *name = ggml_type_name((enum ggml_type)t);
        const char *why = NULL;
        int64_t bad = -1;

        const struct ggml_type_traits *tr = ggml_get_type_traits((enum ggml_type)t);
        uint64_t sb = spite_type_block_bytes(t), se = spite_type_block_elements(t);
        if ((uint64_t)ggml_type_size((enum ggml_type)t) != sb || (uint64_t)ggml_blck_size((enum ggml_type)t) != se)
            why = "block bytes/elements differ from ggml";
        else if (!tr || (!tr->to_float && t != SPITE_TYPE_F32))
            why = "ggml has no to_float for this type";
        else {
            /* ggml's F32 traits have no to_float (identity); reported below, checked against memcpy */
            if (!tr->to_float) printf("note: ggml has no to_float for %s; reference is identity copy\n", name);
            int64_t n = (int64_t)(se * NBLK);
            uint8_t *src = malloc(sb * NBLK);
            float *a = malloc((size_t)n * 4), *b = malloc((size_t)n * 4);
            for (int pass = 0; pass < 2 && !why; pass++) {   /* 0 = finite scales, 1 = raw random */
                for (size_t i = 0; i < sb * NBLK; i++) src[i] = (uint8_t)rnd();
                if (pass == 0) for (int k = 0; k < NBLK; k++) patch(t, src + k * sb);
                memset(a, 0xA5, (size_t)n * 4); memset(b, 0x5A, (size_t)n * 4);
                if (spite_dequantize_row(t, src, a, n) != 0) { why = "spite_dequantize_row returned error"; break; }
                if (tr->to_float) tr->to_float(src, b, n); else memcpy(b, src, (size_t)n * 4);
                if (!same(a, b, n, pass == 1, &bad)) why = pass == 0 ? "mismatch (finite-scale pass)" : "mismatch (raw pass)";
            }
            if (!why && se > 1 && (spite_dequantize_row(t, src, a, n + 1) == 0 || spite_dequantize_row(t, src, a, (int64_t)se - 1) == 0))
                why = "n not multiple of block size was accepted";
            free(src); free(a); free(b);
        }
        if (why) { fails++; printf("FAIL %-8s %s (first bad idx %lld)\n", name, why, (long long)bad); }
        else { npass++; printf("PASS %-8s\n", name); }
    }
    if (spite_dequantize_row((SpiteType)9999, "", (float[1]){0}, 1) != -1) { fails++; puts("FAIL unsupported type not rejected"); }
    printf("%d PASS, %d FAIL\n", npass, fails);
    return fails != 0;
}
