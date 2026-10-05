/*
 * kernels/_engine/kv_quant/generic/kv_quant_blocks.h
 *
 * Portable C99 implementation of the KV-cache block codecs.
 *
 * This is the single definition of the block layouts that every backend must
 * agree on. It has no CUDA/HIP/Metal/SYCL dependency, so a kernel for any
 * vendor can include it and read/write exactly the bytes the Rust host writes:
 *
 *   | tier  | bytes/block | layout                            |
 *   |-------|-------------|-----------------------------------|
 *   | q8_0  | 34          | f16 d; i8 qs[32]                  |
 *   | q5_1  | 24          | f16 d; f16 m; u32 qh; u8 qs[16]   |
 *   | q4_0  | 18          | f16 d; u8 qs[16]                  |
 *
 * The CUDA kernels implement the same layouts with warp-level reductions in
 * kernels/qwen/qwen3/nvidia/kv_attn.inl; this header is the portable
 * reference for backends that do not have that code.
 *
 * It MUST stay bit-identical to crates/spite-kvcache/src/quant.rs. The
 * `portable_codec_parity` test proves that by compiling kv_quant_parity.c
 * against this header and comparing bytes and decoded values with the Rust
 * codec. Change one without the other and that test fails.
 *
 * Every function works on whole 32-element blocks. Real KV rows are always a
 * multiple of 32 elements (n_kv_heads * head_dim), and the host never asks a
 * kernel to encode a partial block.
 */

#ifndef SPITE_KV_QUANT_BLOCKS_H
#define SPITE_KV_QUANT_BLOCKS_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#define KVQ_BLOCK_ELEMS 32
#define KVQ_Q8_0_BYTES  34
#define KVQ_Q5_1_BYTES  24
#define KVQ_Q4_0_BYTES  18

/* ── float / half helpers ──────────────────────────────────────────────── */

static inline uint32_t kvq_f32_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

static inline float kvq_f32_from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* Round-to-nearest-even f32 -> f16 bit pattern (matches Rust f32_to_f16). */
static inline uint16_t kvq_f32_to_f16_bits(float x) {
    const uint32_t b = kvq_f32_bits(x);
    const uint16_t sign = (uint16_t)((b >> 16) & 0x8000u);
    const int32_t exp = (int32_t)((b >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = b & 0x007FFFFFu;

    /* Inf / NaN: preserve the payload class. */
    if ((b & 0x7F800000u) == 0x7F800000u)
        return (uint16_t)(sign | 0x7C00u | (man != 0 ? 0x0200u : 0u));
    if (exp >= 0x1F)
        return (uint16_t)(sign | 0x7C00u); /* overflow saturates to +inf */
    if (exp <= 0) {
        if (exp < -10)
            return sign; /* underflows to signed zero */
        man |= 0x00800000u;
        {
            const uint32_t shift = (uint32_t)(14 - exp);
            const uint16_t h = (uint16_t)(man >> shift);
            const uint32_t rem = man & ((1u << shift) - 1u);
            const uint32_t half = 1u << (shift - 1u);
            const uint16_t round = (uint16_t)((rem > half) || (rem == half && (h & 1u)));
            return (uint16_t)(sign | (uint16_t)(h + round));
        }
    }
    {
        const uint16_t h = (uint16_t)(((uint32_t)exp << 10) | (man >> 13));
        const uint32_t rem = man & 0x1FFFu;
        const uint16_t round = (uint16_t)((rem > 0x1000u) || (rem == 0x1000u && (h & 1u)));
        return (uint16_t)(sign | (uint16_t)(h + round));
    }
}

/* f16 bit pattern -> f32 (matches Rust f16_to_f32, subnormals included). */
static inline float kvq_f16_bits_to_f32(uint16_t bits) {
    const uint32_t sign = (uint32_t)(bits >> 15) << 31;
    const uint32_t exp = (bits >> 10) & 0x1Fu;
    const uint32_t man = bits & 0x3FFu;

    if (exp == 0) {
        if (man == 0)
            return kvq_f32_from_bits(sign);
        /* Subnormal: man * 2^-24, both exact in f32. */
        {
            const float v = (float)man * kvq_f32_from_bits(0x33800000u);
            return sign ? -v : v;
        }
    }
    if (exp == 0x1Fu)
        return kvq_f32_from_bits(sign | 0x7F800000u | (man << 13));
    return kvq_f32_from_bits(sign | ((exp + 127u - 15u) << 23) | (man << 13));
}

static inline void kvq_store_f16(uint8_t* p, uint16_t bits) {
    p[0] = (uint8_t)(bits & 0xFFu);
    p[1] = (uint8_t)(bits >> 8);
}

static inline uint16_t kvq_load_f16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

/*
 * Rust does `x.round().clamp(lo, hi) as i32`: round half away from zero
 * (`roundf` matches), then clamp — infinities clamp to the bound — then a
 * saturating cast, which turns NaN into 0.
 */
static inline int kvq_round_clamp(float x, int lo, int hi) {
    const float r = roundf(x);
    if (!(r == r))
        return 0;
    if (r < (float)lo)
        return lo;
    if (r > (float)hi)
        return hi;
    return (int)r;
}

/* ── q8_0 ──────────────────────────────────────────────────────────────── */

static inline void kvq_encode_q8_0(const float* blk, uint8_t* dst) {
    float amax = 0.0f;
    float d, inv;
    int i;
    for (i = 0; i < KVQ_BLOCK_ELEMS; ++i) {
        const float a = fabsf(blk[i]);
        if (a > amax)
            amax = a;
    }
    d = amax / 127.0f;
    inv = d > 0.0f ? 1.0f / d : 0.0f;
    kvq_store_f16(dst, kvq_f32_to_f16_bits(d));
    for (i = 0; i < KVQ_BLOCK_ELEMS; ++i) {
        const int q = kvq_round_clamp(blk[i] * inv, -128, 127);
        dst[2 + i] = (uint8_t)(int8_t)q;
    }
}

static inline void kvq_decode_q8_0(const uint8_t* src, float* blk) {
    const float d = kvq_f16_bits_to_f32(kvq_load_f16(src));
    int i;
    for (i = 0; i < KVQ_BLOCK_ELEMS; ++i)
        blk[i] = (float)(int8_t)src[2 + i] * d;
}

/* ── q5_1 ──────────────────────────────────────────────────────────────── */

static inline void kvq_encode_q5_1(const float* blk, uint8_t* dst) {
    float mn = blk[0], mx = blk[0];
    float d, inv;
    uint32_t qh = 0;
    uint8_t qs[16];
    int i;

    for (i = 1; i < KVQ_BLOCK_ELEMS; ++i) {
        if (blk[i] < mn)
            mn = blk[i];
        if (blk[i] > mx)
            mx = blk[i];
    }
    d = (mx - mn) / 31.0f;
    inv = d > 0.0f ? 1.0f / d : 0.0f;

    kvq_store_f16(dst, kvq_f32_to_f16_bits(d));
    kvq_store_f16(dst + 2, kvq_f32_to_f16_bits(mn));

    memset(qs, 0, sizeof qs);
    for (i = 0; i < KVQ_BLOCK_ELEMS; ++i) {
        const int q = kvq_round_clamp((blk[i] - mn) * inv, 0, 31);
        if (i < 16)
            qs[i] = (uint8_t)(q & 0x0F);
        else
            qs[i - 16] |= (uint8_t)((q & 0x0F) << 4);
        if (q & 0x10)
            qh |= 1u << i;
    }
    dst[4] = (uint8_t)(qh & 0xFFu);
    dst[5] = (uint8_t)((qh >> 8) & 0xFFu);
    dst[6] = (uint8_t)((qh >> 16) & 0xFFu);
    dst[7] = (uint8_t)((qh >> 24) & 0xFFu);
    memcpy(dst + 8, qs, sizeof qs);
}

static inline void kvq_decode_q5_1(const uint8_t* src, float* blk) {
    const float d = kvq_f16_bits_to_f32(kvq_load_f16(src));
    const float m = kvq_f16_bits_to_f32(kvq_load_f16(src + 2));
    const uint32_t qh = (uint32_t)src[4] | ((uint32_t)src[5] << 8) | ((uint32_t)src[6] << 16) |
                        ((uint32_t)src[7] << 24);
    int i;
    for (i = 0; i < KVQ_BLOCK_ELEMS; ++i) {
        const uint8_t byte = src[8 + (i % 16)];
        const uint32_t lo = (i < 16) ? (uint32_t)(byte & 0x0Fu) : (uint32_t)(byte >> 4);
        const uint32_t q = lo | (((qh >> i) & 1u) << 4);
        blk[i] = (float)q * d + m;
    }
}

/* ── q4_0 ──────────────────────────────────────────────────────────────── */

static inline uint8_t kvq_q4_0_nibble(float x, float inv) {
    const int q = kvq_round_clamp(x * inv, -8, 7);
    return (uint8_t)((q + 8) & 0x0F);
}

static inline void kvq_encode_q4_0(const float* blk, uint8_t* dst) {
    float amax = 0.0f;
    float d, inv;
    int i;
    for (i = 0; i < KVQ_BLOCK_ELEMS; ++i) {
        const float a = fabsf(blk[i]);
        if (a > amax)
            amax = a;
    }
    d = amax / 8.0f;
    inv = d > 0.0f ? 1.0f / d : 0.0f;
    kvq_store_f16(dst, kvq_f32_to_f16_bits(d));
    for (i = 0; i < 16; ++i) {
        const uint8_t lo = kvq_q4_0_nibble(blk[i], inv);
        const uint8_t hi = kvq_q4_0_nibble(blk[i + 16], inv);
        dst[2 + i] = (uint8_t)(lo | (uint8_t)(hi << 4));
    }
}

static inline void kvq_decode_q4_0(const uint8_t* src, float* blk) {
    const float d = kvq_f16_bits_to_f32(kvq_load_f16(src));
    int i;
    for (i = 0; i < 16; ++i) {
        const uint8_t byte = src[2 + i];
        const int lo = (int)(byte & 0x0Fu) - 8;
        const int hi = (int)(byte >> 4) - 8;
        blk[i] = (float)lo * d;
        blk[i + 16] = (float)hi * d;
    }
}

#endif /* SPITE_KV_QUANT_BLOCKS_H */
