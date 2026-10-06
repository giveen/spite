/*
 * spite/core/gpu/quant_dequant.h — GPU-side decoding of every SpiteType.
 *
 * Vendor-neutral and cross-model: the same source builds with nvcc (CUDA) and
 * hipcc (HIP/ROCm).  The backend is picked from the compiler (__HIPCC__ →
 * HIP, else CUDA); only the intrinsics below (SQ_F, __fmul_rn, __half2float,
 * fp16 header) differ and both toolchains spell them identically.  Metal/SYCL
 * need their own shim (ggml-common.h has DECL/IMPL variants for them) and the
 * same slot tables; the per-type layouts here are the specification.
 *
 * Header-only.  Include at GLOBAL scope from a .cu/.hip file.  Everything is
 * `static __device__ __forceinline__`, class templates or `inline` host
 * functions, so any number of translation units may include it (no ODR /
 * multiple-definition problems).  Lookup tables come from core/ggml-common.h
 * (GGML_COMMON_IMPL_*: `static const __device__`).
 *
 * NOTE: ggml-common.h #defines QK4_0, QK8_0, QK_K, QI*, QR*, ... as macros.
 * A kernel that includes this header must not declare identifiers with those
 * names.
 *
 * Decode model (same idea as ggml's dequantize.cuh: one lane decodes a few
 * elements of a block).  For every block type Q:
 *
 *   Q::QK, Q::BYTES   elements / bytes per block (== spite_type_block_*)
 *   Q::NG             4-element groups decoded per slot (8 or 16 or 4 values)
 *   Q::NS             slots per block = QK / (4*NG); always divides 32
 *   Q::start(s, g)    element index (within the block) of group g's first
 *                     element; the 4 elements of a group are consecutive
 *   Q::decode(blk, s, v)
 *                     v[4*g + k] = weight at element start(s, g) + k
 *
 * Slot s in [0, NS) of one block covers each of the QK elements exactly once.
 * Every product is an explicit __fmul_rn/__fadd_rn/__fsub_rn in the exact order
 * of the scalar reference (core/quant.c == ggml-quants.c), so the decoded
 * floats are bit-identical to spite_dequantize_row() — never contracted into an
 * FMA. (The device test, tools/verify/quant_gpu_test.cu, proves this for all
 * types on CUDA; the HIP path is the same source, not built here.)
 *
 * Alignment: block addresses must be 2-byte aligned (MXFP4/NVFP4: byte loads
 * only, any alignment); F32 rows 4-byte aligned.  See Q::ALIGN.
 *
 * Caveat: --use_fast_math (ftz) flushes denormal FLOAT operands/results.
 * Only an MXFP4 scale byte of 0 or 1 (2^-128, 2^-127) can reach that range;
 * no real model has it.  Everything else is exact incl. fp16 subnormal scales.
 *
 * Derived from llama.cpp ggml-cuda (MIT); see core/THIRD_PARTY.md.
 */
#pragma once

#if defined(__HIPCC__)
#ifndef GGML_COMMON_DECL_HIP
#define GGML_COMMON_DECL_HIP
#endif
#ifndef GGML_COMMON_IMPL_HIP
#define GGML_COMMON_IMPL_HIP
#endif
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#ifndef GGML_COMMON_DECL_CUDA
#define GGML_COMMON_DECL_CUDA
#endif
#ifndef GGML_COMMON_IMPL_CUDA
#define GGML_COMMON_IMPL_CUDA
#endif
#endif
#include "../abi.h"
#include "../ggml-common.h"

#if !defined(__HIPCC__)
#include <cuda_fp16.h>
#endif
#include <stdint.h>

namespace sq {

#define SQ_F static __device__ __forceinline__

// ── scalar helpers ────────────────────────────────────────────────────────

SQ_F uint32_t ld8(const uint8_t *p) { return *p; }
/* p must be 2-byte aligned (all block layouts keep every ld16 offset even). */
SQ_F uint32_t ld16(const uint8_t *p) {
  return *reinterpret_cast<const uint16_t *>(p);
}
SQ_F uint32_t ld32(const uint8_t *p) { return ld16(p) | (ld16(p + 2) << 16); }
SQ_F float h2f(uint32_t h) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(h)));
}
SQ_F float fm(float a, float b) { return __fmul_rn(a, b); }
SQ_F float fa(float a, float b) { return __fadd_rn(a, b); }
SQ_F float fs(float a, float b) { return __fsub_rn(a, b); }
SQ_F float sgn(float v, uint32_t neg) { return neg ? -v : v; }
SQ_F uint32_t byte_of(uint32_t w, int k) { return (w >> (8 * k)) & 0xFFu; }

/* E8M0 / 2 (ggml_e8m0_to_fp32_half). */
SQ_F float e8m0_half(uint32_t x) {
  return __uint_as_float(x < 2 ? (0x00200000u << x) : ((x - 1) << 23));
}

/* UE4M3 * 0.5 (ggml_ue4m3_to_fp32); the 2^n factors are exact bit patterns. */
SQ_F float ue4m3_half(uint32_t x) {
  if (x == 0 || x == 0x7F)
    return 0.0f;
  const uint32_t e = (x >> 3) & 0xF, m = x & 7;
  const float raw =
      e == 0 ? fm(static_cast<float>(m), 0.001953125f) /* m * 2^-9 */
             : fm(1.0f + static_cast<float>(m) * 0.125f,
                  __uint_as_float((e - 7 + 127) << 23)); /* * 2^(e-7) */
  return fm(raw, 0.5f);
}

/* 6-bit scale/min pair of Q4_K/Q5_K (get_scale_min_k4). */
SQ_F void scale_min_k4(int j, const uint8_t *q, uint32_t &d, uint32_t &m) {
  if (j < 4) {
    d = ld8(q + j) & 63;
    m = ld8(q + j + 4) & 63;
  } else {
    d = (ld8(q + j + 4) & 0xF) | ((ld8(q + j - 4) >> 6) << 4);
    m = (ld8(q + j + 4) >> 4) | ((ld8(q + j) >> 6) << 4);
  }
}

/* Q3_K 6-bit scale `is` (0..15), unbiased (caller subtracts 32). */
SQ_F int scale_q3k(const uint8_t *raw, int is) {
  const int q = is >> 2, i = is & 3;
  const uint32_t lo = ld8(raw + 4 * (q & 1) + i);
  const uint32_t hi = (ld8(raw + 8 + i) >> (2 * q)) & 3;
  return static_cast<int>((q < 2 ? (lo & 0xF) : (lo >> 4)) | (hi << 4));
}

/* Load 8 consecutive bytes (2-aligned) as two 32-bit words. */
SQ_F void ld8x8(const uint8_t *p, uint32_t &lo, uint32_t &hi) {
  lo = ld32(p);
  hi = ld32(p + 4);
}

// ── type descriptors ──────────────────────────────────────────────────────

template <SpiteType ID_, int QK_, int BYTES_, int NG_, int ALIGN_ = 2>
struct QBase {
  static constexpr SpiteType id = ID_;
  static constexpr int QK = QK_, BYTES = BYTES_, NG = NG_, NV = 4 * NG_,
                       NS = QK_ / (4 * NG_);
  static constexpr int ALIGN =
      ALIGN_; /* required alignment of the weight base pointer */
  static constexpr bool DENSE = false;
  static_assert(QK_ % (4 * NG_) == 0 && 32 % (QK_ / (4 * NG_)) == 0,
                "slots per block must divide the warp");
};

/* 4-bit pairs: byte j holds elements j (low) and j+16 (high). */
#define SQ_NIB(q, k) ((q >> (8 * (k))) & 0xFu)
#define SQ_NIH(q, k) ((q >> (8 * (k) + 4)) & 0xFu)

struct Q4_0 : QBase<SPITE_TYPE_Q4_0, QK4_0, 18, 2> {
  static_assert(sizeof(block_q4_0) == BYTES, "q4_0");
  SQ_F int start(int s, int g) { return 4 * s + 16 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b));
    const uint32_t q = ld32(b + 2 + 4 * s);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[k] = fm(static_cast<float>(static_cast<int>(SQ_NIB(q, k)) - 8), d);
      v[4 + k] = fm(static_cast<float>(static_cast<int>(SQ_NIH(q, k)) - 8), d);
    }
  }
};

struct Q4_1 : QBase<SPITE_TYPE_Q4_1, QK4_1, 20, 2> {
  static_assert(sizeof(block_q4_1) == BYTES, "q4_1");
  SQ_F int start(int s, int g) { return 4 * s + 16 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b)), m = h2f(ld16(b + 2));
    const uint32_t q = ld32(b + 4 + 4 * s);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[k] = fa(fm(static_cast<float>(SQ_NIB(q, k)), d), m);
      v[4 + k] = fa(fm(static_cast<float>(SQ_NIH(q, k)), d), m);
    }
  }
};

struct Q5_0 : QBase<SPITE_TYPE_Q5_0, QK5_0, 22, 2> {
  static_assert(sizeof(block_q5_0) == BYTES, "q5_0");
  SQ_F int start(int s, int g) { return 4 * s + 16 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b));
    const uint32_t qh = ld32(b + 2);
    const uint32_t q = ld32(b + 6 + 4 * s);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int j = 4 * s + k;
      const uint32_t xh0 = ((qh >> j) << 4) & 0x10,
                     xh1 = (qh >> (j + 12)) & 0x10;
      v[k] =
          fm(static_cast<float>(static_cast<int>(SQ_NIB(q, k) | xh0) - 16), d);
      v[4 + k] =
          fm(static_cast<float>(static_cast<int>(SQ_NIH(q, k) | xh1) - 16), d);
    }
  }
};

struct Q5_1 : QBase<SPITE_TYPE_Q5_1, QK5_1, 24, 2> {
  static_assert(sizeof(block_q5_1) == BYTES, "q5_1");
  SQ_F int start(int s, int g) { return 4 * s + 16 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b)), m = h2f(ld16(b + 2));
    const uint32_t qh = ld32(b + 4);
    const uint32_t q = ld32(b + 8 + 4 * s);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int j = 4 * s + k;
      const uint32_t xh0 = ((qh >> j) << 4) & 0x10,
                     xh1 = (qh >> (j + 12)) & 0x10;
      v[k] = fa(fm(static_cast<float>(SQ_NIB(q, k) | xh0), d), m);
      v[4 + k] = fa(fm(static_cast<float>(SQ_NIH(q, k) | xh1), d), m);
    }
  }
};

struct Q8_0 : QBase<SPITE_TYPE_Q8_0, QK8_0, 34, 2> {
  static_assert(sizeof(block_q8_0) == BYTES, "q8_0");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b));
    uint32_t q0, q1;
    ld8x8(b + 2 + 8 * s, q0, q1);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[k] = fm(static_cast<float>(static_cast<int8_t>(byte_of(q0, k))), d);
      v[4 + k] = fm(static_cast<float>(static_cast<int8_t>(byte_of(q1, k))), d);
    }
  }
};

struct Q1_0 : QBase<SPITE_TYPE_Q1_0, QK1_0, 18, 2> {
  static_assert(sizeof(block_q1_0) == BYTES, "q1_0");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b));
    const uint32_t by = ld8(b + 2 + s);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      v[k] = ((by >> k) & 1) ? d : -d;
  }
};

struct Q2_0 : QBase<SPITE_TYPE_Q2_0, QK2_0, 18, 2> {
  static_assert(sizeof(block_q2_0) == BYTES, "q2_0");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b));
    const uint32_t w = ld16(b + 2 + 2 * s);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      v[k] =
          fm(static_cast<float>(static_cast<int>((w >> (2 * k)) & 3) - 1), d);
  }
};

struct MXFP4 : QBase<SPITE_TYPE_MXFP4, QK_MXFP4, 17, 2, 1> {
  static_assert(sizeof(block_mxfp4) == BYTES, "mxfp4");
  SQ_F int start(int s, int g) { return 4 * s + 16 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = e8m0_half(ld8(b));
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t by = ld8(b + 1 + 4 * s + k);
      v[k] = fm(static_cast<float>(kvalues_mxfp4[by & 0xF]), d);
      v[4 + k] = fm(static_cast<float>(kvalues_mxfp4[by >> 4]), d);
    }
  }
};

struct NVFP4 : QBase<SPITE_TYPE_NVFP4, QK_NVFP4, 36, 2, 1> {
  static_assert(sizeof(block_nvfp4) == BYTES, "nvfp4");
  /* slot = (16-element sub-block s>>1, half s&1): bytes 4 per slot */
  SQ_F int start(int s, int g) { return 16 * (s >> 1) + 4 * (s & 1) + 8 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = ue4m3_half(ld8(b + (s >> 1)));
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t by = ld8(b + 4 + 8 * (s >> 1) + 4 * (s & 1) + k);
      v[k] = fm(static_cast<float>(kvalues_mxfp4[by & 0xF]), d);
      v[4 + k] = fm(static_cast<float>(kvalues_mxfp4[by >> 4]), d);
    }
  }
};

// ── ternary ───────────────────────────────────────────────────────────────

struct TQ2_0 : QBase<SPITE_TYPE_TQ2_0, QK_K, 66, 2> {
  static_assert(sizeof(block_tq2_0) == BYTES, "tq2_0");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b + 64));
    const int jj = s >> 4, l = (s >> 2) & 3, m0 = 8 * (s & 3);
    uint32_t q0, q1;
    ld8x8(b + 32 * jj + m0, q0, q1);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[k] = fm(static_cast<float>(
                    static_cast<int>((byte_of(q0, k) >> (2 * l)) & 3) - 1),
                d);
      v[4 + k] = fm(static_cast<float>(
                        static_cast<int>((byte_of(q1, k) >> (2 * l)) & 3) - 1),
                    d);
    }
  }
};

struct TQ1_0 : QBase<SPITE_TYPE_TQ1_0, QK_K, 54, 2> {
  static_assert(sizeof(block_tq1_0) == BYTES, "tq1_0");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  /* byte * 3^n mod 256, then the top "trit": ((q*3)>>8) - 1 */
  SQ_F float trit(uint32_t by, int n, float d) {
    const uint32_t p3 = n == 0   ? 1u
                        : n == 1 ? 3u
                        : n == 2 ? 9u
                        : n == 3 ? 27u
                        : n == 4 ? 81u
                                 : 243u;
    const uint32_t q = (by * p3) & 0xFFu;
    return fm(static_cast<float>(static_cast<int>((q * 3) >> 8) - 1), d);
  }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b + 52));
    uint32_t q0, q1;
    if (s < 20) { /* elements 0..159:   n = e/32, byte qs[e%32] */
      ld8x8(b + 8 * (s & 3), q0, q1);
      const int n = s >> 2;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        v[k] = trit(byte_of(q0, k), n, d);
        v[4 + k] = trit(byte_of(q1, k), n, d);
      }
    } else if (s < 30) { /* elements 160..239: n = (e-160)/16, byte qs[32 +
                            (e-160)%16] */
      const int t = s - 20;
      ld8x8(b + 32 + 8 * (t & 1), q0, q1);
      const int n = t >> 1;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        v[k] = trit(byte_of(q0, k), n, d);
        v[4 + k] = trit(byte_of(q1, k), n, d);
      }
    } else { /* elements 240..255: n = (e-240)/4, byte qh[(e-240)%4] */
      const uint32_t qh = ld32(b + 48);
      const int n0 = 2 * (s - 30);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        v[k] = trit(byte_of(qh, k), n0, d);
        v[4 + k] = trit(byte_of(qh, k), n0 + 1, d);
      }
    }
  }
};

// ── K-quants ──────────────────────────────────────────────────────────────

/* Q2_K / Q3_K: slot = (128-half nn, 16-half h, 4-byte group l4); the same 4
 * bytes serve the four 2-bit planes j (elements 128nn + 32j + 16h + 4l4 + k).
 */
struct Q2_K : QBase<SPITE_TYPE_Q2_K, QK_K, 84, 4> {
  static_assert(sizeof(block_q2_K) == BYTES, "q2_K");
  SQ_F int start(int s, int g) {
    return 128 * (s >> 3) + 32 * g + 16 * ((s >> 2) & 1) + 4 * (s & 3);
  }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int nn = s >> 3, h = (s >> 2) & 1, l4 = s & 3;
    const float d = h2f(ld16(b + 80)), mn = h2f(ld16(b + 82));
    const uint32_t q = ld32(b + 16 + 32 * nn + 16 * h + 4 * l4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t sc = ld8(b + 8 * nn + 2 * j + h);
      const float dl = fm(d, static_cast<float>(sc & 0xF)),
                  ml = fm(mn, static_cast<float>(sc >> 4));
#pragma unroll
      for (int k = 0; k < 4; ++k)
        v[4 * j + k] =
            fs(fm(dl, static_cast<float>((q >> (8 * k + 2 * j)) & 3)), ml);
    }
  }
};

struct Q3_K : QBase<SPITE_TYPE_Q3_K, QK_K, 110, 4> {
  static_assert(sizeof(block_q3_K) == BYTES, "q3_K");
  SQ_F int start(int s, int g) {
    return 128 * (s >> 3) + 32 * g + 16 * ((s >> 2) & 1) + 4 * (s & 3);
  }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int nn = s >> 3, h = (s >> 2) & 1, l4 = s & 3;
    const float d = h2f(ld16(b + 108));
    const uint32_t q = ld32(b + 32 + 32 * nn + 16 * h + 4 * l4);
    const uint32_t hm = ld32(b + 16 * h + 4 * l4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float dl =
          fm(d, static_cast<float>(scale_q3k(b + 96, 8 * nn + 2 * j + h) - 32));
      const int mbit = 4 * nn + j;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int qv = static_cast<int>((q >> (8 * k + 2 * j)) & 3) -
                       static_cast<int>(((hm >> (8 * k + mbit)) & 1) ? 0 : 4);
        v[4 * j + k] = fm(dl, static_cast<float>(qv));
      }
    }
  }
};

/* Q4_K / Q5_K: slot = (64-element pair jj, 4-byte group l4); low nibbles are
 * elements 64jj + 4l4 + k, high nibbles are 32 further. */
struct Q4_K : QBase<SPITE_TYPE_Q4_K, QK_K, 144, 2> {
  static_assert(sizeof(block_q4_K) == BYTES, "q4_K");
  SQ_F int start(int s, int g) { return 64 * (s >> 3) + 32 * g + 4 * (s & 7); }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int jj = s >> 3, l4 = s & 7;
    const float d = h2f(ld16(b)), mn = h2f(ld16(b + 2));
    uint32_t sc1, m1, sc2, m2;
    scale_min_k4(2 * jj, b + 4, sc1, m1);
    scale_min_k4(2 * jj + 1, b + 4, sc2, m2);
    const float d1 = fm(d, static_cast<float>(sc1)),
                mm1 = fm(mn, static_cast<float>(m1));
    const float d2 = fm(d, static_cast<float>(sc2)),
                mm2 = fm(mn, static_cast<float>(m2));
    const uint32_t q = ld32(b + 16 + 32 * jj + 4 * l4);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[k] = fs(fm(d1, static_cast<float>(SQ_NIB(q, k))), mm1);
      v[4 + k] = fs(fm(d2, static_cast<float>(SQ_NIH(q, k))), mm2);
    }
  }
};

struct Q5_K : QBase<SPITE_TYPE_Q5_K, QK_K, 176, 2> {
  static_assert(sizeof(block_q5_K) == BYTES, "q5_K");
  SQ_F int start(int s, int g) { return 64 * (s >> 3) + 32 * g + 4 * (s & 7); }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int jj = s >> 3, l4 = s & 7;
    const float d = h2f(ld16(b)), mn = h2f(ld16(b + 2));
    uint32_t sc1, m1, sc2, m2;
    scale_min_k4(2 * jj, b + 4, sc1, m1);
    scale_min_k4(2 * jj + 1, b + 4, sc2, m2);
    const float d1 = fm(d, static_cast<float>(sc1)),
                mm1 = fm(mn, static_cast<float>(m1));
    const float d2 = fm(d, static_cast<float>(sc2)),
                mm2 = fm(mn, static_cast<float>(m2));
    const uint32_t qh = ld32(b + 16 + 4 * l4);
    const uint32_t q = ld32(b + 48 + 32 * jj + 4 * l4);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t h1 = ((qh >> (8 * k + 2 * jj)) & 1) ? 16 : 0;
      const uint32_t h2 = ((qh >> (8 * k + 2 * jj + 1)) & 1) ? 16 : 0;
      v[k] = fs(fm(d1, static_cast<float>(SQ_NIB(q, k) + h1)), mm1);
      v[4 + k] = fs(fm(d2, static_cast<float>(SQ_NIH(q, k) + h2)), mm2);
    }
  }
};

/* Q6_K: slot = (128-half nn, 4-element group l4 of 32); the 4 output groups c
 * are elements 128nn + 32c + 4l4 + k, built from ql[l], ql[l+32] and one qh
 * byte. */
struct Q6_K : QBase<SPITE_TYPE_Q6_K, QK_K, 210, 4> {
  static_assert(sizeof(block_q6_K) == BYTES, "q6_K");
  SQ_F int start(int s, int g) { return 128 * (s >> 3) + 32 * g + 4 * (s & 7); }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int nn = s >> 3, l4 = s & 7;
    const float d = h2f(ld16(b + 208));
    const uint32_t ql0 = ld32(b + 64 * nn + 4 * l4),
                   ql1 = ld32(b + 64 * nn + 32 + 4 * l4);
    const uint32_t qh = ld32(b + 128 + 32 * nn + 4 * l4);
    const int8_t *sc =
        reinterpret_cast<const int8_t *>(b + 192) + 8 * nn + (l4 >> 2);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float ds = fm(d, static_cast<float>(sc[2 * c]));
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t ql = (c & 1) ? ql1 : ql0;
        const uint32_t lo =
            (c & 2) ? ((ql >> (8 * k + 4)) & 0xF) : ((ql >> (8 * k)) & 0xF);
        const int q =
            static_cast<int>(lo | (((qh >> (8 * k + 2 * c)) & 3) << 4)) - 32;
        v[4 * c + k] = fm(ds, static_cast<float>(q));
      }
    }
  }
};

// ── IQ ────────────────────────────────────────────────────────────────────

struct IQ4_NL : QBase<SPITE_TYPE_IQ4_NL, QK4_NL, 18, 2> {
  static_assert(sizeof(block_iq4_nl) == BYTES, "iq4_nl");
  SQ_F int start(int s, int g) { return 4 * s + 16 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const float d = h2f(ld16(b));
    const uint32_t q = ld32(b + 2 + 4 * s);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[k] = fm(d, static_cast<float>(kvalues_iq4nl[SQ_NIB(q, k)]));
      v[4 + k] = fm(d, static_cast<float>(kvalues_iq4nl[SQ_NIH(q, k)]));
    }
  }
};

struct IQ4_XS : QBase<SPITE_TYPE_IQ4_XS, QK_K, 136, 2> {
  static_assert(sizeof(block_iq4_xs) == BYTES, "iq4_xs");
  SQ_F int start(int s, int g) { return 32 * (s >> 2) + 16 * g + 4 * (s & 3); }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int ib = s >> 2, q4 = s & 3;
    const float d = h2f(ld16(b));
    const uint32_t ls = ((ld8(b + 4 + (ib >> 1)) >> (4 * (ib & 1))) & 0xF) |
                        (((ld16(b + 2) >> (2 * ib)) & 3) << 4);
    const float dl = fm(d, static_cast<float>(static_cast<int>(ls) - 32));
    const uint32_t q = ld32(b + 8 + 16 * ib + 4 * q4);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[k] = fm(dl, static_cast<float>(kvalues_iq4nl[SQ_NIB(q, k)]));
      v[4 + k] = fm(dl, static_cast<float>(kvalues_iq4nl[SQ_NIH(q, k)]));
    }
  }
};

/* Signed grid byte j of an 8-byte grid word. */
SQ_F float grid_val(float db, uint64_t grid, uint32_t signs, int j) {
  return sgn(fm(db, static_cast<float>(static_cast<uint32_t>((grid >> (8 * j)) & 0xFF))),
             signs & (1u << j));
}

/* IQ2_XXS/XS/S and IQ3_XXS/S: slot = (32-element group ib32 = s>>2, 8-element l
 * = s&3). */
struct IQ2_XXS : QBase<SPITE_TYPE_IQ2_XXS, QK_K, 66, 2> {
  static_assert(sizeof(block_iq2_xxs) == BYTES, "iq2_xxs");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int ib32 = s >> 2, l = s & 3;
    const float d = h2f(ld16(b));
    const uint32_t a0 = ld32(b + 2 + 8 * ib32), a1 = ld32(b + 6 + 8 * ib32);
    const float db = fm(fm(d, fa(0.5f, static_cast<float>(a1 >> 28))), 0.25f);
    const uint64_t grid = iq2xxs_grid[byte_of(a0, l)];
    const uint32_t signs = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      v[j] = grid_val(db, grid, signs, j);
  }
};

struct IQ2_XS : QBase<SPITE_TYPE_IQ2_XS, QK_K, 74, 2> {
  static_assert(sizeof(block_iq2_xs) == BYTES, "iq2_xs");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int ib32 = s >> 2, l = s & 3;
    const float d = h2f(ld16(b));
    const uint32_t sc = ld8(b + 66 + ib32);
    const float db = fm(
        fm(d, fa(0.5f, static_cast<float>((l >> 1) ? (sc >> 4) : (sc & 0xF)))),
        0.25f);
    const uint32_t qv = ld16(b + 2 + 2 * s);
    const uint64_t grid = iq2xs_grid[qv & 511];
    const uint32_t signs = ksigns_iq2xs[qv >> 9];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      v[j] = grid_val(db, grid, signs, j);
  }
};

struct IQ2_S : QBase<SPITE_TYPE_IQ2_S, QK_K, 82, 2> {
  static_assert(sizeof(block_iq2_s) == BYTES, "iq2_s");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int ib32 = s >> 2, l = s & 3;
    const float d = h2f(ld16(b));
    const uint32_t sc = ld8(b + 74 + ib32);
    const float dl = fm(
        fm(d, fa(0.5f, static_cast<float>((l >> 1) ? (sc >> 4) : (sc & 0xF)))),
        0.25f);
    const uint32_t idx =
        ld8(b + 2 + s) | ((ld8(b + 66 + ib32) << (8 - 2 * l)) & 0x300);
    const uint64_t grid = iq2s_grid[idx];
    const uint32_t signs = ld8(b + 34 + s);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      v[j] = grid_val(dl, grid, signs, j);
  }
};

struct IQ3_XXS : QBase<SPITE_TYPE_IQ3_XXS, QK_K, 98, 2> {
  static_assert(sizeof(block_iq3_xxs) == BYTES, "iq3_xxs");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int ib32 = s >> 2, l = s & 3;
    const float d = h2f(ld16(b));
    const uint32_t aux = ld32(b + 66 + 4 * ib32);
    const float db = fm(fm(d, fa(0.5f, static_cast<float>(aux >> 28))), 0.5f);
    const uint32_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
    const uint32_t ii = ld16(b + 2 + 8 * ib32 + 2 * l);
    const uint32_t g1 = iq3xxs_grid[ii & 0xFF], g2 = iq3xxs_grid[ii >> 8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = sgn(fm(db, static_cast<float>(byte_of(g1, j))), signs & (1u << j));
      v[4 + j] = sgn(fm(db, static_cast<float>(byte_of(g2, j))),
                     signs & (1u << (j + 4)));
    }
  }
};

struct IQ3_S : QBase<SPITE_TYPE_IQ3_S, QK_K, 110, 2> {
  static_assert(sizeof(block_iq3_s) == BYTES, "iq3_s");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int ib32 = s >> 2, l = s & 3;
    const float d = h2f(ld16(b));
    const uint32_t sc = ld8(b + 106 + (ib32 >> 1));
    const float db = fm(
        d, static_cast<float>(
               1 + 2 * static_cast<int>((ib32 & 1) ? (sc >> 4) : (sc & 0xF))));
    const uint32_t qh = ld8(b + 66 + ib32);
    const uint32_t ii = ld16(b + 2 + 8 * ib32 + 2 * l);
    const uint32_t g1 = iq3s_grid[(ii & 0xFF) | ((qh << (8 - 2 * l)) & 256)];
    const uint32_t g2 = iq3s_grid[(ii >> 8) | ((qh << (7 - 2 * l)) & 256)];
    const uint32_t signs = ld8(b + 74 + s);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = sgn(fm(db, static_cast<float>(byte_of(g1, j))), signs & (1u << j));
      v[4 + j] = sgn(fm(db, static_cast<float>(byte_of(g2, j))),
                     signs & (1u << (j + 4)));
    }
  }
};

/* IQ1: the GPU grid packs 8 trits+1 as nibbles: low nibbles = elements 0..3,
 * high = 4..7. */
SQ_F void iq1_expand(uint32_t g, float dl, float delta, float (&v)[8]) {
  const uint32_t lo = g & 0x0f0f0f0fu, hi = (g >> 4) & 0x0f0f0f0fu;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = fm(dl, fa(static_cast<float>(static_cast<int>(byte_of(lo, j)) - 1),
                     delta));
    v[4 + j] =
        fm(dl,
           fa(static_cast<float>(static_cast<int>(byte_of(hi, j)) - 1), delta));
  }
}

struct IQ1_S : QBase<SPITE_TYPE_IQ1_S, QK_K, 50, 2> {
  static_assert(sizeof(block_iq1_s) == BYTES, "iq1_s");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int ib = s >> 2, il = s & 3;
    const float d = h2f(ld16(b));
    const uint32_t qh = ld16(b + 34 + 2 * ib);
    const float dl =
        fm(d, static_cast<float>(2 * static_cast<int>((qh >> 12) & 7) + 1));
    const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint32_t idx = ld8(b + 2 + s) | (((qh >> (3 * il)) & 7) << 8);
    iq1_expand(iq1s_grid_gpu[idx], dl, delta, v);
  }
};

struct IQ1_M : QBase<SPITE_TYPE_IQ1_M, QK_K, 56, 2> {
  static_assert(sizeof(block_iq1_m) == BYTES, "iq1_m");
  SQ_F int start(int s, int g) { return 8 * s + 4 * g; }
  SQ_F void decode(const uint8_t *b, int s, float (&v)[NV]) {
    const int ib = s >> 2, il = s & 3;
    const uint32_t s0 = ld16(b + 48), s1 = ld16(b + 50), s2 = ld16(b + 52),
                   s3 = ld16(b + 54);
    const float d = h2f((s0 >> 12) | ((s1 >> 8) & 0x00f0) |
                        ((s2 >> 4) & 0x0f00) | (s3 & 0xf000));
    const uint32_t sc = ld16(b + 48 + 2 * (ib >> 1));
    const float dl = fm(
        d,
        static_cast<float>(
            2 * static_cast<int>((sc >> (6 * (ib & 1) + 3 * (il >> 1))) & 7) +
            1));
    const uint32_t qh = ld8(b + 32 + 2 * ib + (il >> 1));
    const float delta =
        (qh & (0x08u << (4 * (il & 1)))) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint32_t idx = ld8(b + s) | ((qh << ((il & 1) ? 4 : 8)) & 0x700);
    iq1_expand(iq1s_grid_gpu[idx], dl, delta, v);
  }
};

// ── full-precision rows (QK = 1): decode(row, i) → element i ──────────────

template <SpiteType ID_, int BYTES_> struct DenseBase {
  static constexpr SpiteType id = ID_;
  static constexpr int QK = 1, BYTES = BYTES_, ALIGN = BYTES_;
  static constexpr bool DENSE = true;
};
struct DenseF32 : DenseBase<SPITE_TYPE_F32, 4> {
  SQ_F float load(const uint8_t *row, int64_t i) {
    return reinterpret_cast<const float *>(row)[i];
  }
};
struct DenseF16 : DenseBase<SPITE_TYPE_F16, 2> {
  SQ_F float load(const uint8_t *row, int64_t i) {
    return h2f(reinterpret_cast<const uint16_t *>(row)[i]);
  }
};
struct DenseBF16 : DenseBase<SPITE_TYPE_BF16, 2> {
  SQ_F float load(const uint8_t *row, int64_t i) {
    return __uint_as_float(
        static_cast<uint32_t>(reinterpret_cast<const uint16_t *>(row)[i])
        << 16);
  }
};

// ── host-side type dispatch ───────────────────────────────────────────────

template <class Q> struct Tag {
  using type = Q;
};

/*
 * Calls f(Tag<Q>{}) for the descriptor of `t` and returns its int result;
 * -1 if `t` is not a SpiteType this library decodes.  f is a generic lambda
 * (host code) — typically launching a kernel templated on Q.
 */
template <class F> inline int visit_type(SpiteType t, F &&f) {
  switch (t) {
  case SPITE_TYPE_F32:
    return f(Tag<DenseF32>{});
  case SPITE_TYPE_F16:
    return f(Tag<DenseF16>{});
  case SPITE_TYPE_BF16:
    return f(Tag<DenseBF16>{});
  case SPITE_TYPE_Q4_0:
    return f(Tag<Q4_0>{});
  case SPITE_TYPE_Q4_1:
    return f(Tag<Q4_1>{});
  case SPITE_TYPE_Q5_0:
    return f(Tag<Q5_0>{});
  case SPITE_TYPE_Q5_1:
    return f(Tag<Q5_1>{});
  case SPITE_TYPE_Q8_0:
    return f(Tag<Q8_0>{});
  case SPITE_TYPE_Q1_0:
    return f(Tag<Q1_0>{});
  case SPITE_TYPE_Q2_0:
    return f(Tag<Q2_0>{});
  case SPITE_TYPE_MXFP4:
    return f(Tag<MXFP4>{});
  case SPITE_TYPE_NVFP4:
    return f(Tag<NVFP4>{});
  case SPITE_TYPE_TQ1_0:
    return f(Tag<TQ1_0>{});
  case SPITE_TYPE_TQ2_0:
    return f(Tag<TQ2_0>{});
  case SPITE_TYPE_Q2_K:
    return f(Tag<Q2_K>{});
  case SPITE_TYPE_Q3_K:
    return f(Tag<Q3_K>{});
  case SPITE_TYPE_Q4_K:
    return f(Tag<Q4_K>{});
  case SPITE_TYPE_Q5_K:
    return f(Tag<Q5_K>{});
  case SPITE_TYPE_Q6_K:
    return f(Tag<Q6_K>{});
  case SPITE_TYPE_IQ4_NL:
    return f(Tag<IQ4_NL>{});
  case SPITE_TYPE_IQ4_XS:
    return f(Tag<IQ4_XS>{});
  case SPITE_TYPE_IQ2_XXS:
    return f(Tag<IQ2_XXS>{});
  case SPITE_TYPE_IQ2_XS:
    return f(Tag<IQ2_XS>{});
  case SPITE_TYPE_IQ2_S:
    return f(Tag<IQ2_S>{});
  case SPITE_TYPE_IQ3_XXS:
    return f(Tag<IQ3_XXS>{});
  case SPITE_TYPE_IQ3_S:
    return f(Tag<IQ3_S>{});
  case SPITE_TYPE_IQ1_S:
    return f(Tag<IQ1_S>{});
  case SPITE_TYPE_IQ1_M:
    return f(Tag<IQ1_M>{});
  default:
    return -1;
  }
}

#undef SQ_NIB
#undef SQ_NIH

} // namespace sq
