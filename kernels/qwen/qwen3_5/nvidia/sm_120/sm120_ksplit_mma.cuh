/*
 * kernels/qwen/qwen3_5/nvidia/sm_120/sm120_ksplit_mma.cuh
 *
 * Blackwell (sm_120) Tensor Core K-Split MMA contractions for quantized GEMV.
 * Evaluates Q8_0, Q4_K, Q5_K, and NVFP4 across 8 K-split warps per CTA.
 */

#pragma once

#include "sm120_mma.cuh"
#include "common.h"
#include "core/gpu/quant_dequant.h"

namespace sm120 {

constexpr int K_WARPS = 8;
constexpr int K_THREADS = K_WARPS * 32;

/* Warp reduction across 32 lanes */
__device__ __forceinline__ float warp_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

/* ── Q8_0 K-Split Contraction ─────────────────────────────────────────── */
__device__ __forceinline__ float ksplit_q8_0(const uint8_t *__restrict__ wr,
                                             const float *__restrict__ x,
                                             int nb, int wid, int lane) {
  float warp_total = 0.0f;
  const block_q8_0 *blocks = reinterpret_cast<const block_q8_0 *>(wr);

  for (int b = wid; b < nb; b += K_WARPS) {
    const block_q8_0 &blk = blocks[b];
    const float d = __half2float(*reinterpret_cast<const __half *>(&blk.d));
    const float xw = __ldg(x + static_cast<size_t>(b) * 32 + lane);
    const int8_t qw = blk.qs[lane];
    float dot = warp_sum(static_cast<float>(qw) * xw);
    if (lane == 0) {
      warp_total = fmaf(dot, d, warp_total);
    }
  }
  return warp_total;
}

/* ── Q4_K K-Split Contraction ─────────────────────────────────────────── */
__device__ __forceinline__ float ksplit_q4_k(const uint8_t *__restrict__ wr,
                                             const float *__restrict__ x,
                                             int nb, int wid, int lane) {
  float warp_total = 0.0f;
  const size_t b_bytes = sizeof(block_q4_K);

  /* Each block has 256 weights. NV=8 per slot, NS=32 slots.
   * Slot = lane, so each warp collaboratively evaluates full 256-weight blocks.
   */
  for (int b = wid; b < nb; b += K_WARPS) {
    const uint8_t *blk = wr + static_cast<size_t>(b) * b_bytes;
    float v[8];
    sq::Q4_K::decode(blk, lane, v);
    const float *xb = x + static_cast<size_t>(b) * 256;

    float acc = 0.0f;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const float *xp = xb + sq::Q4_K::start(lane, g);
      float4 xv = __ldg(reinterpret_cast<const float4 *>(xp));
      acc = fmaf(v[4 * g + 0], xv.x, acc);
      acc = fmaf(v[4 * g + 1], xv.y, acc);
      acc = fmaf(v[4 * g + 2], xv.z, acc);
      acc = fmaf(v[4 * g + 3], xv.w, acc);
    }
    float b_sum = warp_sum(acc);
    if (lane == 0) {
      warp_total += b_sum;
    }
  }
  return warp_total;
}

/* ── Q5_K K-Split Contraction ─────────────────────────────────────────── */
__device__ __forceinline__ float ksplit_q5_k(const uint8_t *__restrict__ wr,
                                             const float *__restrict__ x,
                                             int nb, int wid, int lane) {
  float warp_total = 0.0f;
  const size_t b_bytes = sizeof(block_q5_K);

  for (int b = wid; b < nb; b += K_WARPS) {
    const uint8_t *blk = wr + static_cast<size_t>(b) * b_bytes;
    float v[8];
    sq::Q5_K::decode(blk, lane, v);
    const float *xb = x + static_cast<size_t>(b) * 256;

    float acc = 0.0f;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const float *xp = xb + sq::Q5_K::start(lane, g);
      float4 xv = __ldg(reinterpret_cast<const float4 *>(xp));
      acc = fmaf(v[4 * g + 0], xv.x, acc);
      acc = fmaf(v[4 * g + 1], xv.y, acc);
      acc = fmaf(v[4 * g + 2], xv.z, acc);
      acc = fmaf(v[4 * g + 3], xv.w, acc);
    }
    float b_sum = warp_sum(acc);
    if (lane == 0) {
      warp_total += b_sum;
    }
  }
  return warp_total;
}

/* ── NVFP4 K-Split Contraction ────────────────────────────────────────── */
__device__ __forceinline__ float ksplit_nvfp4(const uint8_t *__restrict__ wr,
                                              const float *__restrict__ x,
                                              int nb, int wid, int lane) {
  float sub_total = 0.0f;
  const size_t b_bytes = sizeof(block_nvfp4);
  const int slot = lane % 8;
  const int sub = lane / 8; // 4 sub-groups of 8 lanes per warp

  for (int b = wid * 4 + sub; b < nb; b += K_WARPS * 4) {
    const uint8_t *blk = wr + static_cast<size_t>(b) * b_bytes;
    float v[8];
    sq::NVFP4::decode(blk, slot, v);
    const float *xb = x + static_cast<size_t>(b) * 64;

    float acc = 0.0f;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const float *xp = xb + sq::NVFP4::start(slot, g);
      float4 xv = __ldg(reinterpret_cast<const float4 *>(xp));
      acc = fmaf(v[4 * g + 0], xv.x, acc);
      acc = fmaf(v[4 * g + 1], xv.y, acc);
      acc = fmaf(v[4 * g + 2], xv.z, acc);
      acc = fmaf(v[4 * g + 3], xv.w, acc);
    }
    // Reduce the 8 slots of block b
    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    if (slot == 0) {
      sub_total += acc;
    }
  }

  // Reduce the 4 sub-groups to lane 0
  sub_total += __shfl_xor_sync(0xffffffffu, sub_total, 16);
  sub_total += __shfl_xor_sync(0xffffffffu, sub_total, 8);
  return sub_total;
}

} // namespace sm120
