/*
 * spite/core/gpu/quant_gemv.h — F32-activation GEMV over any SpiteType weight.
 * Vendor-neutral source for nvcc (CUDA) and hipcc (HIP); see quant_dequant.h.
 *
 *     y[r] (+)= sum_c W[r, c] * x[c]          r in [0, rows), c in [0, cols)
 *
 * W is row-major, contiguous: row stride = spite_type_block_bytes(t) *
 * (cols / spite_type_block_elements(t)).  x and y are F32 device pointers; the
 * weights are dequantized in registers (core/gpu/quant_dequant.h) and
 * accumulated in F32 — activations are NOT quantized (no q8_1 / dp4a), so the
 * result tracks the CPU reference to float rounding.
 *
 * One warp per output row.  Lane = (sub, slot): slot s decodes NV = 4*NG
 * elements of one block, sub = lane / NS picks the block, so a warp consumes
 * 32/NS consecutive blocks per step (NS = slots per block, see Q::NS) with
 * coalesced reads of the row, then a shuffle reduction.  x is read as float4
 * when 16-byte aligned, scalar otherwise.  Memory-bound decode: first correct
 * version, not tuned (no K-split for small row counts, no x staging).
 * Assumes 32-lane warps (NVIDIA, AMD RDNA wave32); CDNA wave64 needs a
 * different lane mapping.
 *
 * Header-only, include at global scope; see quant_dequant.h for ODR notes.
 */
#pragma once

#include "quant_dequant.h"

#include <stddef.h>

#if defined(__HIPCC__)
typedef hipStream_t sq_stream_t;
#define SQ_SHFL_XOR(v, o) __shfl_xor((v), (o), 32)
#else
#include <cuda_runtime.h>
typedef cudaStream_t sq_stream_t;
#define SQ_SHFL_XOR(v, o) __shfl_xor_sync(0xffffffffu, (v), (o))
#endif

namespace sq {

constexpr int GEMV_WARPS =
    4; /* output rows (warps) per CTA; matches the Q8_0 matvec */

SQ_F float warp_reduce_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += SQ_SHFL_XOR(v, o);
  return v;
}

SQ_F void store_row(float *y, int row, float acc, int accumulate) {
  if (threadIdx.x == 0)
    y[row] = accumulate ? y[row] + acc : acc;
}

template <class Q>
__global__ void __launch_bounds__(32 * GEMV_WARPS)
    gemv_block_kernel(const uint8_t *__restrict__ w,
                      const float *__restrict__ x, float *__restrict__ y,
                      int rows, int nb, size_t row_bytes, int accumulate,
                      int x_aligned16) {
  const int row = blockIdx.x * GEMV_WARPS + threadIdx.y;
  if (row >= rows)
    return; /* whole warp leaves together; shuffles below stay convergent */
  constexpr int NS = Q::NS, BPI = 32 / NS; /* blocks per warp step */
  const int lane = threadIdx.x, slot = lane % NS, sub = lane / NS;
  const uint8_t *wr = w + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
#pragma unroll 4
  for (int b = sub; b < nb; b += BPI) {
    float v[Q::NV];
    Q::decode(wr + static_cast<size_t>(b) * Q::BYTES, slot, v);
    const float *xb = x + static_cast<size_t>(b) * Q::QK;
#pragma unroll
    for (int g = 0; g < Q::NG; ++g) {
      const float *xp = xb + Q::start(slot, g);
      float4 xv;
      if (x_aligned16)
        xv = __ldg(reinterpret_cast<const float4 *>(xp));
      else
        xv =
            make_float4(__ldg(xp), __ldg(xp + 1), __ldg(xp + 2), __ldg(xp + 3));
      acc = fmaf(v[4 * g + 0], xv.x, acc);
      acc = fmaf(v[4 * g + 1], xv.y, acc);
      acc = fmaf(v[4 * g + 2], xv.z, acc);
      acc = fmaf(v[4 * g + 3], xv.w, acc);
    }
  }
  store_row(y, row, warp_reduce_sum(acc), accumulate);
}

template <class Q>
__global__ void __launch_bounds__(32 * GEMV_WARPS)
    gemv_dense_kernel(const uint8_t *__restrict__ w,
                      const float *__restrict__ x, float *__restrict__ y,
                      int rows, int cols, size_t row_bytes, int accumulate) {
  const int row = blockIdx.x * GEMV_WARPS + threadIdx.y;
  if (row >= rows)
    return;
  const uint8_t *wr = w + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = threadIdx.x; c < cols; c += 32)
    acc = fmaf(Q::load(wr, c), __ldg(x + c), acc);
  store_row(y, row, warp_reduce_sum(acc), accumulate);
}

/*
 * y[r] (+)= W[r,:] . x for a [rows x cols] weight matrix of type `t`.
 * Returns 0 on success, -1 if `t` is not a supported type, the shape is empty,
 * cols is not a multiple of the type's block size, or w is misaligned for the
 * type (Q::ALIGN).  Only enqueues work on `stream`; callers check
 * cudaGetLastError() / hipGetLastError().
 */
inline int gemv(SpiteType t, const void *w, const float *x, float *y, int rows,
                int cols, bool accumulate, sq_stream_t stream) {
  if (rows <= 0 || cols <= 0 || !w || !x || !y)
    return -1;
  const dim3 block(32, GEMV_WARPS);
  const dim3 grid((rows + GEMV_WARPS - 1) / GEMV_WARPS);
  const uint8_t *wp = static_cast<const uint8_t *>(w);
  return visit_type(t, [&](auto tag) -> int {
    using Q = typename decltype(tag)::type;
    if (reinterpret_cast<uintptr_t>(w) % Q::ALIGN)
      return -1;
    if constexpr (Q::DENSE) {
      gemv_dense_kernel<Q><<<grid, block, 0, stream>>>(
          wp, x, y, rows, cols, static_cast<size_t>(cols) * Q::BYTES,
          accumulate);
    } else {
      if (cols % Q::QK)
        return -1;
      const int nb = cols / Q::QK;
      gemv_block_kernel<Q><<<grid, block, 0, stream>>>(
          wp, x, y, rows, nb, static_cast<size_t>(nb) * Q::BYTES, accumulate,
          (reinterpret_cast<uintptr_t>(x) & 15) == 0);
    }
    return 0;
  });
}

/* Convenience for kernels: gemv over a SpiteTensor [cols=ne[0], rows=ne[1]]
 * weight. */
inline int gemv(const SpiteTensor *w, const float *x, float *y, bool accumulate,
                sq_stream_t stream) {
  return gemv(w->kind, w->data, x, y, static_cast<int>(w->ne[1] ? w->ne[1] : 1),
              static_cast<int>(w->ne[0]), accumulate, stream);
}

} // namespace sq
