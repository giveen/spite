/*
 * kernels/qwen/qwen3_5/nvidia/common.h — helpers shared by the translation
 * units of the Qwen3.5 NVIDIA kernel (kernel.cu, attn.cu, gdn.cu, gemv.cu).
 *
 * The dequantizing GEMV (core/gpu/quant_gemv.h instantiates one CUDA kernel per
 * SpiteType) is compiled exactly once, in gemv.cu, and reached through
 * q35_gemv(); the other translation units stay cheap to build.
 */
#pragma once

#include "core/abi.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

inline cudaStream_t q35_stream(const SpiteCtx *ctx) {
  return ctx ? static_cast<cudaStream_t>(ctx->gpu_stream) : nullptr;
}

/* y[r] (+)= W[r,:] . x for a contiguous [cols=ne[0], rows=ne[1]] weight of any
 * SpiteType. 0 on success, -1 for an undecodable / misaligned / non-contiguous
 * weight. Only enqueues work; callers check cudaGetLastError(). (gemv.cu) */
int q35_gemv(const SpiteTensor *w, const float *x, float *y, bool accumulate,
             cudaStream_t s);

/* Several projections of the same x: y_i (+)= W_i . x. Jobs with one weight type
 * and column count run as a single launch. */
constexpr int kQ35MaxGemvJobs = 4;
struct Q35GemvJob {
  const SpiteTensor *w;
  float *y;
};
int q35_gemv_multi(const Q35GemvJob *jobs, int n, const float *x, bool accumulate,
                   cudaStream_t s);

inline int64_t q35_numel(const SpiteTensor *t) {
  int64_t n = 1;
  for (int i = 0; i < 4; ++i)
    if (t->ne[i])
      n *= t->ne[i];
  return n;
}

/* Non-null contiguous F32 tensor with exactly n elements, 4-byte aligned. */
inline bool q35_f32_n(const SpiteTensor *t, int64_t n) {
  return t && t->data && t->kind == SPITE_TYPE_F32 && q35_numel(t) == n &&
         (reinterpret_cast<uintptr_t>(t->data) & 3) == 0;
}

extern "C" int qwen35_cuda_moe_ffn(
    SpiteTensor*          out,
    const SpiteTensor*    x,
    const SpiteTensor*    w_gate_inp,
    const SpiteTensor*    w_up_exps,
    const SpiteTensor*    w_gate_exps,
    const SpiteTensor*    w_down_exps,
    const SpiteTensor*    w_up_shexp,
    const SpiteTensor*    w_gate_shexp,
    const SpiteTensor*    w_down_shexp,
    const SpiteMoeParams* params,
    const SpiteCtx*       ctx);

#ifdef __CUDACC__
__device__ __forceinline__ float q35_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float q35_silu(float x) {
  return x / (1.0f + __expf(-x));
}

/* Block-wide sum; blockDim.x must be a multiple of 32 and <= 1024. Every thread
 * gets the total. Safe to call repeatedly. */
__device__ __forceinline__ float q35_block_sum(float v) {
  __shared__ float red[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = q35_warp_sum(v);
  __syncthreads(); /* red[] may still be read by a previous call */
  if (lane == 0)
    red[wid] = v;
  __syncthreads();
  float t = 0.0f;
  for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i)
    t += red[i];
  return t;
}

/* Element i of a small F32/F16/BF16 weight vector (kind = SpiteType). */
__device__ __forceinline__ float q35_load_w(const void *w, int kind, int i) {
  if (kind == SPITE_TYPE_F16)
    return __half2float(static_cast<const __half *>(w)[i]);
  if (kind == SPITE_TYPE_BF16)
    return __bfloat162float(static_cast<const __nv_bfloat16 *>(w)[i]);
  return static_cast<const float *>(w)[i];
}
#endif
