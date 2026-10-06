/*
 * kernels/qwen/qwen3_5/nvidia/sm_120/gemv.cu
 *
 * Blackwell (sm_120) Tensor Core accelerated GEMV with K-Split MMA contractions
 * and TMA asynchronous pipeline support.
 */

#include "common.h"
#include "core/gpu/quant_dequant.h"
#include "sm120_mma.cuh"
#include "sm120_tma.cuh"
#include "sm120_ksplit_mma.cuh"

namespace {

constexpr int kRowThreads = 256;
constexpr int kWarps = kRowThreads / 32;

struct MultiArgs {
  const uint8_t *w[kQ35MaxGemvJobs];
  float *y[kQ35MaxGemvJobs];
  int row_end[kQ35MaxGemvJobs]; // exclusive prefix sum of the jobs' row counts
  int n;
  int accumulate;
};

__device__ __forceinline__ void get_job_info(const MultiArgs &a, int row,
                                             const uint8_t *&wsel, float *&ysel,
                                             int &local, size_t row_bytes) {
  wsel = a.w[0];
  ysel = a.y[0];
  int base = 0;
#pragma unroll
  for (int i = 1; i < kQ35MaxGemvJobs; ++i) {
    if (i < a.n && row >= a.row_end[i - 1]) {
      wsel = a.w[i];
      ysel = a.y[i];
      base = a.row_end[i - 1];
    }
  }
  local = row - base;
}

/* Blackwell Tensor Core K-Split MMA contraction kernel (8 warps per CTA) */
template <SpiteType ID>
__global__ void __launch_bounds__(kRowThreads)
    gemv_ksplit_mma_kernel(MultiArgs a, const float *__restrict__ x, int cols,
                           size_t row_bytes) {
  __shared__ float smem_partials[kWarps];
  const int row = blockIdx.x;
  const uint8_t *wsel;
  float *ysel;
  int local;
  get_job_info(a, row, wsel, ysel, local, row_bytes);

  const uint8_t *wr = wsel + static_cast<size_t>(local) * row_bytes;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;

  float warp_part = 0.0f;
  if constexpr (ID == SPITE_TYPE_Q8_0) {
    const int nb = cols / 32;
    warp_part = sm120::ksplit_q8_0(wr, x, nb, wid, lane);
  } else if constexpr (ID == SPITE_TYPE_Q4_K) {
    const int nb = cols / 256;
    warp_part = sm120::ksplit_q4_k(wr, x, nb, wid, lane);
  } else if constexpr (ID == SPITE_TYPE_Q5_K) {
    const int nb = cols / 256;
    warp_part = sm120::ksplit_q5_k(wr, x, nb, wid, lane);
  } else if constexpr (ID == SPITE_TYPE_NVFP4) {
    const int nb = cols / 64;
    warp_part = sm120::ksplit_nvfp4(wr, x, nb, wid, lane);
  }

  if (lane == 0) {
    smem_partials[wid] = warp_part;
  }
  __syncthreads();

  if (wid == 0) {
    float sum = (lane < kWarps) ? smem_partials[lane] : 0.0f;
    sum = sm120::warp_sum(sum);
    if (lane == 0) {
      float *y = ysel + local;
      *y = a.accumulate ? *y + sum : sum;
    }
  }
}

/* General SIMT row kernel with Blackwell dual-issue unrolled reductions */
template <class Q>
__global__ void __launch_bounds__(kRowThreads)
    gemv_row_kernel(MultiArgs a, const float *__restrict__ x, int cols,
                    size_t row_bytes, int x_aligned16) {
  __shared__ float red[kWarps];
  const int row = blockIdx.x;
  const uint8_t *wsel;
  float *ysel;
  int local;
  get_job_info(a, row, wsel, ysel, local, row_bytes);

  const uint8_t *wr = wsel + static_cast<size_t>(local) * row_bytes;
  const int tid = threadIdx.x;

  float acc = 0.0f;
  if constexpr (Q::DENSE) {
    for (int c = tid; c < cols; c += kRowThreads)
      acc = fmaf(Q::load(wr, c), __ldg(x + c), acc);
  } else {
    constexpr int NS = Q::NS,
                  BPS = kRowThreads / NS; // blocks consumed per step
    const int slot = tid % NS, sub = tid / NS;
    const int nb = cols / Q::QK;
    for (int b = sub; b < nb; b += BPS) {
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
          xv = make_float4(__ldg(xp), __ldg(xp + 1), __ldg(xp + 2),
                           __ldg(xp + 3));
        acc = fmaf(v[4 * g + 0], xv.x, acc);
        acc = fmaf(v[4 * g + 1], xv.y, acc);
        acc = fmaf(v[4 * g + 2], xv.z, acc);
        acc = fmaf(v[4 * g + 3], xv.w, acc);
      }
    }
  }
  acc = sm120::warp_sum(acc);
  if ((tid & 31) == 0)
    red[tid >> 5] = acc;
  __syncthreads();
  if (tid == 0) {
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < kWarps; ++i)
      t += red[i];
    float *y = ysel + local;
    *y = a.accumulate ? *y + t : t;
  }
}

bool job_ok(const SpiteTensor *w, const float *x, const float *y) {
  return w && w->data && x && y && w->ne[0] > 0 && w->ne[1] > 0 &&
         spite_tensor_is_contiguous(w);
}

/* One launch over jobs that share kind and cols. */
int launch(const Q35GemvJob *jobs, int n, const float *x, bool accumulate,
           cudaStream_t s) {
  const SpiteTensor *w0 = jobs[0].w;
  const int cols = static_cast<int>(w0->ne[0]);
  MultiArgs a = {};
  int64_t rows = 0;
  for (int i = 0; i < n; ++i) {
    a.w[i] = static_cast<const uint8_t *>(jobs[i].w->data);
    a.y[i] = jobs[i].y;
    rows += jobs[i].w->ne[1];
    a.row_end[i] = static_cast<int>(rows);
  }
  if (rows > 0x7fffffff)
    return -1;
  a.n = n;
  a.accumulate = accumulate;
  const unsigned grid = static_cast<unsigned>(rows);
  const int x_aligned16 = (reinterpret_cast<uintptr_t>(x) & 15) == 0;

  /* Blackwell K-Split MMA tensor core fast paths */
  if (w0->kind == SPITE_TYPE_Q8_0 && (cols % 32 == 0) && x_aligned16) {
    const size_t row_bytes = static_cast<size_t>(cols / 32) * sizeof(block_q8_0);
    gemv_ksplit_mma_kernel<SPITE_TYPE_Q8_0>
        <<<grid, kRowThreads, 0, s>>>(a, x, cols, row_bytes);
    return 0;
  }
  if (w0->kind == SPITE_TYPE_Q4_K && (cols % 256 == 0) && x_aligned16) {
    const size_t row_bytes = static_cast<size_t>(cols / 256) * sizeof(block_q4_K);
    gemv_ksplit_mma_kernel<SPITE_TYPE_Q4_K>
        <<<grid, kRowThreads, 0, s>>>(a, x, cols, row_bytes);
    return 0;
  }
  if (w0->kind == SPITE_TYPE_Q5_K && (cols % 256 == 0) && x_aligned16) {
    const size_t row_bytes = static_cast<size_t>(cols / 256) * sizeof(block_q5_K);
    gemv_ksplit_mma_kernel<SPITE_TYPE_Q5_K>
        <<<grid, kRowThreads, 0, s>>>(a, x, cols, row_bytes);
    return 0;
  }
  if (w0->kind == SPITE_TYPE_NVFP4 && (cols % 64 == 0) && x_aligned16) {
    const size_t row_bytes = static_cast<size_t>(cols / 64) * sizeof(block_nvfp4);
    gemv_ksplit_mma_kernel<SPITE_TYPE_NVFP4>
        <<<grid, kRowThreads, 0, s>>>(a, x, cols, row_bytes);
    return 0;
  }

  return sq::visit_type(w0->kind, [&](auto tag) -> int {
    using Q = typename decltype(tag)::type;
    for (int i = 0; i < n; ++i)
      if (reinterpret_cast<uintptr_t>(a.w[i]) % Q::ALIGN)
        return -1;
    size_t row_bytes;
    if constexpr (Q::DENSE) {
      row_bytes = static_cast<size_t>(cols) * Q::BYTES;
    } else {
      if (cols % Q::QK)
        return -1;
      row_bytes = static_cast<size_t>(cols / Q::QK) * Q::BYTES;
    }
    gemv_row_kernel<Q>
        <<<grid, kRowThreads, 0, s>>>(a, x, cols, row_bytes, x_aligned16);
    return 0;
  });
}

} // namespace

int q35_gemv_multi(const Q35GemvJob *jobs, int n, const float *x,
                   bool accumulate, cudaStream_t s) {
  if (!jobs || n < 1 || n > kQ35MaxGemvJobs)
    return -1;
  bool fuse = true;
  for (int i = 0; i < n; ++i) {
    if (!job_ok(jobs[i].w, x, jobs[i].y))
      return -1;
    if (jobs[i].w->kind != jobs[0].w->kind ||
        jobs[i].w->ne[0] != jobs[0].w->ne[0])
      fuse = false;
  }
  if (fuse)
    return launch(jobs, n, x, accumulate, s);
  for (int i = 0; i < n; ++i)
    if (launch(jobs + i, 1, x, accumulate, s))
      return -1;
  return 0;
}

int q35_gemv(const SpiteTensor *w, const float *x, float *y, bool accumulate,
             cudaStream_t s) {
  const Q35GemvJob job = {w, y};
  return q35_gemv_multi(&job, 1, x, accumulate, s);
}
