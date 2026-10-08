/*
 * kernels/qwen/qwen3_5/nvidia/gemv.cu — the one translation unit that
 * instantiates the dequantizing GEMV for every SpiteType. Every projection in
 * this kernel dir goes through q35_gemv() / q35_gemv_multi().
 *
 *   y[r] (+)= sum_c W[r, c] * x[c]     F32 activations, F32 accumulation
 *
 * Same decoders as core/gpu/quant_gemv.h (core/gpu/quant_dequant.h, bit-exact
 * vs ggml), different work split: ONE BLOCK OF 256 THREADS PER OUTPUT ROW, the
 * K axis spread over all its threads (slot = thread % Q::NS decodes Q::NV
 * elements of block `thread / Q::NS`, the block walks the row), combined by a
 * warp-shuffle + smem reduction. Why not the one-warp-per-row sq::gemv: a warp
 * that walks a whole row serially leaves small projections latency-bound (a
 * 48-row beta/alpha launch is a dozen CTAs) and wave-quantised on big ones.
 * ncu, cache flushed before the launch, RTX 5090, Q5_K, 5120 columns (each
 * number includes ~3 us of fixed launch/tail cost):
 *
 *      rows    sq::gemv    this
 *     10240     46.7 us   37.6 us   (36 MB; a streaming read of it takes ~25 us)
 *     14336     46.9 us   49.6 us   (parity)
 *      1024     20.7 us    7.8 us
 *        48     21.0 us    5.5 us
 *
 * Both stay well below DRAM peak (~50% on the big shapes): the bit-exact decoders
 * cost about one warp instruction per weight byte (schedulers ~60% busy, ncu), so
 * going further needs type-specialised decoders that factor the block scales out of
 * the dot product (not bit-exact vs dequantize-then-dot, within float rounding).
 *
 * q35_gemv_multi() additionally fuses up to kMaxJobs projections that share x,
 * weight type and column count (e.g. wq|wk|wv, or the four GDN input
 * projections) into ONE launch: CTA r belongs to the job whose row range
 * contains r. Different types fall back to one launch each.
 */
#include "common.h"
#include "core/gpu/quant_dequant.h"

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

template <class Q>
__global__ void __launch_bounds__(kRowThreads)
    gemv_row_kernel(MultiArgs a, const float *__restrict__ x, int cols,
                    size_t row_bytes, int x_aligned16) {
  __shared__ float red[kWarps];
  const int row = blockIdx.x;
  // job lookup with compile-time indices only: a dynamically indexed kernel-parameter
  // array would be copied to per-thread local memory (measured 3x slower)
  const uint8_t *wsel = a.w[0];
  float *ysel = a.y[0];
  int base = 0;
#pragma unroll
  for (int i = 1; i < kQ35MaxGemvJobs; ++i) {
    if (i < a.n && row >= a.row_end[i - 1]) {
      wsel = a.w[i];
      ysel = a.y[i];
      base = a.row_end[i - 1];
    }
  }
  const int local = row - base;
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
  acc = q35_warp_sum(acc);
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

/*
 * Batched (multi-column) GEMV: y[r, t] (+)= sum_c W[r, c] * x[c, t] for m
 * columns (tokens). One block per output row, as above, but the weight row is
 * decoded once per chunk of kBatchChunk columns and FMA'd against all of them,
 * so the weight is read m/kBatchChunk times instead of m times. This is the
 * prefill/verify win: the weight stream is the bandwidth wall, extra columns
 * are near-free. x is [cols, m] and y is [rows, m], both token-major.
 */
/// Columns a decoded weight block is reused for. The row is decoded
/// m/kBatchChunk times, so raising this cuts the redundant decode at the cost
/// of kBatchChunk accumulator registers; Q6_K is the register-bound instance
/// (99 regs/thread at 4), so this trades decode against occupancy. The
/// per-column accumulation order is unchanged, so the result stays bit-identical
/// to the m=1 row kernel.
/// `SPITE_GEMV_BATCH_CHUNK` overrides it for A/B measurement.
#ifndef SPITE_GEMV_BATCH_CHUNK
#define SPITE_GEMV_BATCH_CHUNK 8
#endif
constexpr int kBatchChunk = SPITE_GEMV_BATCH_CHUNK;

template <class Q>
__global__ void __launch_bounds__(kRowThreads)
    gemv_batch_kernel(const uint8_t *__restrict__ w, float *__restrict__ y,
                      const float *__restrict__ x, int cols, int rows, int m,
                      size_t row_bytes, int x_aligned16, int accumulate) {
  __shared__ float red[kWarps];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const uint8_t *wr = w + static_cast<size_t>(row) * row_bytes;

  for (int t0 = 0; t0 < m; t0 += kBatchChunk) {
    const int tn = min(kBatchChunk, m - t0);
    float acc[kBatchChunk];
#pragma unroll
    for (int j = 0; j < kBatchChunk; ++j)
      acc[j] = 0.0f;

    if constexpr (Q::DENSE) {
      for (int c = tid; c < cols; c += kRowThreads) {
        const float wv = Q::load(wr, c);
#pragma unroll
        for (int j = 0; j < kBatchChunk; ++j)
          if (j < tn)
            acc[j] = fmaf(wv, __ldg(x + static_cast<size_t>(t0 + j) * cols + c),
                          acc[j]);
      }
    } else {
      constexpr int NS = Q::NS, BPS = kRowThreads / NS;
      const int slot = tid % NS, sub = tid / NS;
      const int nb = cols / Q::QK;
      for (int b = sub; b < nb; b += BPS) {
        float v[Q::NV];
        Q::decode(wr + static_cast<size_t>(b) * Q::BYTES, slot, v);
#pragma unroll
        for (int j = 0; j < kBatchChunk; ++j) {
          if (j >= tn)
            continue;
          const float *xb =
              x + static_cast<size_t>(t0 + j) * cols + static_cast<size_t>(b) * Q::QK;
#pragma unroll
          for (int g = 0; g < Q::NG; ++g) {
            const float *xp = xb + Q::start(slot, g);
            float4 xv;
            if (x_aligned16)
              xv = __ldg(reinterpret_cast<const float4 *>(xp));
            else
              xv = make_float4(__ldg(xp), __ldg(xp + 1), __ldg(xp + 2),
                               __ldg(xp + 3));
            acc[j] = fmaf(v[4 * g + 0], xv.x, acc[j]);
            acc[j] = fmaf(v[4 * g + 1], xv.y, acc[j]);
            acc[j] = fmaf(v[4 * g + 2], xv.z, acc[j]);
            acc[j] = fmaf(v[4 * g + 3], xv.w, acc[j]);
          }
        }
      }
    }

#pragma unroll
    for (int j = 0; j < kBatchChunk; ++j) {
      if (j >= tn)
        continue;
      const float a = q35_warp_sum(acc[j]);
      if ((tid & 31) == 0)
        red[tid >> 5] = a;
      __syncthreads();
      if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < kWarps; ++i)
          t += red[i];
        float *yp = y + static_cast<size_t>(t0 + j) * rows + row;
        *yp = accumulate ? *yp + t : t;
      }
      __syncthreads();
    }
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

/* One batched launch for a single weight, m columns. */
int launch_batch(const SpiteTensor *w, const float *x, float *y, int m,
                 bool accumulate, cudaStream_t s) {
  const int cols = static_cast<int>(w->ne[0]);
  const int rows = static_cast<int>(w->ne[1]);
  const int x_aligned16 = (reinterpret_cast<uintptr_t>(x) & 15) == 0;
  return sq::visit_type(w->kind, [&](auto tag) -> int {
    using Q = typename decltype(tag)::type;
    if (reinterpret_cast<uintptr_t>(w->data) % Q::ALIGN)
      return -1;
    size_t row_bytes;
    if constexpr (Q::DENSE) {
      row_bytes = static_cast<size_t>(cols) * Q::BYTES;
    } else {
      if (cols % Q::QK)
        return -1;
      row_bytes = static_cast<size_t>(cols / Q::QK) * Q::BYTES;
    }
    gemv_batch_kernel<Q><<<static_cast<unsigned>(rows), kRowThreads, 0, s>>>(
        static_cast<const uint8_t *>(w->data), y, x, cols, rows, m, row_bytes,
        x_aligned16, accumulate ? 1 : 0);
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

/* y[r, t] (+)= sum_c W[r, c] * x[c, t] for m columns (tokens); x is [cols, m]
 * and y is [rows, m], both token-major. The weight row is decoded once per
 * kBatchChunk columns. */
int q35_gemv_batch(const SpiteTensor *w, const float *x, float *y, int m,
                   bool accumulate, cudaStream_t s) {
  if (!w || !w->data || !x || !y || m < 1 || w->ne[0] < 1 || w->ne[1] < 1 ||
      !spite_tensor_is_contiguous(w))
    return -1;
  return launch_batch(w, x, y, m, accumulate, s);
}
