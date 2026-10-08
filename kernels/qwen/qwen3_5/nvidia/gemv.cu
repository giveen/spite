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
 * columns (tokens). One block per kRowTile output rows, as above, but the weight
 * row is decoded once per chunk of kRowWideChunk columns and FMA'd against all of
 * them, so the weight is read m/kRowWideChunk times instead of m times. This is
 * the prefill/verify win. x is [cols, m] and y is [rows, m], both token-major.
 *
 * The wall on Pascal is not the weight stream, it is the *activation* stream:
 * every one of the `rows` blocks reads the whole [cols, m] activation, so those
 * bytes are `rows`-fold amplified -- 182.5 GB against 4.68 GB of weights at the
 * 27B FFN-gate shape (rows 17408, cols 5120, Q6_K, m 512). Replacing the
 * activation loads with a constant takes that kernel from 16.7 ms to 4.1 ms
 * (75%, host RTX 5090 through PTX JIT of this sm_60 build), while sweeping
 * kBatchChunk 2/4/8/16 over the same shape gives 18.2/19.8/16.8/19.3 ms -- the
 * weight stream is a register-pressure tradeoff, not the wall. kRowTile is the
 * only lever that moves the activation bytes.
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

/// Output rows one block owns. Every activation float4 a thread loads feeds
/// exactly one FMA per owned row, so a block owning `kRowTile` rows reuses each
/// activation load `kRowTile` times and divides the activation traffic by the
/// same factor. That traffic is what binds this kernel (each of the `rows`
/// blocks streams the whole [cols, m] activation: 39x the weight bytes at the
/// FFN-gate shape), and it is the *only* thing that moves those bytes --
/// staging them in shared memory does not, because it changes where the load
/// comes from rather than how many times the value is used, leaving the FMA
/// loop starved either way (the 16x-slower prefill attempt in
/// docs/p100-pxa-port.md is exactly that; Kaden's mmvq agrees -- his
/// block-wide activation stage and his 1-warp-x-4-rows geometry both lost,
/// while `split_rows`, where rows *share* one activation image, won).
///
/// Costs `kRowTile * Q::NV` registers for the decoded weights plus
/// `kRowTile * kBatchChunk` accumulators, which is what caps it -- and *that* is
/// the real constraint: at kBatchChunk 8 sm_60 ptxas wants 254 registers for
/// kRowTile 2 (one 256-thread block per SM, 0.6x as fast as the tile below) but
/// only 162 for kRowTile 4. Register pressure is not monotone in the tile, so
/// every value here has to be read off `ptxas -arch=sm_60 -v`, not guessed.
/// `SPITE_GEMV_ROW_TILE` overrides it for A/B measurement.
#ifndef SPITE_GEMV_ROW_TILE
#define SPITE_GEMV_ROW_TILE 4
#endif
constexpr int kRowTile = SPITE_GEMV_ROW_TILE;

/// Fewest row-groups the tiled kernel is worth launching for; below it the
/// one-row-per-block form runs instead. Under this many blocks the card is not
/// filled, the per-block serial walk the wider tile lengthens stops being
/// hidden, and the activation saving loses to it: on the host RTX 5090 (PTX JIT
/// of this sm_60 build, cols 5120, m 512) the crossover sits between 160 and 192
/// groups -- at 48 rows (12 groups) the tiled form is 0.78x while one row per
/// block is 1.27x, and 192 groups up it is 1.14-1.61x. 256 is the conservative
/// side of that, and it is a *host* measurement: re-measure on the P100s (56
/// SMs, 2 blocks/SM at 128 registers = 112 resident) before trusting it there.
/// `SPITE_GEMV_ROW_TILE_MIN_BLOCKS` overrides it for A/B measurement.
#ifndef SPITE_GEMV_ROW_TILE_MIN_BLOCKS
#define SPITE_GEMV_ROW_TILE_MIN_BLOCKS 256
#endif
constexpr int kRowTileMinBlocks = SPITE_GEMV_ROW_TILE_MIN_BLOCKS;

/// Columns per chunk in the tiled form. The chunk trades redundant weight
/// decode (m/kBatchChunk re-reads) against register pressure, and the right
/// answer moves with the tile: sm_60 ptxas spends 79 registers on
/// (kRowTile 1, kBatchChunk 8) and 162 on (4, 8) -- one 256-thread block per SM
/// -- but 128 on (4, 4), which is two blocks per SM. Measured on the host RTX
/// 5090, cols 5120, m 512: at kRowTile 1 raising the chunk 4 -> 8 is worth
/// 1.19x (0.601 -> 0.509 ms), at kRowTile 4 lowering it 8 -> 4 is worth 1.20x
/// (12.47 -> 10.42 ms at rows 17408), so the two forms want different chunks and
/// each gets its own. `SPITE_GEMV_ROW_WIDE_CHUNK` overrides the tiled form's.
#ifndef SPITE_GEMV_ROW_WIDE_CHUNK
#define SPITE_GEMV_ROW_WIDE_CHUNK 4
#endif
constexpr int kRowWideChunk = SPITE_GEMV_ROW_WIDE_CHUNK;

static_assert(kRowTile * kRowWideChunk <= kRowThreads &&
                  kBatchChunk <= kRowThreads,
              "write-back needs one thread per (row, column)");

/// `RT` = output rows this block owns, `CHUNK` = columns per chunk. Two forms
/// are instantiated: (kRowTile, kRowWideChunk) for the shapes with the blocks to
/// fill the card, and (1, kBatchChunk) for the rest (see kRowTileMinBlocks).
/// `RT` must divide nothing in particular -- rows need not be a multiple of it,
/// the surplus group is masked (and never addressed).
template <class Q, int RT, int CHUNK>
__global__ void __launch_bounds__(kRowThreads)
    gemv_batch_kernel(const uint8_t *__restrict__ w, float *__restrict__ y,
                      const float *__restrict__ x, int cols, int rows, int m,
                      size_t row_bytes, int x_aligned16, int accumulate) {
  static_assert(RT * CHUNK <= kRowThreads,
                "write-back needs one thread per (row, column)");
  __shared__ float red[kWarps][RT * CHUNK];
  const int row0 = blockIdx.x * RT;
  const int tid = threadIdx.x;

  for (int t0 = 0; t0 < m; t0 += CHUNK) {
    const int tn = min(CHUNK, m - t0);
    float acc[RT][CHUNK];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < CHUNK; ++j)
        acc[r][j] = 0.0f;

    if constexpr (Q::DENSE) {
      for (int c = tid; c < cols; c += kRowThreads) {
        float wv[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r)
          wv[r] = (row0 + r < rows)
                      ? Q::load(w + static_cast<size_t>(row0 + r) * row_bytes, c)
                      : 0.0f;
#pragma unroll
        for (int j = 0; j < CHUNK; ++j) {
          if (j >= tn)
            continue;
          const float xv = __ldg(x + static_cast<size_t>(t0 + j) * cols + c);
#pragma unroll
          for (int r = 0; r < RT; ++r)
            acc[r][j] = fmaf(wv[r], xv, acc[r][j]);
        }
      }
    } else {
      constexpr int NS = Q::NS, BPS = kRowThreads / NS;
      const int slot = tid % NS, sub = tid / NS;
      const int nb = cols / Q::QK;
      for (int b = sub; b < nb; b += BPS) {
        /* One decode per owned row, then one activation float4 per (column,
         * group) shared by all of them: the FMA count per activation load
         * goes 1 -> RT with the per-column order untouched. */
        float v[RT][Q::NV];
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          const int row = row0 + r;
          if (row < rows)
            Q::decode(w + static_cast<size_t>(row) * row_bytes +
                          static_cast<size_t>(b) * Q::BYTES,
                      slot, v[r]);
          else /* surplus row: never read, but keep v[] defined */
#pragma unroll
            for (int k = 0; k < Q::NV; ++k)
              v[r][k] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < CHUNK; ++j) {
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
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              acc[r][j] = fmaf(v[r][4 * g + 0], xv.x, acc[r][j]);
              acc[r][j] = fmaf(v[r][4 * g + 1], xv.y, acc[r][j]);
              acc[r][j] = fmaf(v[r][4 * g + 2], xv.z, acc[r][j]);
              acc[r][j] = fmaf(v[r][4 * g + 3], xv.w, acc[r][j]);
            }
          }
        }
      }
    }

    /* Cross-warp reduction, all RT * tn outputs at once. Doing it one
     * column at a time (a barrier pair each, with thread 0 walking the warps)
     * serialised the whole block CHUNK times per chunk. Every guard
     * here is block-uniform, so the shuffles stay convergent. */
    __syncthreads(); /* red may still be read by the previous chunk */
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      if (row0 + r >= rows)
        continue;
#pragma unroll
      for (int j = 0; j < CHUNK; ++j) {
        const float a = (j < tn) ? q35_warp_sum(acc[r][j]) : 0.0f;
        if ((tid & 31) == 0)
          red[tid >> 5][r * CHUNK + j] = a;
      }
    }
    __syncthreads();
    /* Write-back, one thread per (row, column). Deliberately keyed on
     * `tid < RT * tn` with a runtime divisor rather than the equally correct
     * `tid / CHUNK` and `% CHUNK`: the compile-time-divisor form measured
     * 1.4-2.4x SLOWER over a 384-768 row band (rows 640: 0.593 -> 1.266 ms,
     * reproduced across builds and interleaved runs) for what should be fewer
     * instructions, so it is a scheduling cliff and not work. This branch is
     * cold -- at most RT * CHUNK threads, once per chunk, never per element --
     * so the division is not worth chasing. Re-measure both forms on the P100s. */
    if (tid < RT * tn) {
      const int rr = tid / tn, j = tid - rr * tn;
      const int row = row0 + rr;
      if (row < rows) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < kWarps; ++i)
          t += red[i][rr * CHUNK + j];
        float *yp = y + static_cast<size_t>(t0 + j) * rows + row;
        *yp = accumulate ? *yp + t : t;
      }
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

/* Launch one batched kernel with RT output rows per block. */
template <class Q, int RT, int CHUNK>
void launch_batch_rows(const SpiteTensor *w, const float *x, float *y, int m,
                       int cols, int rows, size_t row_bytes, int x_aligned16,
                       int accumulate, cudaStream_t s) {
  gemv_batch_kernel<Q, RT, CHUNK>
      <<<static_cast<unsigned>((rows + RT - 1) / RT), kRowThreads, 0, s>>>(
          static_cast<const uint8_t *>(w->data), y, x, cols, rows, m, row_bytes,
          x_aligned16, accumulate);
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
    const int acc = accumulate ? 1 : 0;
    /* Tile rows only where there are enough blocks to fill the card
     * (kRowTileMinBlocks); both arms are compile-time, so a build only carries
     * the instantiations it can reach. */
    if (kRowTile > 1 && rows / kRowTile >= kRowTileMinBlocks)
      launch_batch_rows<Q, kRowTile, kRowWideChunk>(w, x, y, m, cols, rows,
                                                    row_bytes, x_aligned16, acc,
                                                    s);
    else
      launch_batch_rows<Q, 1, kBatchChunk>(w, x, y, m, cols, rows, row_bytes,
                                           x_aligned16, acc, s);
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
