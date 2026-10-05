/*
 * kernels/qwen/qwen3/nvidia/sm_120/rtx_5090/kernel.cu
 *
 * RTX 5090-specific kernel for Qwen3 dense decoders.
 * Scope: exclusively NVIDIA GeForce RTX 5090.
 *
 * Physical constraints exploited on top of sm_120 vectorization:
 *   • 192 SMs: grid dimensions padded to multiples of 192 blocks so all
 *     multiprocessors stay 100% occupied with no tail-wave stalls.
 *   • The Q8_0 matvec, which is 90% of a decode token, is run as a K-split:
 *     the op is latency-bound at 40% occupancy, and the warp count is what
 *     hides that latency, so below the crossover the row's K range is split
 *     across 8 warps (see pick_wpr).  Measured 1.22x cold on one Qwen3-8B
 *     layer, 1.08x..1.57x depending on the projection.
 *   • Shared memory staging of the activation vector is kept for the shapes
 *     whose row axis already fills the part, where it divides the activation
 *     reads across a whole 8-row block.
 *
 * Attention is the sm_120 flash tile kernel (kv_attn_flash_sm120.inl) driven by
 * the shared flash-decoding scaffolding: this card's own tuning stays in the
 * projections, which are 90% of a decode token.
 *
 * Weight types: F32, F16, Q8_0. Activations: F32.
 * KV cache: F32, F16, Q8_0, Q5_1 or Q4_0 (see kv_attn.inl).
 * Single-token decode semantics (prefill = repeated decode by the host).
 */

#include "core/abi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int QK8_0 = 32;
struct BlockQ8_0 {
    __half d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must be 34 bytes");

constexpr int WARP          = 32;
constexpr int ROWS_PER_BLOCK = 8;   /* 8 warps/block — double sm_120 baseline */
constexpr int RTX5090_SMS   = 192;  /* physical SM count on RTX 5090 */

inline cudaStream_t stream_of(const SpiteCtx* ctx) {
    return ctx ? static_cast<cudaStream_t>(ctx->gpu_stream) : nullptr;
}

/* Fully unrolled warp reductions (same as sm_120). */
__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v,  8);
    v += __shfl_xor_sync(0xffffffffu, v,  4);
    v += __shfl_xor_sync(0xffffffffu, v,  2);
    v += __shfl_xor_sync(0xffffffffu, v,  1);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v,  8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v,  4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v,  2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v,  1));
    return v;
}

/* block_sum / block_max support up to 32 warps (1024 threads). */
__device__ float block_sum(float v) {
    __shared__ float red[WARP];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_sum(v);
    if (lane == 0) red[wid] = v;
    __syncthreads();
    const int nw = blockDim.x / WARP;
    v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

__device__ float block_max(float v) {
    __shared__ float red[WARP];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_max(v);
    if (lane == 0) red[wid] = v;
    __syncthreads();
    const int nw = blockDim.x / WARP;
    v = (threadIdx.x < nw) ? red[threadIdx.x] : -INFINITY;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

__device__ __forceinline__ float load_w(const void* w, int kind, int i) {
    if (kind == SPITE_TYPE_F16) return __half2float(static_cast<const __half*>(w)[i]);
    return static_cast<const float*>(w)[i];
}

/* Round n up to the nearest multiple of RTX5090_SMS so every SM gets at
 * least one block.  Extra blocks that fall out of range early-exit cheaply. */
inline int pad192(int n) {
    return ((n + RTX5090_SMS - 1) / RTX5090_SMS) * RTX5090_SMS;
}

/* K-split factor for the Q8_0 matvec: how many warps share one output row.
 *
 * The op is latency-bound, not DRAM-bound.  Profiled on the 4096x4096
 * projection it runs at 46.9% of DRAM peak with L1TEX at 47.6% and 40%
 * occupancy: the load path is half empty and short of warps to cover it.  One
 * warp per row gives the matvec exactly `rows` warps of parallelism, so the
 * 1024-row k/v projections put 5.3 warps/SM on a 192-SM part and read 4.2 MiB
 * in 8.9 us -- 461 GB/s, 26% of this card's 1792 GB/s ceiling -- while the
 * 12288-row projections, identical in every other respect, reach 1451 GB/s
 * (81%) at 64 warps/SM.  Splitting each row's K range across WPR warps
 * multiplies the warp count by WPR and closes that gap.
 *
 * Measured cold, one Qwen3-8B layer (195.5 MiB, weights outside the 96 MB L2),
 * sum of the seven projections in us and the resulting weight-read rate:
 *
 *   schedule    1x1     1x2     1x4     1x8    2x1     2x4     4x1
 *   us        174.5   156.5   144.9   142.9  191.7   145.5   296.5
 *   GB/s       1175    1310    1415    1434   1070    1409     691
 *
 * The 2x1/4x1 columns are register blocking (R rows per warp, reusing one
 * activation element across them).  It divides the activation traffic as
 * intended and still loses badly, because it divides the warp count too: the
 * warp count is what this op is actually short of.  WPR=8 is the whole win.
 *
 * Above RTX5090_SMS*64 rows the row axis fills the part on its own (64
 * warps/SM is every warp a Blackwell SM has), and there the one-row-per-warp
 * shared-memory kernel wins instead: at 151936 rows it reads 1763 GB/s against
 * the split's 1707, because 8 warps in a block can share one staged activation
 * tile rather than re-reading it per row.  So the split is used below that
 * crossover and the staged kernel above it.  The 12288-row projections sit
 * just under it and take the split (12% faster); 16384 rows and up measured
 * within 1.4% of the staged kernel and 24576+ slightly behind it.
 *
 * Never splits below 8 VBR blocks of K per warp, so the partial reduce stays a
 * rounding error next to the loop. */
inline int pick_wpr(int rows, int nb) {
    if (rows > RTX5090_SMS * 64) return 1;
    for (int wpr = 8; wpr >= 2; wpr >>= 1) {
        if (ROWS_PER_BLOCK % wpr == 0 && nb % wpr == 0 && nb / wpr >= 8) return wpr;
    }
    return 1;
}

// ── Matvec ─────────────────────────────────────────────────────────────────

/* Q8_0 matvec with the K dimension split across WPR warps per row.
 *
 * Deliberately has no shared memory.  With WPR > 1 each warp owns a disjoint
 * K range and reads every activation element exactly once, so a per-block
 * staging copy would be dead weight; the only cross-warp traffic is the
 * ROWS_PER_BLOCK/WPR partials per row, which meet in a small static array. */
template <int WPR>
__global__ void matvec_q8_0_ksplit(const BlockQ8_0* __restrict__ w, const float* __restrict__ x,
                                   float* __restrict__ y, int rows, int cols, int accumulate) {
    static_assert(WPR >= 2 && WPR <= 8 && ROWS_PER_BLOCK % WPR == 0);
    constexpr int ROWS_PER_CTA = ROWS_PER_BLOCK / WPR;
    __shared__ float partial[ROWS_PER_CTA][WPR];

    const int lane        = threadIdx.x;
    const int row_in_cta  = threadIdx.y / WPR;
    const int warp_in_row = threadIdx.y % WPR;
    const int row         = blockIdx.x * ROWS_PER_CTA + row_in_cta;
    /* Not an early return: the padding blocks pad192() adds have to reach the
     * __syncthreads that collects the partials too. */
    const bool active = row < rows;

    const int nb    = cols / QK8_0;
    const int chunk = nb / WPR;
    float acc       = 0.0f;
    if (active) {
        const BlockQ8_0* wr = w + static_cast<size_t>(row) * nb;
        const int b0        = warp_in_row * chunk;
        const int b1        = b0 + chunk;
        for (int b = b0; b < b1; ++b) {
            const float d = __half2float(wr[b].d);
            acc += d * static_cast<float>(wr[b].qs[lane]) * x[b * QK8_0 + lane];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) partial[row_in_cta][warp_in_row] = acc;
    __syncthreads();
    if (active && warp_in_row == 0 && lane == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int i = 0; i < WPR; ++i) tot += partial[row_in_cta][i];
        y[row] = accumulate ? y[row] + tot : tot;
    }
}

/* Q8_0 matvec, one warp per row, activation vector staged in shared memory.
 * All ROWS_PER_BLOCK warps in the block load x into smem once, then each warp
 * reads its row's weights from global memory while feeding activations from
 * fast L1 rather than L2/DRAM.
 *
 * Dynamic shared memory must be allocated by the caller:
 *   cols * sizeof(float) bytes. */
__global__ void matvec_q8_0_smem(const BlockQ8_0* __restrict__ w, const float* __restrict__ x,
                                  float* __restrict__ y, int rows, int cols, int accumulate) {
    extern __shared__ float sx[];  /* cached activation vector */

    /* Collaborative load: all threads fill sx[0..cols-1]. */
    const int tid = threadIdx.y * WARP + threadIdx.x;
    const int blk_threads = ROWS_PER_BLOCK * WARP;
    for (int i = tid; i < cols; i += blk_threads)
        sx[i] = x[i];
    __syncthreads();

    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;  /* safe after __syncthreads */

    const int nb = cols / QK8_0;
    const BlockQ8_0* wr = w + static_cast<size_t>(row) * nb;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float d = __half2float(wr[b].d);
        acc += d * static_cast<float>(wr[b].qs[threadIdx.x]) * sx[b * QK8_0 + threadIdx.x];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

/* Fallback Q8_0 matvec (no smem) for tiles too large to stage. */
__global__ void matvec_q8_0(const BlockQ8_0* __restrict__ w, const float* __restrict__ x,
                            float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int nb = cols / QK8_0;
    const BlockQ8_0* wr = w + static_cast<size_t>(row) * nb;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float d = __half2float(wr[b].d);
        acc += d * static_cast<float>(wr[b].qs[threadIdx.x]) * x[b * QK8_0 + threadIdx.x];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

/* F32 dense matvec: float4 vectorized. */
__global__ void matvec_f32(const float* __restrict__ w, const float* __restrict__ x,
                           float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const float* wr = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    if (cols % 4 == 0) {
        const float4* wr4 = reinterpret_cast<const float4*>(wr);
        const float4* x4  = reinterpret_cast<const float4*>(x);
        const int cols4 = cols / 4;
        for (int c = threadIdx.x; c < cols4; c += WARP) {
            float4 wv = wr4[c], xv = x4[c];
            acc += wv.x*xv.x + wv.y*xv.y + wv.z*xv.z + wv.w*xv.w;
        }
    } else {
        for (int c = threadIdx.x; c < cols; c += WARP) acc += wr[c] * x[c];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

/* F16 dense matvec: __half2 weight loads + float2 activation loads. */
__global__ void matvec_f16(const __half* __restrict__ w, const float* __restrict__ x,
                           float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const __half* wr = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    if (cols % 2 == 0) {
        const __half2* wr2 = reinterpret_cast<const __half2*>(wr);
        const float2*  x2  = reinterpret_cast<const float2*>(x);
        const int cols2 = cols / 2;
        for (int c = threadIdx.x; c < cols2; c += WARP) {
            float2 wvf = __half22float2(wr2[c]);
            float2 xv  = x2[c];
            acc += wvf.x * xv.x + wvf.y * xv.y;
        }
    } else {
        for (int c = threadIdx.x; c < cols; c += WARP)
            acc += __half2float(wr[c]) * x[c];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

/* Launch matvec with 192-SM grid padding and smem staging for Q8_0. */
int launch_matvec(const SpiteTensor* w, const float* x, float* y, bool accumulate,
                  cudaStream_t s) {
    const int cols = static_cast<int>(w->ne[0]);
    const int rows = static_cast<int>(w->ne[1] ? w->ne[1] : 1);
    const dim3 block(WARP, ROWS_PER_BLOCK);
    /* Pad to multiple of 192 so all SMs receive at least one block. */
    const dim3 grid(pad192((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK));
    switch (w->kind) {
    case SPITE_TYPE_Q8_0: {
        if (cols % QK8_0) return -1;
        const int nb  = cols / QK8_0;
        const int wpr = pick_wpr(rows, nb);
        const BlockQ8_0* wq = static_cast<const BlockQ8_0*>(w->data);
        if (wpr > 1) {
            const dim3 kblock(WARP, ROWS_PER_BLOCK);
            const dim3 kgrid(pad192((rows + ROWS_PER_BLOCK / wpr - 1) /
                                    (ROWS_PER_BLOCK / wpr)));
            switch (wpr) {
            case 2: matvec_q8_0_ksplit<2><<<kgrid, kblock, 0, s>>>(wq, x, y, rows, cols,
                                                                   accumulate); break;
            case 4: matvec_q8_0_ksplit<4><<<kgrid, kblock, 0, s>>>(wq, x, y, rows, cols,
                                                                   accumulate); break;
            default: matvec_q8_0_ksplit<8><<<kgrid, kblock, 0, s>>>(wq, x, y, rows, cols,
                                                                    accumulate); break;
            }
            break;
        }
        const size_t smem = static_cast<size_t>(cols) * sizeof(float);
        /* 48 KB is the default per-block dynamic shared memory cap; asking for
         * more is a launch failure, not a fallback. */
        if (smem <= 49152) {
            matvec_q8_0_smem<<<grid, block, smem, s>>>(wq, x, y, rows, cols, accumulate);
        } else {
            matvec_q8_0<<<grid, block, 0, s>>>(wq, x, y, rows, cols, accumulate);
        }
        break;
    }
    case SPITE_TYPE_F32:
        matvec_f32<<<grid, block, 0, s>>>(static_cast<const float*>(w->data), x, y, rows,
                                          cols, accumulate);
        break;
    case SPITE_TYPE_F16:
        matvec_f16<<<grid, block, 0, s>>>(static_cast<const __half*>(w->data), x, y, rows,
                                          cols, accumulate);
        break;
    default:
        return -1;
    }
    return 0;
}

// ── RMSNorm: vectorized float4 + half2 (same as sm_120) ──────────────────

__global__ void rms_norm_rows(float* __restrict__ out, const float* __restrict__ x,
                              const void* __restrict__ w, int wkind, int cols, float eps) {
    const float* xr   = x   + static_cast<size_t>(blockIdx.x) * cols;
    float*       orow = out + static_cast<size_t>(blockIdx.x) * cols;
    float ss = 0.0f;
    if (cols % 4 == 0) {
        const float4* xr4 = reinterpret_cast<const float4*>(xr);
        const int cols4 = cols / 4;
        for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
            float4 v = xr4[i];
            ss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    }
    ss = block_sum(ss);
    const float scale = rsqrtf(ss / cols + eps);

    if (cols % 4 == 0 && wkind == SPITE_TYPE_F32) {
        const float4* xr4 = reinterpret_cast<const float4*>(xr);
        const float4* w4  = reinterpret_cast<const float4*>(static_cast<const float*>(w));
        float4* o4 = reinterpret_cast<float4*>(orow);
        const int cols4 = cols / 4;
        for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
            float4 v = xr4[i], wv = w4[i];
            float4 o;
            o.x = v.x * scale * wv.x;
            o.y = v.y * scale * wv.y;
            o.z = v.z * scale * wv.z;
            o.w = v.w * scale * wv.w;
            o4[i] = o;
        }
    } else if (cols % 2 == 0 && wkind == SPITE_TYPE_F16) {
        const float2*  xr2 = reinterpret_cast<const float2*>(xr);
        const __half2* w2  = reinterpret_cast<const __half2*>(static_cast<const __half*>(w));
        float2* o2 = reinterpret_cast<float2*>(orow);
        const int cols2 = cols / 2;
        for (int i = threadIdx.x; i < cols2; i += blockDim.x) {
            float2 v  = xr2[i];
            float2 wv = __half22float2(w2[i]);
            float2 o  = {v.x * scale * wv.x, v.y * scale * wv.y};
            o2[i] = o;
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            orow[i] = xr[i] * scale * load_w(w, wkind, i);
    }
}

// ── Attention pieces ─────────────────────────────────────────────────────

// The F32-only qk_norm_rope / attn_scores / attn_weighted_v that used to live
// here are superseded by the shared VBR kernels in kv_attn.inl; only the
// softmax below is still shared with that file.

__global__ void attn_softmax(float* __restrict__ scores, int n_tok, int n_ctx) {
    float* s = scores + static_cast<size_t>(blockIdx.x) * n_ctx;
    float m = -INFINITY;
    for (int t = threadIdx.x; t < n_tok; t += blockDim.x) m = fmaxf(m, s[t]);
    m = block_max(m);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_tok; t += blockDim.x) {
        const float e = __expf(s[t] - m);
        s[t] = e;
        sum += e;
    }
    sum = block_sum(sum);
    const float inv = 1.0f / sum;
    for (int t = threadIdx.x; t < n_tok; t += blockDim.x) s[t] *= inv;
}

// ── FFN activation ────────────────────────────────────────────────────────

__global__ void glu_act(float* __restrict__ gate, const float* __restrict__ up, int n, int gelu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float a = gelu ? 0.5f * g * (1.0f + tanhf(0.7978846f * (g + 0.044715f * g * g * g)))
                         : g / (1.0f + __expf(-g));
    gate[i] = a * up[i];
}

inline int threads_for(int n) {
    int t = ((n + WARP - 1) / WARP) * WARP;
    return t < WARP ? WARP : (t > 1024 ? 1024 : t);
}

inline int finish() { return cudaGetLastError() == cudaSuccess ? 0 : -2; }

}  // namespace

// Shared VBR (quantized-KV) attention. Uses this file's launch_matvec,
// threads_for, stream_of, attn_softmax, block_sum and load_w.
#include "kernels/qwen/qwen3/nvidia/kv_attn.inl"

// sm_120 flash tile kernel + hd dispatch (this card shares the arch-level tile
// kernel; SPITE_KVFLASH_ARCH keeps the shared scaffolding from emitting the
// portable tile kernel too).
#define SPITE_KVFLASH_ARCH 1
#include "kernels/qwen/qwen3/nvidia/sm_120/kv_attn_flash_sm120.inl"

// Flash-decoding scaffolding: chunking, split-K workspace, combine, entry point.
#include "kernels/qwen/qwen3/nvidia/kv_attn_flash.inl"

// ── ABI ops ───────────────────────────────────────────────────────────────

extern "C" int qwen3_rtx5090_rms_norm(SpiteTensor* out, const SpiteTensor* x,
                                      const SpiteTensor* weight, float eps, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    const int cols = static_cast<int>(x->ne[0]);
    const int rows = static_cast<int>(x->ne[1] ? x->ne[1] : 1);
    if (weight->ne[0] != x->ne[0]) return -1;
    rms_norm_rows<<<rows, threads_for(cols), 0, stream_of(ctx)>>>(
        static_cast<float*>(out->data), static_cast<const float*>(x->data),
        weight->data, weight->kind, cols, eps);
    return finish();
}

extern "C" int qwen3_rtx5090_attention(SpiteTensor* out, const SpiteTensor* x,
                                    const SpiteTensor* wq, const SpiteTensor* wk,
                                    const SpiteTensor* wv, const SpiteTensor* wo,
                                    const SpiteTensor* q_norm, const SpiteTensor* k_norm,
                                    float norm_eps, SpiteKvCache* kv, float rope_freq_base,
                                    const SpiteCtx* ctx) {
    return kvflash_run(out, x, wq, wk, wv, wo, q_norm, k_norm, norm_eps, kv, rope_freq_base, ctx);
}

extern "C" int qwen3_rtx5090_ffn(SpiteTensor* out, const SpiteTensor* x,
                                 const SpiteTensor* w_gate, const SpiteTensor* w_up,
                                 const SpiteTensor* w_down, SpiteFfnActivation act,
                                 const SpiteCtx* ctx) {
    if (act != SPITE_FFN_SILU_GATE && act != SPITE_FFN_GELU_GATE) return -1;
    if (!ctx) return -1;
    const int d_ffn = static_cast<int>(w_gate->ne[1]);
    const size_t need = sizeof(float) * 2 * static_cast<size_t>(d_ffn);
    if (!ctx->scratchpad || ctx->scratchpad_bytes < need) return -2;
    float* gate = static_cast<float*>(ctx->scratchpad);
    float* up   = gate + d_ffn;
    cudaStream_t s = stream_of(ctx);
    const float* xin = static_cast<const float*>(x->data);
    if (launch_matvec(w_gate, xin, gate, false, s) ||
        launch_matvec(w_up,   xin, up,   false, s))
        return -1;
    glu_act<<<(d_ffn + 255) / 256, 256, 0, s>>>(gate, up, d_ffn, act == SPITE_FFN_GELU_GATE);
    if (launch_matvec(w_down, gate, static_cast<float*>(out->data), true, s)) return -1;
    return finish();
}

extern "C" int qwen3_rtx5090_matmul(SpiteTensor* out, const SpiteTensor* x,
                                    const SpiteTensor* w, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (launch_matvec(w, static_cast<const float*>(x->data),
                      static_cast<float*>(out->data), false, stream_of(ctx)))
        return -1;
    return finish();
}

// ── Kernel descriptor ─────────────────────────────────────────────────────

/* KV-cache tiers this attention op reads and writes (bit = SpiteType). */
extern "C" uint64_t qwen3_rtx5090_kv_cache_kinds() {
    return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16) | (1ull << SPITE_TYPE_Q8_0) |
           (1ull << SPITE_TYPE_Q5_1) | (1ull << SPITE_TYPE_Q4_0);
}

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen3",
    // Architecture, not the card: `KernelSpec` carries the card separately and
    // the host maps arch -> vendor with `company_from_arch`. Declaring the
    // card here (as this file did) makes that mapping fall through to
    // "generic", which drops the host onto the CPU fallback.
    "sm_120",
    "spite project (RTX 5090: 192-SM grid, 8-warp K-split Q8_0 matvec, smem x-staging)",
    {SPITE_TYPE_F32, SPITE_TYPE_F16, SPITE_TYPE_Q8_0, 0, 0, 0, 0, 0},
    qwen3_rtx5090_rms_norm,
    qwen3_rtx5090_attention,
    nullptr, /* mla */
    qwen3_rtx5090_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    qwen3_rtx5090_matmul,
    qwen3_rtx5090_kv_cache_kinds,
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
