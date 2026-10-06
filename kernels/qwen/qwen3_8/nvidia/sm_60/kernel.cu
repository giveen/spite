/*
 * kernels/qwen/qwen3_8/nvidia/sm_60/kernel.cu
 *
 * Pascal (GP100, sm_60) optimised kernel for Qwen3.8-27B.
 *
 * This layer sits between the vendor-wide nvidia/ baseline and the card-specific
 * tesla_p100/ layer.  It applies to every Pascal GPU (P100, GP100, Titan X Pascal,
 * etc.) and exploits features available on sm_60:
 *
 *   - __half2 SIMD: GP100 delivers 2x FP16 throughput via paired half2 ops.
 *     The matvec and rms_norm paths select the __half2 kernels from p100_fp16.cuh
 *     when the weight type is F16.
 *   - __ldg() read-only cache: all weight loads go through the texture cache.
 *   - No Tensor Cores / wmma / mma.sync — those require sm_70+.
 *   - No cp.async / async barriers — those require sm_80+.
 *
 * The flash-decode attention tile uses KVFLASH_TILE=32 and KVFLASH_WARPS=4
 * (128 threads/block).  K/V stored as __half2 → 32 × 64 × 2 × 4 = 16 KB/block,
 * enabling 4 concurrent blocks/SM on GP100 within its 64 KB shared memory.
 * TILE=32 keeps blockDim.x within one warp, making warp_sum() correct.
 *
 * Override: tesla_p100/kernel.cu further refines tile sizes and adds multi-GPU
 * NVLink all-reduce via multi_gpu.cuh.
 */

#include "core/abi.h"
#include "core/gpu/quant_gemv.h"
#include "kernels/qwen/qwen3_8/nvidia/sm_60/p100_fp16.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

struct BlockQ8_0 { __half d; int8_t qs[32]; };
static_assert(sizeof(BlockQ8_0) == 34, "");

/* Pascal: 8 output rows per block — more threads per SM on GP100 (64 SMs). */
constexpr int WARP           = 32;
constexpr int ROWS_PER_BLOCK = 8;

inline cudaStream_t stream_of(const SpiteCtx* ctx) {
    return ctx ? static_cast<cudaStream_t>(ctx->gpu_stream) : nullptr;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ float block_sum(float v) {
    __shared__ float r[WARP];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_sum(v);
    if (lane == 0) r[wid] = v;
    __syncthreads();
    v = (threadIdx.x < blockDim.x / WARP) ? r[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) r[0] = v;
    __syncthreads();
    return r[0];
}
__device__ float block_max(float v) {
    __shared__ float r[WARP];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_max(v);
    if (lane == 0) r[wid] = v;
    __syncthreads();
    v = (threadIdx.x < blockDim.x / WARP) ? r[threadIdx.x] : -INFINITY;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) r[0] = v;
    __syncthreads();
    return r[0];
}

__device__ __forceinline__ float load_w(const void* w, int kind, int i) {
    if (kind == SPITE_TYPE_F16) return __half2float(__ldg(static_cast<const __half*>(w) + i));
    return static_cast<const float*>(w)[i];
}

/* ── Matvec ─────────────────────────────────────────────────────────────── */

__global__ void matvec_q8_0(const BlockQ8_0* __restrict__ w, const float* __restrict__ x,
                              float* __restrict__ y, int rows, int cols, int acc) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int nb = cols / 32;
    const BlockQ8_0* wr = w + static_cast<size_t>(row) * nb;
    float a = 0.0f;
    for (int b = 0; b < nb; ++b)
        a += __half2float(__ldg(&wr[b].d)) * __ldg(&wr[b].qs[threadIdx.x]) * x[b * 32 + threadIdx.x];
    a = warp_sum(a);
    if (threadIdx.x == 0) y[row] = acc ? y[row] + a : a;
}

/* fp32 weight fallback (also used for F32 activations) */
__global__ void matvec_f32(const float* __restrict__ w, const float* __restrict__ x,
                             float* __restrict__ y, int rows, int cols, int acc) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const float* wr = w + static_cast<size_t>(row) * cols;
    float a = 0.0f;
    for (int c = threadIdx.x; c < cols; c += WARP) a += __ldg(wr + c) * x[c];
    a = warp_sum(a);
    if (threadIdx.x == 0) y[row] = acc ? y[row] + a : a;
}

int launch_matvec(const SpiteTensor* w, const float* x, float* y, bool accumulate,
                  cudaStream_t s) {
    const int cols = static_cast<int>(w->ne[0]);
    const int rows = static_cast<int>(w->ne[1] ? w->ne[1] : 1);
    const dim3 block(WARP, ROWS_PER_BLOCK);
    const dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    switch (w->kind) {
    case SPITE_TYPE_Q8_0:
        if (cols % 32) return -1;
        matvec_q8_0<<<grid, block, 0, s>>>(
            static_cast<const BlockQ8_0*>(w->data), x, y, rows, cols, accumulate);
        break;
    case SPITE_TYPE_F16:
        /* Use __half2 SIMD path for F16 weights (2x throughput on GP100). */
        if (cols % 2 == 0) {
            const dim3 blk_h2(WARP, ROWS_PER_BLOCK);
            const dim3 grd_h2((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
            matvec_f16_h2<ROWS_PER_BLOCK><<<grd_h2, blk_h2, 0, s>>>(
                static_cast<const __half*>(w->data), x, y, rows, cols, accumulate);
            break;
        }
        /* fall through to fp32 path for odd cols */
        [[fallthrough]];
    case SPITE_TYPE_F32:
        matvec_f32<<<grid, block, 0, s>>>(
            static_cast<const float*>(w->data), x, y, rows, cols, accumulate);
        break;
    default:
        return sq::gemv(w, x, y, accumulate, s);
    }
    return 0;
}

/* ── RMSNorm ────────────────────────────────────────────────────────────── */

__global__ void rms_norm_rows(float* __restrict__ out, const float* __restrict__ x,
                               const void* __restrict__ w, int wkind, int cols, float eps) {
    const float* xr  = x   + static_cast<size_t>(blockIdx.x) * cols;
    float*       orow = out + static_cast<size_t>(blockIdx.x) * cols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        orow[i] = xr[i] * sc * load_w(w, wkind, i);
}

/* ── GLU activation ─────────────────────────────────────────────────────── */

__global__ void glu_act(float* __restrict__ gate, const float* __restrict__ up,
                         int n, int gelu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float a = gelu
        ? 0.5f * g * (1.0f + tanhf(0.7978846f * (g + 0.044715f * g * g * g)))
        : g / (1.0f + __expf(-g));
    gate[i] = a * up[i];
}

__global__ void attn_softmax(float* __restrict__ scores, int n_tok, int n_ctx) {
    float* s = scores + static_cast<size_t>(blockIdx.x) * n_ctx;
    float m = -INFINITY;
    for (int t = threadIdx.x; t < n_tok; t += blockDim.x) m = fmaxf(m, s[t]);
    m = block_max(m);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_tok; t += blockDim.x) {
        const float e = __expf(s[t] - m); s[t] = e; sum += e;
    }
    sum = block_sum(sum);
    const float inv = 1.0f / sum;
    for (int t = threadIdx.x; t < n_tok; t += blockDim.x) s[t] *= inv;
}

inline int threads_for(int n) {
    int t = ((n + WARP - 1) / WARP) * WARP;
    return t < WARP ? WARP : (t > 1024 ? 1024 : t);
}
inline int finish() { return cudaGetLastError() == cudaSuccess ? 0 : -2; }

}  // namespace

/*
 * Pascal flash-decode tile: KVFLASH_TILE=32 (32 KV steps per tile, one warp),
 * KVFLASH_WARPS=4 (4 query heads per block, 128 threads total).
 * K/V stored as __half2 in smem → 2 × 32 × 64 × 4 = 16 384 bytes = 16 KB/block.
 * TILE=32 ensures blockDim.x fits in one warp, making warp_sum() correct.
 */
#define KVFLASH_TILE  32
#define KVFLASH_WARPS 4

#include "kernels/qwen/qwen3_8/nvidia/kv_attn.inl"
#include "kernels/qwen/qwen3_8/nvidia/kv_attn_flash.inl"

/* ── ABI entry points ────────────────────────────────────────────────────── */

extern "C" int qwen38_sm60_rms_norm(SpiteTensor* out, const SpiteTensor* x,
                                     const SpiteTensor* weight, float eps,
                                     const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (weight->kind != SPITE_TYPE_F32 && weight->kind != SPITE_TYPE_F16) return -1;
    const int cols = static_cast<int>(x->ne[0]);
    const int rows = static_cast<int>(x->ne[1] ? x->ne[1] : 1);
    if (weight->ne[0] != x->ne[0]) return -1;
    /* Use __half2 rms_norm for f16 weights (2x bandwidth on GP100). */
    if (weight->kind == SPITE_TYPE_F16 && cols % 2 == 0) {
        rms_norm_f16w_h2<<<rows, threads_for(cols), 0, stream_of(ctx)>>>(
            static_cast<float*>(out->data), static_cast<const float*>(x->data),
            static_cast<const __half*>(weight->data), cols, eps);
    } else {
        rms_norm_rows<<<rows, threads_for(cols), 0, stream_of(ctx)>>>(
            static_cast<float*>(out->data), static_cast<const float*>(x->data),
            weight->data, weight->kind, cols, eps);
    }
    return finish();
}

extern "C" int qwen38_sm60_attention(SpiteTensor* out, const SpiteTensor* x,
                                      const SpiteTensor* wq, const SpiteTensor* wk,
                                      const SpiteTensor* wv, const SpiteTensor* wo,
                                      const SpiteTensor* q_norm, const SpiteTensor* k_norm,
                                      float norm_eps, SpiteKvCache* kv,
                                      float rope_freq_base, const SpiteCtx* ctx) {
    return kvflash_run(out, x, wq, wk, wv, wo, q_norm, k_norm,
                       norm_eps, kv, rope_freq_base, ctx);
}

extern "C" int qwen38_sm60_ffn(SpiteTensor* out, const SpiteTensor* x,
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
    if (launch_matvec(w_gate, static_cast<const float*>(x->data), gate, false, s)) return -1;
    if (launch_matvec(w_up,   static_cast<const float*>(x->data), up,   false, s)) return -1;
    glu_act<<<(d_ffn + 255) / 256, 256, 0, s>>>(gate, up, d_ffn, act == SPITE_FFN_GELU_GATE);
    if (launch_matvec(w_down, gate, static_cast<float*>(out->data), true, s)) return -1;
    return finish();
}

extern "C" int qwen38_sm60_matmul(SpiteTensor* out, const SpiteTensor* x,
                                   const SpiteTensor* w, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (launch_matvec(w, static_cast<const float*>(x->data),
                      static_cast<float*>(out->data), false, stream_of(ctx))) return -1;
    return finish();
}

extern "C" uint64_t qwen38_sm60_kv_cache_kinds() {
    return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16) |
           (1ull << SPITE_TYPE_Q8_0) | (1ull << SPITE_TYPE_Q5_1) |
           (1ull << SPITE_TYPE_Q4_0);
}

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen38",
    "cuda",
    "spite project (qwen3.8 pascal sm_60: __half2 SIMD, 64-step KV tile)",
    {SPITE_TYPE_F16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q4_K, SPITE_TYPE_Q5_K,
     SPITE_TYPE_Q6_K, SPITE_TYPE_Q4_0, SPITE_TYPE_Q3_K, 0},
    qwen38_sm60_rms_norm,
    qwen38_sm60_attention,
    nullptr, /* mla */
    qwen38_sm60_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    qwen38_sm60_matmul,
    qwen38_sm60_kv_cache_kinds,
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
