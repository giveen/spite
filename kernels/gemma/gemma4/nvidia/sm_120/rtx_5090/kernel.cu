/*
 * kernels/gemma/gemma4/nvidia/sm_120/rtx_5090/kernel.cu
 *
 * RTX 5090-specific kernel for Gemma 4.
 * Scope: exclusively NVIDIA GeForce RTX 5090.
 *
 * Physical constraints exploited on top of sm_120 vectorization:
 *   • 192 SMs: grid dimensions padded to multiples of 192 blocks so all
 *     multiprocessors stay occupied with minimal tail-wave stalls.
 *   • The Q8_0 matvec, which is the bulk of a decode token, is run as a K-split:
 *     below the crossover the row's K range is split across up to 8 warps (pick_wpr).
 *   • Shared memory staging of the activation vector is kept for large row shapes.
 *   • Vectorized 128-bit memory transactions (float4, half2).
 *   • GeGLU activation with fused tanh GELU approximation.
 *
 * Attention uses the sm_120 flash tile kernel (kv_attn_flash_sm120.inl) driven by
 * the flash-decoding scaffolding.
 *
 * Weight types: any SpiteType (F32/F16/Q8_0 tuned here, the rest via core/gpu/quant_gemv.h).
 * Activations: F32. KV cache: F32, F16, Q8_0, Q5_1 or Q4_0.
 */

#include "core/abi.h"
#include "core/gpu/quant_gemv.h"  // also #defines QK8_0 (ggml-common.h)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

struct BlockQ8_0 {
    __half d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must be 34 bytes");

constexpr int WARP          = 32;
constexpr int ROWS_PER_BLOCK = 8;   /* 8 warps/block */
constexpr int RTX5090_SMS   = 192;  /* physical SM count / warp-wave schedule on RTX 5090 */

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

inline int pad192(int n) {
    return ((n + RTX5090_SMS - 1) / RTX5090_SMS) * RTX5090_SMS;
}

inline int pick_wpr(int rows, int nb) {
    if (rows > RTX5090_SMS * 64) return 1;
    for (int wpr = 8; wpr >= 2; wpr >>= 1) {
        if (ROWS_PER_BLOCK % wpr == 0 && nb % wpr == 0 && nb / wpr >= 8) return wpr;
    }
    return 1;
}

// ── Matvec ─────────────────────────────────────────────────────────────────

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

__global__ void matvec_q8_0_smem(const BlockQ8_0* __restrict__ w, const float* __restrict__ x,
                                 float* __restrict__ y, int rows, int cols, int accumulate) {
    extern __shared__ float sx[];
    const int tid = threadIdx.y * WARP + threadIdx.x;
    const int blk_threads = ROWS_PER_BLOCK * WARP;
    for (int i = tid; i < cols; i += blk_threads)
        sx[i] = x[i];
    __syncthreads();

    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;

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

int launch_matvec(const SpiteTensor* w, const float* x, float* y, bool accumulate,
                  cudaStream_t s) {
    const int cols = static_cast<int>(w->ne[0]);
    const int rows = static_cast<int>(w->ne[1] ? w->ne[1] : 1);
    const dim3 block(WARP, ROWS_PER_BLOCK);
    switch (w->kind) {
    case SPITE_TYPE_Q8_0: {
        if (cols % QK8_0) return -1;
        const int nb = cols / QK8_0;
        const int wpr = pick_wpr(rows, nb);
        if (wpr > 1) {
            const int rpc = ROWS_PER_BLOCK / wpr;
            const int raw_grid = (rows + rpc - 1) / rpc;
            const dim3 grid(pad192(raw_grid));
            const auto* ptr = static_cast<const BlockQ8_0*>(w->data);
            if      (wpr == 8) matvec_q8_0_ksplit<8><<<grid, block, 0, s>>>(ptr, x, y, rows, cols, accumulate);
            else if (wpr == 4) matvec_q8_0_ksplit<4><<<grid, block, 0, s>>>(ptr, x, y, rows, cols, accumulate);
            else if (wpr == 2) matvec_q8_0_ksplit<2><<<grid, block, 0, s>>>(ptr, x, y, rows, cols, accumulate);
        } else {
            const dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
            const size_t smem = static_cast<size_t>(cols) * sizeof(float);
            if (smem <= 48 * 1024) {
                matvec_q8_0_smem<<<grid, block, smem, s>>>(
                    static_cast<const BlockQ8_0*>(w->data), x, y, rows, cols, accumulate);
            } else {
                matvec_q8_0<<<grid, block, 0, s>>>(
                    static_cast<const BlockQ8_0*>(w->data), x, y, rows, cols, accumulate);
            }
        }
        break;
    }
    case SPITE_TYPE_F32: {
        const dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        matvec_f32<<<grid, block, 0, s>>>(static_cast<const float*>(w->data), x, y, rows,
                                          cols, accumulate);
        break;
    }
    case SPITE_TYPE_F16: {
        const dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        matvec_f16<<<grid, block, 0, s>>>(static_cast<const __half*>(w->data), x, y, rows,
                                          cols, accumulate);
        break;
    }
    default:
        return sq::gemv(w, x, y, accumulate, s);
    }
    return 0;
}

// ── RMSNorm ───────────────────────────────────────────────────────────────

__global__ void rms_norm_rows(float* __restrict__ out, const float* __restrict__ x,
                              const void* __restrict__ w, int wkind, int cols, float eps) {
    const float* xr  = x   + static_cast<size_t>(blockIdx.x) * cols;
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

// Shared VBR (quantized-KV) attention.
#include "kernels/gemma/gemma4/nvidia/kv_attn.inl"

// Blackwell flash tile kernel + hd dispatch.
#define SPITE_KVFLASH_ARCH 1
#include "kernels/gemma/gemma4/nvidia/sm_120/kv_attn_flash_sm120.inl"

// Flash-decoding scaffolding.
#include "kernels/gemma/gemma4/nvidia/kv_attn_flash.inl"

// ── ABI ops ───────────────────────────────────────────────────────────────

extern "C" int gemma4_rtx5090_rms_norm(SpiteTensor* out, const SpiteTensor* x,
                                       const SpiteTensor* weight, float eps, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (weight->kind != SPITE_TYPE_F32 && weight->kind != SPITE_TYPE_F16) return -1;
    const int cols = static_cast<int>(x->ne[0]);
    const int rows = static_cast<int>(x->ne[1] ? x->ne[1] : 1);
    if (weight->ne[0] != x->ne[0]) return -1;
    rms_norm_rows<<<rows, threads_for(cols), 0, stream_of(ctx)>>>(
        static_cast<float*>(out->data), static_cast<const float*>(x->data),
        weight->data, weight->kind, cols, eps);
    return finish();
}

extern "C" int gemma4_rtx5090_attention(SpiteTensor* out, const SpiteTensor* x,
                                        const SpiteTensor* wq, const SpiteTensor* wk,
                                        const SpiteTensor* wv, const SpiteTensor* wo,
                                        const SpiteTensor* q_norm, const SpiteTensor* k_norm,
                                        float norm_eps, SpiteKvCache* kv, float rope_freq_base,
                                        const SpiteCtx* ctx) {
    return kvflash_run(out, x, wq, wk, wv, wo, q_norm, k_norm, norm_eps, kv, rope_freq_base, ctx);
}

extern "C" int gemma4_rtx5090_ffn(SpiteTensor* out, const SpiteTensor* x,
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

extern "C" int gemma4_rtx5090_matmul(SpiteTensor* out, const SpiteTensor* x,
                                     const SpiteTensor* w, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (launch_matvec(w, static_cast<const float*>(x->data),
                      static_cast<float*>(out->data), false, stream_of(ctx)))
        return -1;
    return finish();
}

// ── Kernel descriptor ─────────────────────────────────────────────────────

/* KV-cache tiers this attention op reads and writes (bit = SpiteType). */
extern "C" uint64_t gemma4_rtx5090_kv_cache_kinds() {
    return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16) | (1ull << SPITE_TYPE_Q8_0) |
           (1ull << SPITE_TYPE_Q5_1) | (1ull << SPITE_TYPE_Q4_0);
}

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "gemma4",
    "sm_120",
    "spite project (Gemma 4 RTX 5090: 192-SM grid, 8-warp K-split Q8_0 matvec, smem x-staging)",
    {SPITE_TYPE_F16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q4_K, SPITE_TYPE_Q5_K, SPITE_TYPE_Q6_K,
     SPITE_TYPE_Q4_0, SPITE_TYPE_Q3_K, 0},
    gemma4_rtx5090_rms_norm,
    gemma4_rtx5090_attention,
    nullptr, /* mla */
    gemma4_rtx5090_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    gemma4_rtx5090_matmul,
    gemma4_rtx5090_kv_cache_kinds,
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
