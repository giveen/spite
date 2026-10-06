/*
 * kernels/gemma/gemma4/nvidia/sm_120/kernel.cu
 *
 * Blackwell (sm_120) architecture kernel for Gemma 4.
 * Scope: all sm_120-family GPUs (RTX 5090, 5080, 5070, 5060, …).
 *
 * Improvements over kernels/gemma/gemma4/nvidia/ (generic CUDA baseline):
 *   • Vectorized 128-bit memory transactions — float4 in rms_norm,
 *     F32 dense matvec, and the whole flash attention tile (staging, Q.K
 *     dot-products, P.V accumulation), plus __half2 F16 staging.
 *   • KV-tier-specialized attention staging: q8_0/q5_1/q4_0 block decoders are
 *     compiled in rather than switched on per element (see
 *     kv_attn_flash_sm120.inl).
 *   • __half2 pair loads + arithmetic in the F16 dense matvec path.
 *   • Fully unrolled warp reductions (5 explicit shuffles, no loop)
 *     exploiting Blackwell's dual-warp issue scheduler.
 *   • Pinned exact RoPE tables in __constant__ memory.
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

constexpr int WARP = 32;
constexpr int ROWS_PER_BLOCK = 4;

inline cudaStream_t stream_of(const SpiteCtx* ctx) {
    return ctx ? static_cast<cudaStream_t>(ctx->gpu_stream) : nullptr;
}

/* Fully unrolled warp reduction — Blackwell dual-issue scheduler fires both
 * halves of the shuffle tree without stalls when the loop is removed. */
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

// ── Exact bitwise RoPE inverse frequencies in __constant__ memory ─────────
// Pinned Transformers 5.12 Gemma 4 values (eliminates powf transcendental evaluation).
static __device__ __constant__ uint32_t kLocalBits[128] = {
    0x3f800000U, 0x3f6e39f8U, 0x3f5dafd7U, 0x3f4e4badU, 0x3f3ff911U,
    0x3f32a506U, 0x3f263de0U, 0x3f1ab32bU, 0x3f0ff59aU, 0x3f05f6efU,
    0x3ef953cfU, 0x3ee8045fU, 0x3ed7e89bU, 0x3ec8eb24U, 0x3ebaf81bU,
    0x3eadfcffU, 0x3ea1e89bU, 0x3e96aaeaU, 0x3e8c3504U, 0x3e827909U,
    0x3e72d423U, 0x3e61f835U, 0x3e5247edU, 0x3e43ae7cU, 0x3e361887U,
    0x3e297409U, 0x3e1db040U, 0x3e12bd91U, 0x3e088d77U, 0x3dfe24e0U,
    0x3dec7fd6U, 0x3ddc1466U, 0x3dcccccdU, 0x3dbe94c6U, 0x3db15978U,
    0x3da50956U, 0x3d99940dU, 0x3d8eea6bU, 0x3d84fe4dU, 0x3d778513U,
    0x3d6655c2U, 0x3d5657e4U, 0x3d47763fU, 0x3d399d19U, 0x3d2cba15U,
    0x3d20bc1dU, 0x3d159348U, 0x3d0b30ccU, 0x3d0186e3U, 0x3cf11177U,
    0x3ce054d2U, 0x3cd0c1a8U, 0x3cc2434fU, 0x3cb4c691U, 0x3ca8398bU,
    0x3c9c8b97U, 0x3c91ad39U, 0x3c879008U, 0x3c7c4d33U, 0x3c6ac8e7U,
    0x3c5a7bf2U, 0x3c4b50b3U, 0x3c3d3311U, 0x3c301052U, 0x3c23d70aU,
    0x3c187705U, 0x3c0de12dU, 0x3c040779U, 0x3bf5b9b0U, 0x3be4aa46U,
    0x3bd4ca15U, 0x3bc6040fU, 0x3bb8449cU, 0x3bab7983U, 0x3b9f91ccU,
    0x3b947daeU, 0x3b8a2e77U, 0x3b80967dU, 0x3b6f520eU, 0x3b5eb47aU,
    0x3b4f3e38U, 0x3b40dac5U, 0x3b33770fU, 0x3b270153U, 0x3b1b690dU,
    0x3b109edbU, 0x3b06946fU, 0x3afa78f0U, 0x3ae91528U, 0x3ad8e673U,
    0x3ac9d75cU, 0x3abbd3edU, 0x3aaec98eU, 0x3aa2a6f7U, 0x3a975c0eU,
    0x3a8cd9dbU, 0x3a83126fU, 0x3a73f1a3U, 0x3a6301e2U, 0x3a533f28U,
    0x3a44948cU, 0x3a36ee9eU, 0x3a2a3b44U, 0x3a1e69a5U, 0x3a136a16U,
    0x3a092e02U, 0x39ff4facU, 0x39ed95e2U, 0x39dd1725U, 0x39cdbd96U,
    0x39bf74d7U, 0x39b229fbU, 0x39a5cb60U, 0x399a489eU, 0x398f9272U,
    0x39859aa9U, 0x3978a815U, 0x39676492U, 0x395753e4U, 0x394860c1U,
    0x393a7753U, 0x392d8529U, 0x39217916U, 0x39164324U, 0x390bd472U,
    0x39021f2bU, 0x38f22ce2U, 0x38e15c91U,
};

// ── Matvec ─────────────────────────────────────────────────────────────────

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

/* F32 dense matvec: 128-bit float4 loads cut memory transactions by 4×. */
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

/* F16 dense matvec: __half2 weight loads + float2 activation loads halve load instructions. */
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
    const dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    switch (w->kind) {
    case SPITE_TYPE_Q8_0:
        if (cols % QK8_0) return -1;
        matvec_q8_0<<<grid, block, 0, s>>>(static_cast<const BlockQ8_0*>(w->data), x, y, rows,
                                           cols, accumulate);
        break;
    case SPITE_TYPE_F32:
        matvec_f32<<<grid, block, 0, s>>>(static_cast<const float*>(w->data), x, y, rows,
                                          cols, accumulate);
        break;
    case SPITE_TYPE_F16:
        matvec_f16<<<grid, block, 0, s>>>(static_cast<const __half*>(w->data), x, y, rows,
                                          cols, accumulate);
        break;
    default:
        return sq::gemv(w, x, y, accumulate, s);
    }
    return 0;
}

// ── RMSNorm: vectorized float4 (F32 weight) or half2+float2 (F16 weight) ──

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

extern "C" int gemma4_sm120_rms_norm(SpiteTensor* out, const SpiteTensor* x,
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

extern "C" int gemma4_sm120_attention(SpiteTensor* out, const SpiteTensor* x,
                                      const SpiteTensor* wq, const SpiteTensor* wk,
                                      const SpiteTensor* wv, const SpiteTensor* wo,
                                      const SpiteTensor* q_norm, const SpiteTensor* k_norm,
                                      float norm_eps, SpiteKvCache* kv, float rope_freq_base,
                                      const SpiteCtx* ctx) {
    return kvflash_run(out, x, wq, wk, wv, wo, q_norm, k_norm, norm_eps, kv, rope_freq_base, ctx);
}

extern "C" int gemma4_sm120_ffn(SpiteTensor* out, const SpiteTensor* x,
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

extern "C" int gemma4_sm120_matmul(SpiteTensor* out, const SpiteTensor* x,
                                   const SpiteTensor* w, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (launch_matvec(w, static_cast<const float*>(x->data),
                      static_cast<float*>(out->data), false, stream_of(ctx)))
        return -1;
    return finish();
}

/* Fused Gemma 4 MTP stem (Blackwell sm_120): float4 128-bit vectorized stem packing. */
__global__ void gemma4_sm120_mtp_stem_kernel(float* __restrict__ out,
                                             const float* __restrict__ embed,
                                             const float* __restrict__ hidden,
                                             const void* __restrict__ w_enorm, int enorm_kind,
                                             const void* __restrict__ w_hnorm, int hnorm_kind,
                                             int d, float emb_scale, float eps) {
    const int tok = blockIdx.y;
    const float* e_tok = embed + static_cast<size_t>(tok) * d;
    const float* h_tok = hidden + static_cast<size_t>(tok) * d;
    float* out_tok = out + static_cast<size_t>(tok) * (2 * d);

    const int d4 = d / 4;
    const float4* e4 = reinterpret_cast<const float4*>(e_tok);
    const float4* h4 = reinterpret_cast<const float4*>(h_tok);
    float4* o_e4 = reinterpret_cast<float4*>(out_tok);
    float4* o_h4 = reinterpret_cast<float4*>(out_tok + d);

    for (int i = threadIdx.x; i < d4; i += blockDim.x) {
        float4 ev = e4[i];
        ev.x *= emb_scale; ev.y *= emb_scale; ev.z *= emb_scale; ev.w *= emb_scale;
        if (w_enorm) {
            if (enorm_kind == SPITE_TYPE_F32) {
                float4 nw = reinterpret_cast<const float4*>(w_enorm)[i];
                ev.x *= nw.x; ev.y *= nw.y; ev.z *= nw.z; ev.w *= nw.w;
            } else if (enorm_kind == SPITE_TYPE_F16) {
                const __half* hw = static_cast<const __half*>(w_enorm) + 4 * i;
                ev.x *= __half2float(hw[0]); ev.y *= __half2float(hw[1]);
                ev.z *= __half2float(hw[2]); ev.w *= __half2float(hw[3]);
            }
        }
        o_e4[i] = ev;

        float4 hv = h4[i];
        if (w_hnorm) {
            if (hnorm_kind == SPITE_TYPE_F32) {
                float4 nw = reinterpret_cast<const float4*>(w_hnorm)[i];
                hv.x *= nw.x; hv.y *= nw.y; hv.z *= nw.z; hv.w *= nw.w;
            } else if (hnorm_kind == SPITE_TYPE_F16) {
                const __half* hw = static_cast<const __half*>(w_hnorm) + 4 * i;
                hv.x *= __half2float(hw[0]); hv.y *= __half2float(hw[1]);
                hv.z *= __half2float(hw[2]); hv.w *= __half2float(hw[3]);
            }
        }
        o_h4[i] = hv;
    }

    for (int i = d4 * 4 + threadIdx.x; i < d; i += blockDim.x) {
        float ev = e_tok[i] * emb_scale;
        if (w_enorm) ev *= load_w(w_enorm, enorm_kind, i);
        out_tok[i] = ev;

        float hv = h_tok[i];
        if (w_hnorm) hv *= load_w(w_hnorm, hnorm_kind, i);
        out_tok[d + i] = hv;
    }
}

extern "C" int gemma4_sm120_mtp_stem(SpiteTensor* out, const SpiteTensor* embed,
                                     const SpiteTensor* hidden,
                                     const SpiteTensor* w_enorm,
                                     const SpiteTensor* w_hnorm, float eps,
                                     const SpiteCtx* ctx) {
    if (!out || !embed || !hidden || !out->data || !embed->data || !hidden->data) return -1;
    if (embed->kind != SPITE_TYPE_F32 || hidden->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;

    const int64_t d = embed->ne[0];
    const int64_t t = embed->ne[1] ? embed->ne[1] : 1;
    if (d < 1 || hidden->ne[0] != d || out->ne[0] != 2 * d) return -1;

    const float emb_scale = sqrtf(static_cast<float>(d));
    int threads = static_cast<int>(((d / 4 + 31) / 32) * 32);
    threads = threads < 32 ? 32 : (threads > 1024 ? 1024 : threads);
    dim3 grid(1, static_cast<unsigned>(t));

    gemma4_sm120_mtp_stem_kernel<<<grid, threads, 0, stream_of(ctx)>>>(
        static_cast<float*>(out->data), static_cast<const float*>(embed->data),
        static_cast<const float*>(hidden->data),
        w_enorm ? w_enorm->data : nullptr, w_enorm ? w_enorm->kind : SPITE_TYPE_F32,
        w_hnorm ? w_hnorm->data : nullptr, w_hnorm ? w_hnorm->kind : SPITE_TYPE_F32,
        static_cast<int>(d), emb_scale, eps);

    return finish();
}

// ── Kernel descriptor ─────────────────────────────────────────────────────

/* KV-cache tiers this attention op reads and writes (bit = SpiteType). */
extern "C" uint64_t gemma4_sm120_kv_cache_kinds() {
    return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16) | (1ull << SPITE_TYPE_Q8_0) |
           (1ull << SPITE_TYPE_Q5_1) | (1ull << SPITE_TYPE_Q4_0);
}

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "gemma4",
    "sm_120",
    "spite project (Gemma 4 Blackwell sm_120: float4 + half2 vectorized)",
    {SPITE_TYPE_F16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q4_K, SPITE_TYPE_Q5_K, SPITE_TYPE_Q6_K,
     SPITE_TYPE_Q4_0, SPITE_TYPE_Q3_K, 0},
    gemma4_sm120_rms_norm,
    gemma4_sm120_attention,
    nullptr, /* mla */
    gemma4_sm120_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    gemma4_sm120_matmul,
    gemma4_sm120_kv_cache_kinds,
    nullptr, /* linear_attn */
    nullptr, /* attention_ex */
    gemma4_sm120_mtp_stem,
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
