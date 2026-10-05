/*
 * kernels/qwen/qwen3/nvidia/sm_120/kernel.cu
 *
 * Blackwell (sm_120) architecture kernel for Qwen3 dense decoders.
 * Scope: all sm_120-family GPUs (RTX 5090, 5080, 5070, 5060, …).
 *
 * Improvements over kernels/qwen/qwen3/nvidia/ (generic CUDA baseline):
 *   • Vectorized 128-bit memory transactions — float4 in rms_norm,
 *     F32 dense matvec, and attention score dot-products.
 *   • __half2 pair loads + arithmetic in the F16 dense matvec path.
 *   • Fully unrolled warp reductions (5 explicit shuffles, no loop)
 *     exploiting Blackwell's dual-warp issue scheduler.
 *
 * Weight types: F32, F16, Q8_0. Activations and KV cache: F32.
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

/* F16 dense matvec: __half2 weight loads + float2 activation loads halve
 * load instructions versus element-by-element __half access. */
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
        return -1;
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

// ── Attention pieces ──────────────────────────────────────────────────────

/* dst may alias src (in-place Q); no __restrict__ on those two. */
__global__ void qk_norm_rope(float* dst, const float* src, const void* __restrict__ nw,
                             int nkind, float eps, int hd, int pos, float theta) {
    const float* s = src + static_cast<size_t>(blockIdx.x) * hd;
    float* d = dst + static_cast<size_t>(blockIdx.x) * hd;
    const int half = hd / 2;
    float scale = 1.0f;
    if (nw) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += s[i] * s[i];
        ss = block_sum(ss);
        scale = rsqrtf(ss / hd + eps);
    }
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float x0 = s[i] * scale, x1 = s[i + half] * scale;
        if (nw) {
            x0 *= load_w(nw, nkind, i);
            x1 *= load_w(nw, nkind, i + half);
        }
        const float freq = powf(theta, -2.0f * i / hd);
        float sn, cs;
        sincosf(pos * freq, &sn, &cs);
        d[i]        = x0 * cs - x1 * sn;
        d[i + half] = x0 * sn + x1 * cs;
    }
}

/* Vectorized dot-product: float4 loads when hd % 4 == 0. */
__global__ void attn_scores(float* __restrict__ scores, const float* __restrict__ q,
                            const float* __restrict__ kc, int n_tok, int n_ctx, int hd,
                            int group, int kv_stride, float scale) {
    const int h = blockIdx.y;
    const int t = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (t >= n_tok) return;
    const float* qh = q  + static_cast<size_t>(h) * hd;
    const float* kt = kc + static_cast<size_t>(t) * kv_stride
                         + static_cast<size_t>(h / group) * hd;
    float acc = 0.0f;
    if (hd % 4 == 0) {
        const float4* qh4 = reinterpret_cast<const float4*>(qh);
        const float4* kt4 = reinterpret_cast<const float4*>(kt);
        const int hd4 = hd / 4;
        for (int i = threadIdx.x; i < hd4; i += WARP) {
            float4 q4 = qh4[i], k4 = kt4[i];
            acc += q4.x*k4.x + q4.y*k4.y + q4.z*k4.z + q4.w*k4.w;
        }
    } else {
        for (int i = threadIdx.x; i < hd; i += WARP) acc += qh[i] * kt[i];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) scores[static_cast<size_t>(h) * n_ctx + t] = acc * scale;
}

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

__global__ void attn_weighted_v(float* __restrict__ att, const float* __restrict__ scores,
                                const float* __restrict__ vc, int n_tok, int n_ctx, int hd,
                                int group, int kv_stride) {
    const int h = blockIdx.x;
    const float* p  = scores + static_cast<size_t>(h) * n_ctx;
    const float* vb = vc     + static_cast<size_t>(h / group) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_tok; ++t)
            acc += p[t] * vb[static_cast<size_t>(t) * kv_stride + i];
        att[static_cast<size_t>(h) * hd + i] = acc;
    }
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

// ── ABI ops ───────────────────────────────────────────────────────────────

extern "C" int qwen3_sm120_rms_norm(SpiteTensor* out, const SpiteTensor* x,
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

extern "C" int qwen3_sm120_attention(SpiteTensor* out, const SpiteTensor* x,
                                     const SpiteTensor* wq, const SpiteTensor* wk,
                                     const SpiteTensor* wv, const SpiteTensor* wo,
                                     const SpiteTensor* q_norm, const SpiteTensor* k_norm,
                                     float norm_eps, SpiteKvCache* kv, float rope_freq_base,
                                     const SpiteCtx* ctx) {
    if (!ctx || !kv || ctx->n_heads <= 0 || ctx->n_kv_heads <= 0) return -1;
    if (kv->k.kind != SPITE_TYPE_F32 || kv->v.kind != SPITE_TYPE_F32) return -1;
    const int nh = ctx->n_heads, nkv = ctx->n_kv_heads;
    const int hd = static_cast<int>(wq->ne[1]) / nh;
    const int kv_stride = nkv * hd;
    const int n_ctx = static_cast<int>(kv->k.ne[1]);
    const int pos = ctx->pos;
    if (hd <= 0 || hd % 2 || nh % nkv || static_cast<int>(kv->k.ne[0]) != kv_stride) return -1;
    if (pos < 0 || pos >= n_ctx) return -2;

    const size_t need = sizeof(float) * (static_cast<size_t>(nh) * hd * 2 +
                                         static_cast<size_t>(kv_stride) +
                                         static_cast<size_t>(nh) * n_ctx);
    if (!ctx->scratchpad || ctx->scratchpad_bytes < need) return -2;
    float* q      = static_cast<float*>(ctx->scratchpad);
    float* k      = q + static_cast<size_t>(nh) * hd;
    float* att    = k + kv_stride;
    float* scores = att + static_cast<size_t>(nh) * hd;

    cudaStream_t s = stream_of(ctx);
    const float* xin  = static_cast<const float*>(x->data);
    float* kc = static_cast<float*>(kv->k.data);
    float* vc = static_cast<float*>(kv->v.data);
    float* k_row = kc + static_cast<size_t>(pos) * kv_stride;
    float* v_row = vc + static_cast<size_t>(pos) * kv_stride;

    if (launch_matvec(wq, xin, q,     false, s) ||
        launch_matvec(wk, xin, k,     false, s) ||
        launch_matvec(wv, xin, v_row, false, s))
        return -1;

    const int rt = threads_for(hd / 2);
    qk_norm_rope<<<nh,  rt, 0, s>>>(q,     q, q_norm ? q_norm->data : nullptr,
                                    q_norm ? q_norm->kind : 0, norm_eps, hd, pos, rope_freq_base);
    qk_norm_rope<<<nkv, rt, 0, s>>>(k_row, k, k_norm ? k_norm->data : nullptr,
                                    k_norm ? k_norm->kind : 0, norm_eps, hd, pos, rope_freq_base);

    const int n_tok = pos + 1;
    const int group = nh / nkv;
    const float scale = rsqrtf(static_cast<float>(hd));
    attn_scores<<<dim3((n_tok + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, nh),
                  dim3(WARP, ROWS_PER_BLOCK), 0, s>>>(
        scores, q, kc, n_tok, n_ctx, hd, group, kv_stride, scale);
    attn_softmax<<<nh, threads_for(n_tok > 256 ? 256 : n_tok), 0, s>>>(scores, n_tok, n_ctx);
    attn_weighted_v<<<nh, threads_for(hd), 0, s>>>(att, scores, vc, n_tok, n_ctx, hd, group,
                                                   kv_stride);

    if (launch_matvec(wo, att, static_cast<float*>(out->data), true, s)) return -1;
    return finish();
}

extern "C" int qwen3_sm120_ffn(SpiteTensor* out, const SpiteTensor* x,
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

extern "C" int qwen3_sm120_matmul(SpiteTensor* out, const SpiteTensor* x,
                                  const SpiteTensor* w, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (launch_matvec(w, static_cast<const float*>(x->data),
                      static_cast<float*>(out->data), false, stream_of(ctx)))
        return -1;
    return finish();
}

// ── Kernel descriptor ─────────────────────────────────────────────────────

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen3",
    "sm_120",
    "spite project (Blackwell sm_120: float4 + half2 vectorized)",
    {SPITE_TYPE_F32, SPITE_TYPE_F16, SPITE_TYPE_Q8_0, 0, 0, 0, 0, 0},
    qwen3_sm120_rms_norm,
    qwen3_sm120_attention,
    nullptr, /* mla */
    qwen3_sm120_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    qwen3_sm120_matmul,
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
