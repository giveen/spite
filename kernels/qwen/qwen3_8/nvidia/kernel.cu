/*
 * kernels/qwen/qwen3_8/nvidia/kernel.cu
 *
 * Generic CUDA kernel for Qwen3.8-27B dense decoders on all NVIDIA/CUDA
 * architectures (sm_60 Pascal through sm_120 Blackwell and beyond).
 *
 * Every SpiteTensor.data is a DEVICE pointer; all work is issued on
 * ctx->gpu_stream (null = default stream).
 *
 *   rms_norm  — row-wise RMSNorm (also used for per-head QK norm)
 *   attention — Q/K/V matvec → NEOX RoPE → KV write → flash GQA → out-proj
 *   ffn       — SwiGLU, out += down(silu(gate) * up)
 *   matmul    — out = W · x (LM head)
 *
 * Architecture-specific overrides live in sm_60/ (Pascal __half2 SIMD,
 * tuned tiles) and sm_60/tesla_p100/ (P100 tile sizes, NVLink TP helpers).
 *
 * Weight types: F32/F16/Q8_0 handled natively; all 28 SpiteTypes via the
 * shared GEMV in core/gpu/quant_gemv.h. Activations: F32 end-to-end.
 *
 * Scratchpad layout (ctx->scratchpad, device floats):
 *   attention: q[nh*hd] k[nkv*hd] v[nkv*hd] acc[chunks*nh*hd] m[chunks*nh] l[chunks*nh]
 *   ffn:       gate[d_ffn] up[d_ffn]
 */

#include "core/abi.h"
#include "core/gpu/quant_gemv.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

struct BlockQ8_0 {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must be 34 bytes");

constexpr int WARP          = 32;
constexpr int ROWS_PER_BLOCK = 4;

inline cudaStream_t stream_of(const SpiteCtx* ctx) {
    return ctx ? static_cast<cudaStream_t>(ctx->gpu_stream) : nullptr;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
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
    return red[0];
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
    return red[0];
}

__device__ __forceinline__ float load_w(const void* w, int kind, int i) {
    if (kind == SPITE_TYPE_F16) return __half2float(static_cast<const __half*>(w)[i]);
    return static_cast<const float*>(w)[i];
}

/* ── Matvec ──────────────────────────────────────────────────────────────── */

__global__ void matvec_q8_0(const BlockQ8_0* __restrict__ w, const float* __restrict__ x,
                             float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int nb = cols / 32;
    const BlockQ8_0* wr = w + static_cast<size_t>(row) * nb;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float d = __half2float(wr[b].d);
        acc += d * static_cast<float>(wr[b].qs[threadIdx.x]) * x[b * 32 + threadIdx.x];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }

template <typename T>
__global__ void matvec_dense(const T* __restrict__ w, const float* __restrict__ x,
                              float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const T* wr = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < cols; c += WARP) acc += to_f(wr[c]) * x[c];
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
        if (cols % 32) return -1;
        matvec_q8_0<<<grid, block, 0, s>>>(
            static_cast<const BlockQ8_0*>(w->data), x, y, rows, cols, accumulate);
        break;
    case SPITE_TYPE_F32:
        matvec_dense<float><<<grid, block, 0, s>>>(
            static_cast<const float*>(w->data), x, y, rows, cols, accumulate);
        break;
    case SPITE_TYPE_F16:
        matvec_dense<__half><<<grid, block, 0, s>>>(
            static_cast<const __half*>(w->data), x, y, rows, cols, accumulate);
        break;
    default:
        return sq::gemv(w, x, y, accumulate, s);
    }
    return 0;
}

/* ── RMSNorm ─────────────────────────────────────────────────────────────── */

__global__ void rms_norm_rows(float* __restrict__ out, const float* __restrict__ x,
                               const void* __restrict__ w, int wkind, int cols, float eps) {
    const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
    float* orow = out + static_cast<size_t>(blockIdx.x) * cols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float scale = rsqrtf(ss / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        orow[i] = xr[i] * scale * load_w(w, wkind, i);
}

/* ── Attention softmax ───────────────────────────────────────────────────── */

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

/* ── FFN activation ─────────────────────────────────────────────────────── */

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

inline int threads_for(int n) {
    int t = ((n + WARP - 1) / WARP) * WARP;
    return t < WARP ? WARP : (t > 1024 ? 1024 : t);
}

inline int finish() { return cudaGetLastError() == cudaSuccess ? 0 : -2; }

}  // namespace

#include "kernels/qwen/qwen3_8/nvidia/kv_attn.inl"
#include "kernels/qwen/qwen3_8/nvidia/kv_attn_flash.inl"

/* ── ABI entry points ────────────────────────────────────────────────────── */

extern "C" int qwen38_cuda_rms_norm(SpiteTensor* out, const SpiteTensor* x,
                                    const SpiteTensor* weight, float eps,
                                    const SpiteCtx* ctx) {
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

extern "C" int qwen38_cuda_attention(SpiteTensor* out, const SpiteTensor* x,
                                     const SpiteTensor* wq, const SpiteTensor* wk,
                                     const SpiteTensor* wv, const SpiteTensor* wo,
                                     const SpiteTensor* q_norm, const SpiteTensor* k_norm,
                                     float norm_eps, SpiteKvCache* kv,
                                     float rope_freq_base, const SpiteCtx* ctx) {
    return kvflash_run(out, x, wq, wk, wv, wo, q_norm, k_norm,
                       norm_eps, kv, rope_freq_base, ctx);
}

extern "C" int qwen38_cuda_ffn(SpiteTensor* out, const SpiteTensor* x,
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
    if (launch_matvec(w_gate, xin, gate, false, s)) return -1;
    if (launch_matvec(w_up,   xin, up,   false, s)) return -1;
    glu_act<<<(d_ffn + 255) / 256, 256, 0, s>>>(gate, up, d_ffn, act == SPITE_FFN_GELU_GATE);
    if (launch_matvec(w_down, gate, static_cast<float*>(out->data), true, s)) return -1;
    return finish();
}

extern "C" int qwen38_cuda_matmul(SpiteTensor* out, const SpiteTensor* x,
                                   const SpiteTensor* w, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (launch_matvec(w, static_cast<const float*>(x->data),
                      static_cast<float*>(out->data), false, stream_of(ctx))) return -1;
    return finish();
}

extern "C" uint64_t qwen38_cuda_kv_cache_kinds() {
    return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16) |
           (1ull << SPITE_TYPE_Q8_0) | (1ull << SPITE_TYPE_Q5_1) |
           (1ull << SPITE_TYPE_Q4_0);
}

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen38",
    "cuda",
    "spite project (qwen3.8 generic nvidia/cuda path)",
    {SPITE_TYPE_F16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q4_K, SPITE_TYPE_Q5_K,
     SPITE_TYPE_Q6_K, SPITE_TYPE_Q4_0, SPITE_TYPE_Q3_K, 0},
    qwen38_cuda_rms_norm,
    qwen38_cuda_attention,
    nullptr, /* mla */
    qwen38_cuda_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    qwen38_cuda_matmul,
    qwen38_cuda_kv_cache_kinds,
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
