/*
 * kernels/qwen/qwen3/nvidia/sm_120/rtx_5090/kernel.cu
 *
 * Card-specialized CUDA kernel for Qwen3 dense decoders tuned specifically
 * for NVIDIA GeForce RTX 5090 (Blackwell sm_120, 192 SMs, 96 MB L2, 128 KB SM shmem).
 *
 * Physical hardware tuning:
 *   - Fused Gate-Up FFN with integrated SwiGLU/GeGLU activation:
 *     Eliminates 72 kernel launches per token across 36 decoder layers,
 *     halves memory bandwidth for input activation reads, and completely eliminates
 *     intermediate VRAM roundtrips for the 12,288-dim gate and up tensors.
 *   - 4-warp threadblocks with fully unrolled warp-level fma reductions.
 *   - Vectorized 128-bit float4 loads for RMSNorm and attention dot-products.
 *
 * Weight types: F32, F16, Q8_0. Activations and KV cache: F32.
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

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
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

// ── Matvec Q8_0 ──────────────────────────────────────────────────────────

__global__ void matvec_q8_0(const BlockQ8_0* __restrict__ w, const float* __restrict__ x,
                            float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int nb = cols / QK8_0;
    const BlockQ8_0* wr = w + static_cast<size_t>(row) * nb;
    float acc = 0.0f;
    #pragma unroll 4
    for (int b = 0; b < nb; ++b) {
        const float d = __half2float(wr[b].d);
        acc = fmaf(d * static_cast<float>(wr[b].qs[threadIdx.x]), x[b * QK8_0 + threadIdx.x], acc);
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

// ── Fused Dual Gate-Up Matvec + SwiGLU Activation for RTX 5090 ───────────

__global__ void matvec_q8_0_fused_gate_up(
    const BlockQ8_0* __restrict__ w_gate,
    const BlockQ8_0* __restrict__ w_up,
    const float* __restrict__ x,
    float* __restrict__ gate_out,
    int rows, int cols, int gelu)
{
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int nb = cols / QK8_0;
    const BlockQ8_0* wr_gate = w_gate + static_cast<size_t>(row) * nb;
    const BlockQ8_0* wr_up = w_up + static_cast<size_t>(row) * nb;
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    #pragma unroll 4
    for (int b = 0; b < nb; ++b) {
        const float x_val = x[b * QK8_0 + threadIdx.x];
        const float dg = __half2float(wr_gate[b].d);
        const float du = __half2float(wr_up[b].d);
        acc_gate = fmaf(dg * static_cast<float>(wr_gate[b].qs[threadIdx.x]), x_val, acc_gate);
        acc_up = fmaf(du * static_cast<float>(wr_up[b].qs[threadIdx.x]), x_val, acc_up);
    }
    acc_gate = warp_sum(acc_gate);
    acc_up = warp_sum(acc_up);
    if (threadIdx.x == 0) {
        const float a = gelu ? 0.5f * acc_gate * (1.0f + tanhf(0.7978846f * (acc_gate + 0.044715f * acc_gate * acc_gate * acc_gate)))
                             : acc_gate / (1.0f + __expf(-acc_gate));
        gate_out[row] = a * acc_up;
    }
}

// ── Dense Matvec ─────────────────────────────────────────────────────────

__global__ void matvec_dense_f32_vec4(const float* __restrict__ w, const float* __restrict__ x,
                                      float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const float4* wr4 = reinterpret_cast<const float4*>(w + static_cast<size_t>(row) * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int cols4 = cols / 4;
    float acc = 0.0f;
    #pragma unroll 4
    for (int c = threadIdx.x; c < cols4; c += WARP) {
        float4 wv = wr4[c];
        float4 xv = x4[c];
        acc = fmaf(wv.x, xv.x, acc);
        acc = fmaf(wv.y, xv.y, acc);
        acc = fmaf(wv.z, xv.z, acc);
        acc = fmaf(wv.w, xv.w, acc);
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

__global__ void matvec_dense_f16(const __half* __restrict__ w, const float* __restrict__ x,
                                 float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const __half* wr = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    #pragma unroll 4
    for (int c = threadIdx.x; c < cols; c += WARP) {
        acc = fmaf(__half2float(wr[c]), x[c], acc);
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
        if ((cols % 4) == 0 && (reinterpret_cast<uintptr_t>(w->data) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(x) % 16 == 0)) {
            matvec_dense_f32_vec4<<<grid, block, 0, s>>>(static_cast<const float*>(w->data), x, y,
                                                         rows, cols, accumulate);
        } else {
            matvec_dense_f16<<<grid, block, 0, s>>>(reinterpret_cast<const __half*>(w->data), x, y,
                                                    rows, cols, accumulate);
        }
        break;
    case SPITE_TYPE_F16:
        matvec_dense_f16<<<grid, block, 0, s>>>(static_cast<const __half*>(w->data), x, y,
                                                rows, cols, accumulate);
        break;
    default:
        return -1;
    }
    return 0;
}

// ── Vectorized RMSNorm ───────────────────────────────────────────────────

__global__ void rms_norm_rows_vec4(float* __restrict__ out, const float* __restrict__ x,
                                   const void* __restrict__ w, int wkind, int cols, float eps) {
    const size_t row_offset = static_cast<size_t>(blockIdx.x) * cols;
    const float* xr = x + row_offset;
    float* orow = out + row_offset;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    const int cols4 = cols / 4;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
        float4 v = xr4[i];
        ss = fmaf(v.x, v.x, ss);
        ss = fmaf(v.y, v.y, ss);
        ss = fmaf(v.z, v.z, ss);
        ss = fmaf(v.w, v.w, ss);
    }
    ss = block_sum(ss);
    const float scale = rsqrtf(ss / cols + eps);
    for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
        float4 v = xr4[i];
        const int idx = i * 4;
        v.x *= scale * load_w(w, wkind, idx);
        v.y *= scale * load_w(w, wkind, idx + 1);
        v.z *= scale * load_w(w, wkind, idx + 2);
        v.w *= scale * load_w(w, wkind, idx + 3);
        reinterpret_cast<float4*>(orow)[i] = v;
    }
}

// ── Attention pieces ─────────────────────────────────────────────────────
//
// The F32-only attention kernels that used to live here (qk_norm_rope,
// attn_scores_vec4, attn_weighted_v) are superseded by the shared VBR kernels
// in ../kv_attn.inl; only the softmax is still shared with that file.

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

inline int finish() {
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace

// Shared VBR (quantized-KV) attention. Uses the file's launch_matvec,
// threads_for, stream_of, attn_softmax, block_sum and load_w.
#include "kernels/qwen/qwen3/nvidia/kv_attn.inl"

// ── ABI ops ──────────────────────────────────────────────────────────────

extern "C" int qwen3_cuda_rms_norm(SpiteTensor* out, const SpiteTensor* x,
                                   const SpiteTensor* weight, float eps, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    const int cols = static_cast<int>(x->ne[0]);
    const int rows = static_cast<int>(x->ne[1] ? x->ne[1] : 1);
    if (weight->ne[0] != x->ne[0]) return -1;
    if (cols % 4 == 0) {
        rms_norm_rows_vec4<<<rows, threads_for(cols / 4), 0, stream_of(ctx)>>>(
            static_cast<float*>(out->data), static_cast<const float*>(x->data), weight->data,
            weight->kind, cols, eps);
    }
    return finish();
}

extern "C" int qwen3_cuda_attention(SpiteTensor* out, const SpiteTensor* x,
                                    const SpiteTensor* wq, const SpiteTensor* wk,
                                    const SpiteTensor* wv, const SpiteTensor* wo,
                                    const SpiteTensor* q_norm, const SpiteTensor* k_norm,
                                    float norm_eps, SpiteKvCache* kv, float rope_freq_base,
                                    const SpiteCtx* ctx) {
    return kvattn_run(out, x, wq, wk, wv, wo, q_norm, k_norm, norm_eps, kv, rope_freq_base, ctx);
}

extern "C" int qwen3_cuda_ffn(SpiteTensor* out, const SpiteTensor* x, const SpiteTensor* w_gate,
                              const SpiteTensor* w_up, const SpiteTensor* w_down,
                              SpiteFfnActivation act, const SpiteCtx* ctx) {
    if (act != SPITE_FFN_SILU_GATE && act != SPITE_FFN_GELU_GATE) return -1;
    if (!ctx) return -1;
    const int d_ffn = static_cast<int>(w_gate->ne[1]);
    const int cols = static_cast<int>(w_gate->ne[0]);
    cudaStream_t s = stream_of(ctx);
    const float* xin = static_cast<const float*>(x->data);

    // If Q8_0 weights and scratchpad available, run RTX 5090 fused dual gate-up kernel
    if (w_gate->kind == SPITE_TYPE_Q8_0 && w_up->kind == SPITE_TYPE_Q8_0 && ctx->scratchpad &&
        ctx->scratchpad_bytes >= sizeof(float) * static_cast<size_t>(d_ffn)) {
        float* gate = static_cast<float*>(ctx->scratchpad);
        const dim3 block(WARP, ROWS_PER_BLOCK);
        const dim3 grid((d_ffn + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        matvec_q8_0_fused_gate_up<<<grid, block, 0, s>>>(
            static_cast<const BlockQ8_0*>(w_gate->data),
            static_cast<const BlockQ8_0*>(w_up->data),
            xin, gate, d_ffn, cols, act == SPITE_FFN_GELU_GATE);
        if (launch_matvec(w_down, gate, static_cast<float*>(out->data), true, s)) return -1;
        return finish();
    }

    // Fallback for non-Q8_0
    const size_t need = sizeof(float) * 2 * static_cast<size_t>(d_ffn);
    if (!ctx->scratchpad || ctx->scratchpad_bytes < need) return -2;
    float* gate = static_cast<float*>(ctx->scratchpad);
    float* up = gate + d_ffn;
    if (launch_matvec(w_gate, xin, gate, false, s) || launch_matvec(w_up, xin, up, false, s))
        return -1;
    glu_act<<<(d_ffn + 255) / 256, 256, 0, s>>>(gate, up, d_ffn, act == SPITE_FFN_GELU_GATE);
    if (launch_matvec(w_down, gate, static_cast<float*>(out->data), true, s)) return -1;
    return finish();
}

extern "C" int qwen3_cuda_matmul(SpiteTensor* out, const SpiteTensor* x, const SpiteTensor* w,
                                 const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (launch_matvec(w, static_cast<const float*>(x->data), static_cast<float*>(out->data),
                      false, stream_of(ctx)))
        return -1;
    return finish();
}

/* KV-cache tiers this attention op reads and writes (bit = SpiteType). */
extern "C" uint64_t qwen3_cuda_kv_cache_kinds() {
    return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16) | (1ull << SPITE_TYPE_Q8_0) |
           (1ull << SPITE_TYPE_Q5_1) | (1ull << SPITE_TYPE_Q4_0);
}

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen3",
    "sm_120",
    "spite project (rtx_5090 tuned card path)",
    {SPITE_TYPE_F32, SPITE_TYPE_F16, SPITE_TYPE_Q8_0, 0, 0, 0, 0, 0},
    qwen3_cuda_rms_norm,
    qwen3_cuda_attention,
    nullptr, /* mla */
    qwen3_cuda_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    qwen3_cuda_matmul,
    qwen3_cuda_kv_cache_kinds,
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
