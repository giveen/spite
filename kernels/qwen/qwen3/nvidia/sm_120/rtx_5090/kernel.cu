/*
 * kernels/qwen/qwen3/nvidia/sm_120/rtx_5090/kernel.cu
 *
 * General-purpose CUDA kernel for Qwen3 dense decoders (no tensor cores,
 * no tiling — correct first, fast later). Every SpiteTensor.data is a
 * DEVICE pointer; all work is issued on ctx->gpu_stream (null = default).
 *
 *   rms_norm  — row-wise RMSNorm (also used for per-head QK norm)
 *   attention — Q/K/V matvec → per-head QK RMSNorm → NEOX RoPE → KV write
 *               at ctx->pos → causal GQA softmax → out-proj, out += result
 *   ffn       — SwiGLU / GeGLU, out += down(act(gate) * up)
 *   matmul    — out = W · x (LM head)
 *
 * Weight types: F32, F16, Q8_0. Activations and KV cache: F32.
 * Single-token decode semantics (prefill = repeated decode by the host).
 *
 * Scratchpad (ctx->scratchpad, device, floats):
 *   attention: q[nh*hd] k[nkv*hd] att[nh*hd] scores[nh*n_ctx]
 *   ffn:       gate[d_ffn] up[d_ffn]
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
constexpr int ROWS_PER_BLOCK = 4;  // one warp per output row

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

/* Block-wide sum; blockDim.x must be a multiple of 32 (≤ 1024). */
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

/* Load element i of a small (norm) weight vector of type kind. */
__device__ __forceinline__ float load_w(const void* w, int kind, int i) {
    if (kind == SPITE_TYPE_F16) return __half2float(static_cast<const __half*>(w)[i]);
    return static_cast<const float*>(w)[i];
}

// ── Matvec: y[r] (+)= Σ_c W[r,c] · x[c] ──────────────────────────────────

__global__ void matvec_q8_0(const BlockQ8_0* __restrict__ w, const float* __restrict__ x,
                            float* __restrict__ y, int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows) return;
    const int nb = cols / QK8_0;
    const BlockQ8_0* wr = w + static_cast<size_t>(row) * nb;
    float acc = 0.0f;
    // Each lane owns one quant within a block; the warp walks blocks.
    for (int b = 0; b < nb; ++b) {
        const float d = __half2float(wr[b].d);
        acc += d * static_cast<float>(wr[b].qs[threadIdx.x]) * x[b * QK8_0 + threadIdx.x];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }

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

/* Launch matvec for weight tensor w ([cols, rows]); returns 0 or -1. */
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
        matvec_dense<float><<<grid, block, 0, s>>>(static_cast<const float*>(w->data), x, y,
                                                   rows, cols, accumulate);
        break;
    case SPITE_TYPE_F16:
        matvec_dense<__half><<<grid, block, 0, s>>>(static_cast<const __half*>(w->data), x, y,
                                                    rows, cols, accumulate);
        break;
    default:
        return -1;
    }
    return 0;
}

// ── RMSNorm (row-wise) ───────────────────────────────────────────────────

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

// ── Attention pieces ─────────────────────────────────────────────────────

/*
 * One block per head (blockDim = head_dim/2 rounded up to a warp multiple).
 * Optional per-head RMSNorm, then NEOX RoPE (pairs (i, i+hd/2)).
 * Writes the result to dst + head*hd (dst may be the KV-cache row).
 */
/* dst may alias src (in-place Q); no __restrict__ on those two. */
__global__ void qk_norm_rope(float* dst, const float* src, const void* __restrict__ nw, int nkind, float eps, int hd, int pos,
                             float theta) {
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
        d[i] = x0 * cs - x1 * sn;
        d[i + half] = x0 * sn + x1 * cs;
    }
}

/* scores[h, t] = q_h · k_t[kvh] * scale, t in [0, n_tok). One warp per (h, t). */
__global__ void attn_scores(float* __restrict__ scores, const float* __restrict__ q,
                            const float* __restrict__ kc, int n_tok, int n_ctx, int hd,
                            int group, int kv_stride, float scale) {
    const int h = blockIdx.y;
    const int t = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (t >= n_tok) return;
    const float* qh = q + static_cast<size_t>(h) * hd;
    const float* kt = kc + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(h / group) * hd;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hd; i += WARP) acc += qh[i] * kt[i];
    acc = warp_sum(acc);
    if (threadIdx.x == 0) scores[static_cast<size_t>(h) * n_ctx + t] = acc * scale;
}

/* In-place softmax over scores[h, 0..n_tok). One block per head. */
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

/* att[h, i] = Σ_t p[h, t] · v_t[kvh, i]. Block per head, thread per dim. */
__global__ void attn_weighted_v(float* __restrict__ att, const float* __restrict__ scores,
                                const float* __restrict__ vc, int n_tok, int n_ctx, int hd,
                                int group, int kv_stride) {
    const int h = blockIdx.x;
    const float* p = scores + static_cast<size_t>(h) * n_ctx;
    const float* vb = vc + static_cast<size_t>(h / group) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_tok; ++t) acc += p[t] * vb[static_cast<size_t>(t) * kv_stride + i];
        att[static_cast<size_t>(h) * hd + i] = acc;
    }
}

// ── FFN activation ───────────────────────────────────────────────────────

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

// ── ABI ops ──────────────────────────────────────────────────────────────

extern "C" int qwen3_cuda_rms_norm(SpiteTensor* out, const SpiteTensor* x,
                                   const SpiteTensor* weight, float eps, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    const int cols = static_cast<int>(x->ne[0]);
    const int rows = static_cast<int>(x->ne[1] ? x->ne[1] : 1);
    if (weight->ne[0] != x->ne[0]) return -1;
    rms_norm_rows<<<rows, threads_for(cols), 0, stream_of(ctx)>>>(
        static_cast<float*>(out->data), static_cast<const float*>(x->data), weight->data,
        weight->kind, cols, eps);
    return finish();
}

extern "C" int qwen3_cuda_attention(SpiteTensor* out, const SpiteTensor* x,
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
    float* q = static_cast<float*>(ctx->scratchpad);
    float* k = q + static_cast<size_t>(nh) * hd;
    float* att = k + kv_stride;
    float* scores = att + static_cast<size_t>(nh) * hd;

    cudaStream_t s = stream_of(ctx);
    const float* xin = static_cast<const float*>(x->data);
    float* kc = static_cast<float*>(kv->k.data);
    float* vc = static_cast<float*>(kv->v.data);
    float* k_row = kc + static_cast<size_t>(pos) * kv_stride;
    float* v_row = vc + static_cast<size_t>(pos) * kv_stride;

    // Projections; V goes straight into its cache row.
    if (launch_matvec(wq, xin, q, false, s) || launch_matvec(wk, xin, k, false, s) ||
        launch_matvec(wv, xin, v_row, false, s))
        return -1;

    const int rt = threads_for(hd / 2);
    qk_norm_rope<<<nh, rt, 0, s>>>(q, q, q_norm ? q_norm->data : nullptr,
                                   q_norm ? q_norm->kind : 0, norm_eps, hd, pos, rope_freq_base);
    qk_norm_rope<<<nkv, rt, 0, s>>>(k_row, k, k_norm ? k_norm->data : nullptr,
                                    k_norm ? k_norm->kind : 0, norm_eps, hd, pos, rope_freq_base);

    const int n_tok = pos + 1;
    const int group = nh / nkv;
    const float scale = rsqrtf(static_cast<float>(hd));
    attn_scores<<<dim3((n_tok + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, nh),
                  dim3(WARP, ROWS_PER_BLOCK), 0, s>>>(scores, q, kc, n_tok, n_ctx, hd, group,
                                                      kv_stride, scale);
    attn_softmax<<<nh, threads_for(n_tok > 256 ? 256 : n_tok), 0, s>>>(scores, n_tok, n_ctx);
    attn_weighted_v<<<nh, threads_for(hd), 0, s>>>(att, scores, vc, n_tok, n_ctx, hd, group,
                                                   kv_stride);

    if (launch_matvec(wo, att, static_cast<float*>(out->data), true, s)) return -1;
    return finish();
}

extern "C" int qwen3_cuda_ffn(SpiteTensor* out, const SpiteTensor* x, const SpiteTensor* w_gate,
                              const SpiteTensor* w_up, const SpiteTensor* w_down,
                              SpiteFfnActivation act, const SpiteCtx* ctx) {
    if (act != SPITE_FFN_SILU_GATE && act != SPITE_FFN_GELU_GATE) return -1;
    if (!ctx) return -1;
    const int d_ffn = static_cast<int>(w_gate->ne[1]);
    const size_t need = sizeof(float) * 2 * static_cast<size_t>(d_ffn);
    if (!ctx->scratchpad || ctx->scratchpad_bytes < need) return -2;
    float* gate = static_cast<float*>(ctx->scratchpad);
    float* up = gate + d_ffn;
    cudaStream_t s = stream_of(ctx);
    const float* xin = static_cast<const float*>(x->data);
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

// ── Kernel descriptor ────────────────────────────────────────────────────

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen3",
    "sm_120",
    "spite project (general CUDA path)",
    {SPITE_TYPE_F32, SPITE_TYPE_F16, SPITE_TYPE_Q8_0, 0, 0, 0, 0, 0},
    qwen3_cuda_rms_norm,
    qwen3_cuda_attention,
    nullptr, /* mla */
    qwen3_cuda_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    qwen3_cuda_matmul,
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
