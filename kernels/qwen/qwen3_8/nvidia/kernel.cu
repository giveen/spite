/*
 * kernels/qwen/qwen3_8/nvidia/kernel.cu
 *
 * Generic CUDA kernel for Qwen3.8 dense decoders on all NVIDIA/CUDA
 * architectures (sm_60 Pascal through sm_120 Blackwell).  The decoder ops are
 * the Qwen3 vendor kernel (kernels/qwen/qwen3/nvidia/kernel.cu) unchanged;
 * Qwen3.8 adds the MTP (NextN) draft head, whose stem is fused here.
 *
 * Every SpiteTensor.data is a DEVICE pointer; all work is issued on
 * ctx->gpu_stream (null = default).
 *
 *   rms_norm  — row-wise RMSNorm (also used for per-head QK norm)
 *   attention — Q/K/V matvec → per-head QK RMSNorm → NEOX RoPE → KV write
 *               at ctx->pos → causal GQA attention → out-proj, out += result
 *   ffn       — SwiGLU / GeGLU, out += down(act(gate) * up)
 *   matmul    — out = W · x (LM head, and the MTP eh_proj)
 *   mtp_stem  — out[t] = [rmsnorm(embed[t], enorm) || rmsnorm(hidden[t], hnorm)]
 *
 * The rest of the MTP block (eh_proj, one attention + FFN layer with its own
 * KV cache, shared head) is ordinary matmul/attention/ffn calls by the host.
 *
 * Attention is the flash-decoding back end from kv_attn_flash.inl: the KV axis
 * is split across blocks, each block stages one KV tile in shared memory and
 * shares it across the whole GQA group, and the softmax is folded into the
 * weighted-V pass (no nh*n_ctx score vector).  kv_attn.inl keeps the portable
 * VBR back end that it falls back to for shapes the tile kernel does not cover.
 *
 * Weight types: any SpiteType (F32/F16/Q8_0 tuned here, the rest via core/gpu/quant_gemv.h). Activations and KV cache: F32.
 * Single-token decode semantics (prefill = repeated decode by the host).
 *
 * Scratchpad (ctx->scratchpad, device, floats):
 *   attention: q[nh*hd] k[nkv*hd] att[nh*hd] workspace[nh*n_ctx] vtmp[nkv*hd]
 *   ffn:       gate[d_ffn] up[d_ffn]
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

/* Block-wide sum; blockDim.x must be a multiple of 32 (<= 1024). */
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
        // every other SpiteType: dequantize-in-register GEMV (0, or -1 if undecodable)
        return sq::gemv(w, x, y, accumulate, s);
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

// The F32-only qk_norm_rope / attn_scores / attn_weighted_v that used to live
// here are superseded by the shared VBR kernels in kv_attn.inl; only the
// softmax below is still shared with that file.

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

// Shared VBR (quantized-KV) attention. Uses this file's launch_matvec,
// threads_for, stream_of, attn_softmax, block_sum and load_w.
#include "kernels/qwen/qwen3_8/nvidia/kv_attn.inl"

// Flash-decoding attention (portable CUDA C++): split-KV over the KV axis,
// GQA-group tile reuse, fused online softmax. Uses this file's warp_max,
// warp_sum, threads_for, stream_of and launch_matvec, plus kv_attn.inl's
// kvq_get/kvq_row_bytes/kvattn_prologue.
#include "kernels/qwen/qwen3_8/nvidia/kv_attn_flash.inl"

// ── ABI ops ──────────────────────────────────────────────────────────────

extern "C" int qwen38_cuda_rms_norm(SpiteTensor* out, const SpiteTensor* x,
                                   const SpiteTensor* weight, float eps, const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    // load_w() only decodes F32/F16 norm weights; anything else must be -1, not garbage
    if (weight->kind != SPITE_TYPE_F32 && weight->kind != SPITE_TYPE_F16) return -1;
    const int cols = static_cast<int>(x->ne[0]);
    const int rows = static_cast<int>(x->ne[1] ? x->ne[1] : 1);
    if (weight->ne[0] != x->ne[0]) return -1;
    rms_norm_rows<<<rows, threads_for(cols), 0, stream_of(ctx)>>>(
        static_cast<float*>(out->data), static_cast<const float*>(x->data), weight->data,
        weight->kind, cols, eps);
    return finish();
}

extern "C" int qwen38_cuda_attention(SpiteTensor* out, const SpiteTensor* x,
                                    const SpiteTensor* wq, const SpiteTensor* wk,
                                    const SpiteTensor* wv, const SpiteTensor* wo,
                                    const SpiteTensor* q_norm, const SpiteTensor* k_norm,
                                    float norm_eps, SpiteKvCache* kv, float rope_freq_base,
                                    const SpiteCtx* ctx) {
    return kvflash_run(out, x, wq, wk, wv, wo, q_norm, k_norm, norm_eps, kv, rope_freq_base, ctx);
}

extern "C" int qwen38_cuda_ffn(SpiteTensor* out, const SpiteTensor* x, const SpiteTensor* w_gate,
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

extern "C" int qwen38_cuda_matmul(SpiteTensor* out, const SpiteTensor* x, const SpiteTensor* w,
                                 const SpiteCtx* ctx) {
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) return -1;
    if (launch_matvec(w, static_cast<const float*>(x->data), static_cast<float*>(out->data),
                      false, stream_of(ctx)))
        return -1;
    return finish();
}

// ── MTP (NextN) stem ─────────────────────────────────────────────────────

/* One block per token: both RMSNorms share the block, and the two halves of the
 * packed [2*d] row are written in one pass. */
__global__ void mtp_stem_rows(float* __restrict__ out, const float* __restrict__ embed,
                              const float* __restrict__ hidden, const void* __restrict__ w_e,
                              int ekind, const void* __restrict__ w_h, int hkind, int d,
                              float eps) {
    const float* e = embed + static_cast<size_t>(blockIdx.x) * d;
    const float* h = hidden + static_cast<size_t>(blockIdx.x) * d;
    float* o = out + static_cast<size_t>(blockIdx.x) * 2 * d;
    float se = 0.0f, sh = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        se = fmaf(e[i], e[i], se);
        sh = fmaf(h[i], h[i], sh);
    }
    se = block_sum(se);
    sh = block_sum(sh);
    const float ce = rsqrtf(se / d + eps);
    const float ch = rsqrtf(sh / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        o[i] = e[i] * ce * load_w(w_e, ekind, i);
        o[d + i] = h[i] * ch * load_w(w_h, hkind, i);
    }
}

extern "C" int qwen38_cuda_mtp_stem(SpiteTensor* out, const SpiteTensor* embed,
                                    const SpiteTensor* hidden, const SpiteTensor* w_enorm,
                                    const SpiteTensor* w_hnorm, float eps, const SpiteCtx* ctx) {
    // Qwen NextN always carries both norms; the norm-less (Gemma) variant is not this model.
    if (!out || !embed || !hidden || !w_enorm || !w_hnorm) return -1;
    if (out->kind != SPITE_TYPE_F32 || embed->kind != SPITE_TYPE_F32 ||
        hidden->kind != SPITE_TYPE_F32)
        return -1;
    if ((w_enorm->kind != SPITE_TYPE_F32 && w_enorm->kind != SPITE_TYPE_F16) ||
        (w_hnorm->kind != SPITE_TYPE_F32 && w_hnorm->kind != SPITE_TYPE_F16))
        return -1;
    const int d = static_cast<int>(embed->ne[0]);
    const int t = static_cast<int>(embed->ne[1] ? embed->ne[1] : 1);
    const int th = static_cast<int>(hidden->ne[1] ? hidden->ne[1] : 1);
    if (d < 1 || static_cast<int>(hidden->ne[0]) != d || th != t ||
        static_cast<int>(out->ne[0]) != 2 * d || static_cast<int>(w_enorm->ne[0]) != d ||
        static_cast<int>(w_hnorm->ne[0]) != d)
        return -1;
    mtp_stem_rows<<<t, threads_for(d), 0, stream_of(ctx)>>>(
        static_cast<float*>(out->data), static_cast<const float*>(embed->data),
        static_cast<const float*>(hidden->data), w_enorm->data, w_enorm->kind, w_hnorm->data,
        w_hnorm->kind, d, eps);
    return finish();
}

// ── Kernel descriptor ────────────────────────────────────────────────────

/* KV-cache tiers this attention op reads and writes (bit = SpiteType). */
extern "C" uint64_t qwen38_cuda_kv_cache_kinds() {
    return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16) | (1ull << SPITE_TYPE_Q8_0) |
           (1ull << SPITE_TYPE_Q5_1) | (1ull << SPITE_TYPE_Q4_0);
}

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen38",
    "cuda",
    "spite project (qwen3.8 generic nvidia/cuda path + MTP stem)",
    /* supported_quants has 8 slots and 0 terminates the list (so F32 == 0 cannot be
     * listed and is always accepted): at most 7 types are advertised. The shared GEMV
     * (core/gpu/quant_gemv.h) handles all 28 SpiteTypes; types outside this list still
     * work in matvec/ffn/matmul, and anything it cannot decode returns -1. */
    {SPITE_TYPE_F16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q4_K, SPITE_TYPE_Q5_K, SPITE_TYPE_Q6_K,
     SPITE_TYPE_Q4_0, SPITE_TYPE_Q3_K, 0},
    qwen38_cuda_rms_norm,
    qwen38_cuda_attention,
    nullptr, /* mla */
    qwen38_cuda_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify (resolved from kernels/_engine/speculative/) */
    nullptr, /* prefill */
    qwen38_cuda_matmul,
    qwen38_cuda_kv_cache_kinds,
    nullptr, /* linear_attn */
    nullptr, /* attention_ex */
    qwen38_cuda_mtp_stem,
    nullptr, /* moe_ffn */
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
