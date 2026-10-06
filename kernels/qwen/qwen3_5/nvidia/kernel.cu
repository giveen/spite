/*
 * kernels/qwen/qwen3_5/nvidia/kernel.cu
 *
 * Portable CUDA kernels for Qwen3.5 hybrid decoders on every NVIDIA GPU
 * (no architecture-locked PTX; sm_75 .. sm_120). This file holds the kernel
 * descriptor and the small ops; the big ones live next to it:
 *
 *   attn.cu    attention_ex  - gated-Q full attention, partial RoPE, F32/F16 KV
 *   gdn.cu     linear_attn   - whole Gated Delta Net layer
 *   gemv.cu    q35_gemv      - the shared dequantizing GEMV (all 28 SpiteTypes)
 *
 *   rms_norm   row-wise RMSNorm, F32 activations, F32/F16/BF16 norm weight
 *   ffn        SwiGLU / GeGLU, out += down(act(gate.x) * up.x), any weight type
 *   matmul     out = W . x for a weight of ANY SpiteType (LM head, projections)
 *
 * Every SpiteTensor.data is a DEVICE pointer; work is issued on
 * ctx->gpu_stream. Activations are F32 end to end, accumulation is F32, weights
 * are dequantized in registers (core/gpu/quant_gemv.h).
 *
 * Scratchpad: ffn needs gate[d_ffn] up[d_ffn]; attention_ex and linear_attn
 * document theirs (spite_attn_ex_scratch_floats / spite_gdn_scratch_floats).
 */
#include "common.h"

namespace {

/* One block per row; y = x * rsqrt(mean(x^2) + eps) * w. */
__global__ void rms_norm_rows(float *__restrict__ out,
                              const float *__restrict__ x,
                              const void *__restrict__ w, int wkind, int cols,
                              float eps) {
  const float *xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float *orow = out + static_cast<size_t>(blockIdx.x) * cols;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    ss += xr[i] * xr[i];
  ss = q35_block_sum(ss);
  const float scale = rsqrtf(ss / cols + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    orow[i] = xr[i] * scale * q35_load_w(w, wkind, i);
}

__global__ void glu_act(float *__restrict__ gate, const float *__restrict__ up,
                        int n, int gelu) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float g = gate[i];
  const float a =
      gelu ? 0.5f * g * (1.0f + tanhf(0.7978846f * (g + 0.044715f * g * g * g)))
           : q35_silu(g);
  gate[i] = a * up[i];
}

} // namespace

extern "C" int qwen35_cuda_rms_norm(SpiteTensor *out, const SpiteTensor *x,
                                    const SpiteTensor *weight, float eps,
                                    const SpiteCtx *ctx) {
  if (!out || !x || !weight || !out->data || !x->data || !weight->data)
    return -1;
  if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32)
    return -1;
  if (weight->kind != SPITE_TYPE_F32 && weight->kind != SPITE_TYPE_F16 &&
      weight->kind != SPITE_TYPE_BF16)
    return -1; // q35_load_w decodes only these; anything else must be -1, not
               // garbage
  if ((reinterpret_cast<uintptr_t>(weight->data) &
       (weight->kind == SPITE_TYPE_F32 ? 3 : 1)))
    return -1;
  const int64_t cols = x->ne[0], rows = x->ne[1] ? x->ne[1] : 1;
  if (cols < 1 || weight->ne[0] != x->ne[0] || out->ne[0] != x->ne[0])
    return -1;
  if (static_cast<int64_t>(out->ne[1] ? out->ne[1] : 1) < rows)
    return -1;
  // a block is a whole number of warps, 1024 threads at most
  int threads = static_cast<int>(((cols + 31) / 32) * 32);
  threads = threads < 32 ? 32 : (threads > 1024 ? 1024 : threads);
  rms_norm_rows<<<static_cast<unsigned>(rows), threads, 0, q35_stream(ctx)>>>(
      static_cast<float *>(out->data), static_cast<const float *>(x->data),
      weight->data, weight->kind, static_cast<int>(cols), eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int qwen35_cuda_ffn(SpiteTensor *out, const SpiteTensor *x,
                               const SpiteTensor *w_gate,
                               const SpiteTensor *w_up,
                               const SpiteTensor *w_down,
                               SpiteFfnActivation act, const SpiteCtx *ctx) {
  if (act != SPITE_FFN_SILU_GATE && act != SPITE_FFN_GELU_GATE)
    return -1;
  if (!out || !x || !w_gate || !w_up || !w_down || !ctx || !out->data ||
      !x->data)
    return -1;
  if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32)
    return -1;
  const int64_t d_ffn = w_gate->ne[1];
  if (d_ffn < 1 || w_up->ne[1] != w_gate->ne[1] ||
      w_up->ne[0] != w_gate->ne[0] || w_down->ne[0] != w_gate->ne[1] ||
      x->ne[0] != w_gate->ne[0] || out->ne[0] < w_down->ne[1])
    return -1;
  const int64_t m = x->ne[1] ? x->ne[1] : 1;   /* tokens (columns) */
  const int64_t n_act = d_ffn * m;
  if (!ctx->scratchpad ||
      ctx->scratchpad_bytes < sizeof(float) * static_cast<size_t>(2 * n_act))
    return -2;
  float *gate = static_cast<float *>(ctx->scratchpad);
  float *up = gate + n_act;
  const cudaStream_t st = q35_stream(ctx);
  const float *xin = static_cast<const float *>(x->data);
  if (m == 1) {
    const Q35GemvJob proj[2] = {{w_gate, gate}, {w_up, up}};
    if (q35_gemv_multi(proj, 2, xin, false, st))
      return -1;
  } else {
    /* Batched: one weight read per kBatchChunk columns. */
    if (q35_gemv_batch(w_gate, xin, gate, static_cast<int>(m), false, st) ||
        q35_gemv_batch(w_up, xin, up, static_cast<int>(m), false, st))
      return -1;
  }
  glu_act<<<static_cast<unsigned>((n_act + 255) / 256), 256, 0, st>>>(
      gate, up, static_cast<int>(n_act), act == SPITE_FFN_GELU_GATE);
  if (m == 1) {
    if (q35_gemv(w_down, gate, static_cast<float *>(out->data), true, st))
      return -1;
  } else if (q35_gemv_batch(w_down, gate, static_cast<float *>(out->data),
                            static_cast<int>(m), true, st)) {
    return -1;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* out[r] = sum_c W[r,c] * x[c]; W is [cols=ne[0], rows=ne[1]] of any SpiteType.
 */
extern "C" int qwen35_cuda_matmul(SpiteTensor *out, const SpiteTensor *x,
                                  const SpiteTensor *w, const SpiteCtx *ctx) {
  if (!out || !x || !w || !out->data || !x->data || !w->data)
    return -1;
  if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32)
    return -1;
  const int64_t rows = w->ne[1] ? w->ne[1] : 1;
  const int64_t m = x->ne[1] ? x->ne[1] : 1;
  if (x->ne[0] != w->ne[0])
    return -1; // x must supply exactly `cols` activations
  if (static_cast<int64_t>(out->ne[0]) < rows ||
      static_cast<int64_t>(out->ne[1] ? out->ne[1] : 1) < m)
    return -1;
  const cudaStream_t st = q35_stream(ctx);
  if (m == 1) {
    if (q35_gemv(w, static_cast<const float *>(x->data),
                 static_cast<float *>(out->data), false, st))
      return -1; // unsupported type, cols % block != 0 or misaligned weights
  } else if (q35_gemv_batch(w, static_cast<const float *>(x->data),
                            static_cast<float *>(out->data), static_cast<int>(m),
                            false, st)) {
    return -1;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* KV-cache tiers attention_ex reads and writes (bit = SpiteType). */
extern "C" uint64_t qwen35_cuda_kv_cache_kinds() {
  return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16);
}

extern "C" int qwen35_cuda_linear_attn(SpiteTensor *, const SpiteTensor *,
                                       const SpiteTensor *, const SpiteTensor *,
                                       const SpiteTensor *, const SpiteTensor *,
                                       const SpiteTensor *, const SpiteTensor *,
                                       const SpiteTensor *, const SpiteTensor *,
                                       const SpiteTensor *, SpiteTensor *,
                                       SpiteTensor *, const SpiteGdnParams *,
                                       const SpiteCtx *);
extern "C" int qwen35_cuda_attention_ex(
    SpiteTensor *, const SpiteTensor *, const SpiteTensor *,
    const SpiteTensor *, const SpiteTensor *, const SpiteTensor *,
    const SpiteTensor *, const SpiteTensor *, float, SpiteKvCache *, float,
    const SpiteAttnParams *, const SpiteCtx *);

/* Fused MTP stem: row RMSNorm on embedding and hidden, then pack into [2*d, T]. */
__global__ void mtp_stem_kernel(float *__restrict__ out,
                                const float *__restrict__ embed,
                                const float *__restrict__ hidden,
                                const void *__restrict__ w_enorm, int enorm_kind,
                                const void *__restrict__ w_hnorm, int hnorm_kind,
                                int d, float eps) {
  const int tok = blockIdx.y;
  const float *e_tok = embed + static_cast<size_t>(tok) * d;
  const float *h_tok = hidden + static_cast<size_t>(tok) * d;
  float *out_tok = out + static_cast<size_t>(tok) * (2 * d);

  float ss_e = 0.0f;
  float ss_h = 0.0f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    ss_e += e_tok[i] * e_tok[i];
    ss_h += h_tok[i] * h_tok[i];
  }
  ss_e = q35_block_sum(ss_e);
  ss_h = q35_block_sum(ss_h);

  const float scale_e = rsqrtf(ss_e / d + eps);
  const float scale_h = rsqrtf(ss_h / d + eps);

  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    out_tok[i]     = e_tok[i] * scale_e * q35_load_w(w_enorm, enorm_kind, i);
    out_tok[d + i] = h_tok[i] * scale_h * q35_load_w(w_hnorm, hnorm_kind, i);
  }
}

extern "C" int qwen35_cuda_mtp_stem(SpiteTensor *out, const SpiteTensor *embed,
                                    const SpiteTensor *hidden,
                                    const SpiteTensor *w_enorm,
                                    const SpiteTensor *w_hnorm, float eps,
                                    const SpiteCtx *ctx) {
  if (!out || !embed || !hidden || !w_enorm || !w_hnorm || !out->data ||
      !embed->data || !hidden->data)
    return -1;
  if (embed->kind != SPITE_TYPE_F32 || hidden->kind != SPITE_TYPE_F32 ||
      out->kind != SPITE_TYPE_F32)
    return -1;

  const int64_t d = embed->ne[0];
  const int64_t t = embed->ne[1] ? embed->ne[1] : 1;
  if (d < 1 || hidden->ne[0] != d || out->ne[0] != 2 * d)
    return -1;

  int threads = static_cast<int>(((d + 31) / 32) * 32);
  threads = threads < 32 ? 32 : (threads > 1024 ? 1024 : threads);
  dim3 grid(1, static_cast<unsigned>(t));

  mtp_stem_kernel<<<grid, threads, 0, q35_stream(ctx)>>>(
      static_cast<float *>(out->data), static_cast<const float *>(embed->data),
      static_cast<const float *>(hidden->data), w_enorm->data, w_enorm->kind,
      w_hnorm->data, w_hnorm->kind, static_cast<int>(d), eps);

  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

#include "kernels/_engine/speculative/nvidia/speculative_round.cuh"

// ── Kernel descriptor ────────────────────────────────────────────────────

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen3_5",
    "cuda",
    "spite project (generic nvidia/cuda path)",
    /* 8 slots, 0-terminated => at most 7 advertised (F32 == 0 cannot be listed
     * and is always accepted). gemv.cu decodes all 28 SpiteTypes;
     * this is the "most used" subset. */
    {SPITE_TYPE_F16, SPITE_TYPE_BF16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q4_K,
     SPITE_TYPE_Q5_K, SPITE_TYPE_Q6_K, SPITE_TYPE_Q4_0, 0},
    qwen35_cuda_rms_norm,
    nullptr, /* attention (hybrid Qwen3.5 attends through attention_ex) */
    nullptr, /* mla */
    qwen35_cuda_ffn,
    nullptr, /* layer */
    spite::engine::spite_speculative_verify_cuda, /* speculative_verify */
    nullptr, /* prefill */
    qwen35_cuda_matmul,
    qwen35_cuda_kv_cache_kinds,
    qwen35_cuda_linear_attn,
    qwen35_cuda_attention_ex,
    qwen35_cuda_mtp_stem,
    qwen35_cuda_moe_ffn,
};

extern "C" const SpiteKernelInfo *spite_kernel_info() { return &KERNEL_INFO; }
