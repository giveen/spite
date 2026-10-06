/*
 * kernels/qwen/qwen3_5/nvidia/sm_120/kernel.cu
 *
 * Blackwell (sm_120) architecture kernel for Qwen3.5 hybrid decoders.
 * Scope: all sm_120-family GPUs (RTX 5090, 5080, 5070, 5060, ...).
 *
 * Improvements over generic CUDA baseline:
 *   • Vectorized 128-bit memory transactions (float4) in rms_norm and mtp_stem.
 *   • Fully unrolled warp reductions (5 explicit shuffles, no loop)
 *     exploiting Blackwell's dual-warp issue scheduler.
 *   • Fused MTP stem with vectorized dual-stream reduction and float4 packing.
 */

#include "common.h"

namespace {

/* Fully unrolled warp reduction for Blackwell dual-issue scheduler */
__device__ __forceinline__ float sm120_warp_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v,  8);
  v += __shfl_xor_sync(0xffffffffu, v,  4);
  v += __shfl_xor_sync(0xffffffffu, v,  2);
  v += __shfl_xor_sync(0xffffffffu, v,  1);
  return v;
}

__device__ __forceinline__ float sm120_block_sum(float v) {
  __shared__ float red[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = sm120_warp_sum(v);
  __syncthreads();
  if (lane == 0)
    red[wid] = v;
  __syncthreads();
  float t = 0.0f;
  for (int i = 0; i < static_cast<int>(blockDim.x >> 5); ++i)
    t += red[i];
  return t;
}

/* Vectorized 128-bit float4 RMSNorm */
__global__ void sm120_rms_norm_rows(float *__restrict__ out,
                                    const float *__restrict__ x,
                                    const void *__restrict__ w, int wkind,
                                    int cols, float eps) {
  const float *xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float *orow = out + static_cast<size_t>(blockIdx.x) * cols;
  const int cols4 = cols / 4;
  const float4 *x4 = reinterpret_cast<const float4 *>(xr);

  float ss = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    float4 v = x4[i];
    ss = fmaf(v.x, v.x, ss);
    ss = fmaf(v.y, v.y, ss);
    ss = fmaf(v.z, v.z, ss);
    ss = fmaf(v.w, v.w, ss);
  }
  for (int i = cols4 * 4 + threadIdx.x; i < cols; i += blockDim.x) {
    float v = xr[i];
    ss = fmaf(v, v, ss);
  }

  ss = sm120_block_sum(ss);
  const float scale = rsqrtf(ss / cols + eps);

  float4 *o4 = reinterpret_cast<float4 *>(orow);
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    float4 v = x4[i];
    v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
    if (wkind == SPITE_TYPE_F32) {
      float4 wv = reinterpret_cast<const float4 *>(w)[i];
      v.x *= wv.x; v.y *= wv.y; v.z *= wv.z; v.w *= wv.w;
    } else {
      v.x *= q35_load_w(w, wkind, 4 * i + 0);
      v.y *= q35_load_w(w, wkind, 4 * i + 1);
      v.z *= q35_load_w(w, wkind, 4 * i + 2);
      v.w *= q35_load_w(w, wkind, 4 * i + 3);
    }
    o4[i] = v;
  }
  for (int i = cols4 * 4 + threadIdx.x; i < cols; i += blockDim.x) {
    orow[i] = xr[i] * scale * q35_load_w(w, wkind, i);
  }
}

/* Vectorized 128-bit float4 MTP stem */
__global__ void sm120_mtp_stem_kernel(float *__restrict__ out,
                                      const float *__restrict__ embed,
                                      const float *__restrict__ hidden,
                                      const void *__restrict__ w_enorm, int enorm_kind,
                                      const void *__restrict__ w_hnorm, int hnorm_kind,
                                      int d, float eps) {
  const int tok = blockIdx.y;
  const float *e_tok = embed + static_cast<size_t>(tok) * d;
  const float *h_tok = hidden + static_cast<size_t>(tok) * d;
  float *out_tok = out + static_cast<size_t>(tok) * (2 * d);

  const int d4 = d / 4;
  const float4 *e4 = reinterpret_cast<const float4 *>(e_tok);
  const float4 *h4 = reinterpret_cast<const float4 *>(h_tok);

  float ss_e = 0.0f;
  float ss_h = 0.0f;
  for (int i = threadIdx.x; i < d4; i += blockDim.x) {
    float4 ev = e4[i];
    ss_e = fmaf(ev.x, ev.x, ss_e);
    ss_e = fmaf(ev.y, ev.y, ss_e);
    ss_e = fmaf(ev.z, ev.z, ss_e);
    ss_e = fmaf(ev.w, ev.w, ss_e);

    float4 hv = h4[i];
    ss_h = fmaf(hv.x, hv.x, ss_h);
    ss_h = fmaf(hv.y, hv.y, ss_h);
    ss_h = fmaf(hv.z, hv.z, ss_h);
    ss_h = fmaf(hv.w, hv.w, ss_h);
  }
  for (int i = d4 * 4 + threadIdx.x; i < d; i += blockDim.x) {
    float ev = e_tok[i];
    ss_e = fmaf(ev, ev, ss_e);
    float hv = h_tok[i];
    ss_h = fmaf(hv, hv, ss_h);
  }

  ss_e = sm120_block_sum(ss_e);
  ss_h = sm120_block_sum(ss_h);

  const float scale_e = rsqrtf(ss_e / d + eps);
  const float scale_h = rsqrtf(ss_h / d + eps);

  float4 *o_e4 = reinterpret_cast<float4 *>(out_tok);
  float4 *o_h4 = reinterpret_cast<float4 *>(out_tok + d);

  for (int i = threadIdx.x; i < d4; i += blockDim.x) {
    float4 ev = e4[i];
    ev.x *= scale_e; ev.y *= scale_e; ev.z *= scale_e; ev.w *= scale_e;
    if (enorm_kind == SPITE_TYPE_F32) {
      float4 nw = reinterpret_cast<const float4 *>(w_enorm)[i];
      ev.x *= nw.x; ev.y *= nw.y; ev.z *= nw.z; ev.w *= nw.w;
    } else {
      ev.x *= q35_load_w(w_enorm, enorm_kind, 4 * i + 0);
      ev.y *= q35_load_w(w_enorm, enorm_kind, 4 * i + 1);
      ev.z *= q35_load_w(w_enorm, enorm_kind, 4 * i + 2);
      ev.w *= q35_load_w(w_enorm, enorm_kind, 4 * i + 3);
    }
    o_e4[i] = ev;

    float4 hv = h4[i];
    hv.x *= scale_h; hv.y *= scale_h; hv.z *= scale_h; hv.w *= scale_h;
    if (hnorm_kind == SPITE_TYPE_F32) {
      float4 nw = reinterpret_cast<const float4 *>(w_hnorm)[i];
      hv.x *= nw.x; hv.y *= nw.y; hv.z *= nw.z; hv.w *= nw.w;
    } else {
      hv.x *= q35_load_w(w_hnorm, hnorm_kind, 4 * i + 0);
      hv.y *= q35_load_w(w_hnorm, hnorm_kind, 4 * i + 1);
      hv.z *= q35_load_w(w_hnorm, hnorm_kind, 4 * i + 2);
      hv.w *= q35_load_w(w_hnorm, hnorm_kind, 4 * i + 3);
    }
    o_h4[i] = hv;
  }

  for (int i = d4 * 4 + threadIdx.x; i < d; i += blockDim.x) {
    out_tok[i]     = e_tok[i] * scale_e * q35_load_w(w_enorm, enorm_kind, i);
    out_tok[d + i] = h_tok[i] * scale_h * q35_load_w(w_hnorm, hnorm_kind, i);
  }
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

extern "C" int qwen35_sm120_rms_norm(SpiteTensor *out, const SpiteTensor *x,
                                     const SpiteTensor *weight, float eps,
                                     const SpiteCtx *ctx) {
  if (!out || !x || !weight || !out->data || !x->data || !weight->data)
    return -1;
  if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32)
    return -1;
  if (weight->kind != SPITE_TYPE_F32 && weight->kind != SPITE_TYPE_F16 &&
      weight->kind != SPITE_TYPE_BF16)
    return -1;
  if ((reinterpret_cast<uintptr_t>(weight->data) &
       (weight->kind == SPITE_TYPE_F32 ? 3 : 1)))
    return -1;
  const int64_t cols = x->ne[0], rows = x->ne[1] ? x->ne[1] : 1;
  if (cols < 1 || weight->ne[0] != x->ne[0] || out->ne[0] != x->ne[0])
    return -1;
  if (static_cast<int64_t>(out->ne[1] ? out->ne[1] : 1) < rows)
    return -1;

  int threads = static_cast<int>(((cols / 4 + 31) / 32) * 32);
  threads = threads < 32 ? 32 : (threads > 1024 ? 1024 : threads);
  sm120_rms_norm_rows<<<static_cast<unsigned>(rows), threads, 0, q35_stream(ctx)>>>(
      static_cast<float *>(out->data), static_cast<const float *>(x->data),
      weight->data, weight->kind, static_cast<int>(cols), eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int qwen35_sm120_ffn(SpiteTensor *out, const SpiteTensor *x,
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
  const size_t scratch_needed = 2 * static_cast<size_t>(d_ffn) * sizeof(float);
  if (!ctx->scratchpad || ctx->scratchpad_bytes < scratch_needed)
    return -2;

  float *gate = static_cast<float *>(ctx->scratchpad);
  float *up = gate + d_ffn;
  cudaStream_t s = q35_stream(ctx);
  const float *xin = static_cast<const float *>(x->data);

  Q35GemvJob jobs[2] = {{w_gate, gate}, {w_up, up}};
  if (q35_gemv_multi(jobs, 2, xin, false, s) != 0)
    return -1;

  const int block = 256;
  const int grid = static_cast<int>((d_ffn + block - 1) / block);
  glu_act<<<grid, block, 0, s>>>(gate, up, static_cast<int>(d_ffn),
                                 act == SPITE_FFN_GELU_GATE);

  if (q35_gemv(w_down, gate, static_cast<float *>(out->data), true, s) != 0)
    return -1;

  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int qwen35_sm120_matmul(SpiteTensor *out, const SpiteTensor *x,
                                   const SpiteTensor *w, const SpiteCtx *ctx) {
  if (!out || !x || !w || !out->data || !x->data)
    return -1;
  if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32)
    return -1;
  if (q35_gemv(w, static_cast<const float *>(x->data),
               static_cast<float *>(out->data), false, q35_stream(ctx)) != 0)
    return -1;
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int qwen35_sm120_mtp_stem(SpiteTensor *out, const SpiteTensor *embed,
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

  int threads = static_cast<int>(((d / 4 + 31) / 32) * 32);
  threads = threads < 32 ? 32 : (threads > 1024 ? 1024 : threads);
  dim3 grid(1, static_cast<unsigned>(t));

  sm120_mtp_stem_kernel<<<grid, threads, 0, q35_stream(ctx)>>>(
      static_cast<float *>(out->data), static_cast<const float *>(embed->data),
      static_cast<const float *>(hidden->data), w_enorm->data, w_enorm->kind,
      w_hnorm->data, w_hnorm->kind, static_cast<int>(d), eps);

  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" uint64_t qwen35_sm120_kv_cache_kinds() {
  return (1ull << SPITE_TYPE_F32) | (1ull << SPITE_TYPE_F16);
}

extern "C" int qwen35_cuda_linear_attn(
    SpiteTensor *, const SpiteTensor *, const SpiteTensor *,
    const SpiteTensor *, const SpiteTensor *, const SpiteTensor *,
    const SpiteTensor *, const SpiteTensor *, const SpiteTensor *,
    const SpiteTensor *, const SpiteTensor *, SpiteTensor *, SpiteTensor *,
    const SpiteGdnParams *, const SpiteCtx *);
extern "C" int qwen35_cuda_attention_ex(
    SpiteTensor *, const SpiteTensor *, const SpiteTensor *,
    const SpiteTensor *, const SpiteTensor *, const SpiteTensor *,
    const SpiteTensor *, const SpiteTensor *, float, SpiteKvCache *, float,
    const SpiteAttnParams *, const SpiteCtx *);

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen3_5",
    "sm_120",
    "spite project (Qwen3.5 Blackwell sm_120: float4 vectorized + dual-issue reductions)",
    {SPITE_TYPE_F16, SPITE_TYPE_BF16, SPITE_TYPE_Q8_0, SPITE_TYPE_Q4_K,
     SPITE_TYPE_Q5_K, SPITE_TYPE_Q6_K, SPITE_TYPE_NVFP4, 0},
    qwen35_sm120_rms_norm,
    nullptr, /* attention (hybrid attends via attention_ex) */
    nullptr, /* mla */
    qwen35_sm120_ffn,
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    qwen35_sm120_matmul,
    qwen35_sm120_kv_cache_kinds,
    qwen35_cuda_linear_attn,
    qwen35_cuda_attention_ex,
    qwen35_sm120_mtp_stem,
    qwen35_cuda_moe_ffn,
};

extern "C" const SpiteKernelInfo *spite_kernel_info() { return &KERNEL_INFO; }
