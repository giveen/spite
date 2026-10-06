/*
 * kernels/qwen/qwen3_5/nvidia/gdn.cu — Gated Delta Net layer (ABI v7
 * linear_attn) for every NVIDIA GPU (portable CUDA C++, sm_75 .. sm_120).
 *
 * One decode token:  out += W_out . gated_norm(core(x)). Math is specified at
 * SpiteGdnFn in core/abi.h and checked against kernels/generic/generic by
 * tools/verify/verify.py.
 *
 *   1. q35_gemv: qkv = w_qkv.x, z = w_gate.x, beta/alpha = w_beta.x / w_alpha.x
 *      (any weight SpiteType; F32 activations, F32 accumulation) into the
 *      caller's scratchpad: qkv | z | core | beta | alpha.
 *   2. gdn_conv_kernel: depthwise causal conv + SiLU on the C fused channels,
 *      one thread per channel. The thread owns its history column, so the
 *      history shift happens in the same pass with no inter-block ordering.
 *      conv_w is channel-major (tap k of channel c at conv_w[c*K + k]); the
 *      private history is tap-major (hist[i*C + c], i = 0 oldest).
 *   3. gdn_core_kernel: one block per value head. Threads own state COLUMNS in
 *      registers (thread (rg, s) holds rows [rg*S/2, (rg+1)*S/2) of column s of
 *      M[r][s]), so state traffic is coalesced across the block and the S-long
 *      dot products over r need only one 2-way smem exchange. q/k L2 norm,
 *      gate/beta, delta-rule update, readout o = M^T q, the gated RMS norm and
 *      the silu(z) gate are all fused; y lands in the `core` scratch.
 *   4. q35_gemv: out += w_out . y.
 *
 * State: [n_vh, S, S] F32, M[r][s] row-major. S in {64, 128} (register-resident
 * columns); anything else returns -1.
 */
#include "common.h"

namespace {

constexpr int kRowGroups =
    2; // threads per state column (rows split in two halves)

__global__ void gdn_conv_kernel(float *__restrict__ qkv,
                                float *__restrict__ hist,
                                const float *__restrict__ w, int C, int K) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C)
    return;
  const float x = qkv[c];
  const float *wc = w + static_cast<size_t>(c) * K;
  float acc = x * wc[K - 1];
  for (int i = 0; i < K - 1; ++i) {
    const float h = hist[static_cast<size_t>(i) * C + c];
    acc = fmaf(h, wc[i], acc);
    if (i > 0)
      hist[static_cast<size_t>(i - 1) * C + c] =
          h; // shift left in the same pass
  }
  if (K > 1)
    hist[static_cast<size_t>(K - 2) * C + c] = x;
  qkv[c] = q35_silu(acc);
}

template <int S>
__global__ void __launch_bounds__(kRowGroups *S)
    gdn_core_kernel(float *__restrict__ y, const float *__restrict__ qkv,
                    const float *__restrict__ z,
                    const float *__restrict__ beta_raw,
                    const float *__restrict__ alpha_raw,
                    const float *__restrict__ ssm_dt,
                    const float *__restrict__ ssm_a,
                    const float *__restrict__ ssm_norm,
                    float *__restrict__ state, int n_kh, float eps) {
  constexpr int NT = kRowGroups * S;
  constexpr int R = S / kRowGroups; // state rows held per thread
  constexpr int NW = NT / 32;

  const int vh = blockIdx.x;
  const int kh = vh % n_kh;
  const int tid = threadIdx.x;
  const int s = tid % S;  // owned state column
  const int rg = tid / S; // owned row half
  const int key_dim = n_kh * S;

  __shared__ float q_s[S];
  __shared__ float k_s[S];
  __shared__ float red[NW];
  __shared__ float xch_sk[kRowGroups][S];
  __shared__ float xch_o[kRowGroups][S];

  // threads [0,S) fetch q, [S,2S) fetch k (kRowGroups == 2)
  float val;
  if (tid < S) {
    val = qkv[kh * S + tid];
    q_s[tid] = val;
  } else {
    val = qkv[key_dim + kh * S + (tid - S)];
    k_s[tid - S] = val;
  }
  float ss = q35_warp_sum(val * val);
  if ((tid & 31) == 0)
    red[tid >> 5] = ss;
  __syncthreads();
  float tq = 0.f, tk = 0.f;
#pragma unroll
  for (int w = 0; w < S / 32; ++w) {
    tq += red[w];
    tk += red[S / 32 + w];
  }
  // l2norm(x) = x / sqrt(mean(x^2) + eps/S) / sqrt(S) == x * rsqrt(sum(x^2) +
  // eps); q is additionally scaled by 1/sqrt(S).
  const float nq = rsqrtf(tq + eps) * rsqrtf(static_cast<float>(S));
  const float nk = rsqrtf(tk + eps);
  if (tid < S)
    q_s[tid] *= nq;
  else
    k_s[tid - S] *= nk;
  __syncthreads();

  const float v = qkv[2 * key_dim + vh * S + s];
  const float beta = 1.0f / (1.0f + __expf(-beta_raw[vh]));
  const float a = alpha_raw[vh] + ssm_dt[vh];
  const float softplus = a > 20.0f ? a : log1pf(__expf(a));
  const float decay = __expf(softplus * ssm_a[vh]);

  const int r0 = rg * R;
  float *m =
      state + static_cast<size_t>(vh) * S * S + static_cast<size_t>(r0) * S + s;
  float col[R];
  float sk = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    col[r] = m[static_cast<size_t>(r) * S] * decay;
    sk = fmaf(col[r], k_s[r0 + r], sk);
  }
  xch_sk[rg][s] = sk;
  __syncthreads();
  sk = xch_sk[0][s] + xch_sk[1][s];
  const float d = (v - sk) * beta;

  float o = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    col[r] = fmaf(k_s[r0 + r], d, col[r]);
    m[static_cast<size_t>(r) * S] = col[r];
    o = fmaf(col[r], q_s[r0 + r], o);
  }
  xch_o[rg][s] = o;
  __syncthreads();
  o = xch_o[0][s] + xch_o[1][s];

  // gated RMS norm over the S outputs of this head (one contribution per
  // column)
  const float t = q35_block_sum(rg == 0 ? o * o : 0.0f);
  if (rg == 0) {
    const float scale = rsqrtf(t / S + eps);
    y[vh * S + s] = o * scale * ssm_norm[s] * q35_silu(z[vh * S + s]);
  }
}

} // namespace

extern "C" int
qwen35_cuda_linear_attn(SpiteTensor *out, const SpiteTensor *x,
                        const SpiteTensor *w_qkv, const SpiteTensor *w_gate,
                        const SpiteTensor *w_beta, const SpiteTensor *w_alpha,
                        const SpiteTensor *w_out, const SpiteTensor *conv_w,
                        const SpiteTensor *ssm_dt, const SpiteTensor *ssm_a,
                        const SpiteTensor *ssm_norm, SpiteTensor *conv_hist,
                        SpiteTensor *state, const SpiteGdnParams *p,
                        const SpiteCtx *ctx) {
  if (!out || !x || !w_qkv || !w_gate || !w_beta || !w_alpha || !w_out ||
      !conv_w || !ssm_dt || !ssm_a || !ssm_norm || !conv_hist || !state || !p ||
      !ctx)
    return -1;
  if (p->n_kh < 1 || p->n_vh < 1 || p->d_conv < 1 || p->n_vh % p->n_kh != 0)
    return -1;
  const int S = p->head_dim, K = p->d_conv;
  if (S != 64 && S != 128)
    return -1; // register-resident state columns
  const int64_t C = 2LL * p->n_kh * S + static_cast<int64_t>(p->n_vh) * S;
  const int64_t V = static_cast<int64_t>(p->n_vh) * S;
  const int64_t d_model = w_qkv->ne[0];

  if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32 || !x->data ||
      !out->data)
    return -1;
  if (static_cast<int64_t>(x->ne[0]) != d_model || w_gate->ne[0] != d_model ||
      w_beta->ne[0] != d_model || w_alpha->ne[0] != d_model)
    return -1;
  if (static_cast<int64_t>(w_qkv->ne[1]) != C ||
      static_cast<int64_t>(w_gate->ne[1]) != V ||
      w_beta->ne[1] != static_cast<uint32_t>(p->n_vh) ||
      w_alpha->ne[1] != static_cast<uint32_t>(p->n_vh) ||
      static_cast<int64_t>(w_out->ne[0]) != V ||
      static_cast<int64_t>(out->ne[0]) < static_cast<int64_t>(w_out->ne[1]))
    return -1;
  if (!q35_f32_n(conv_w, K * C) || !q35_f32_n(ssm_dt, p->n_vh) ||
      !q35_f32_n(ssm_a, p->n_vh) || !q35_f32_n(ssm_norm, S) ||
      !q35_f32_n(state, static_cast<int64_t>(p->n_vh) * S * S))
    return -1;
  if (K > 1 && !q35_f32_n(conv_hist, (K - 1) * C)) // K == 1 keeps no history
    return -1;

  if (!ctx->scratchpad ||
      ctx->scratchpad_bytes < sizeof(float) * spite_gdn_scratch_floats(p) ||
      (reinterpret_cast<uintptr_t>(ctx->scratchpad) & 15))
    return -2;
  float *qkv = static_cast<float *>(ctx->scratchpad);
  float *z = qkv + C;
  float *core = z + V;
  float *beta = core + V;
  float *alpha = beta + p->n_vh;

  const cudaStream_t st = q35_stream(ctx);
  const float *xin = static_cast<const float *>(x->data);
  const Q35GemvJob proj[4] = {{w_qkv, qkv}, {w_gate, z}, {w_beta, beta}, {w_alpha, alpha}};
  if (q35_gemv_multi(proj, 4, xin, false, st))
    return -1;

  gdn_conv_kernel<<<static_cast<unsigned>((C + 127) / 128), 128, 0, st>>>(
      qkv, K > 1 ? static_cast<float *>(conv_hist->data) : nullptr,
      static_cast<const float *>(conv_w->data), static_cast<int>(C), K);

  const float *dt = static_cast<const float *>(ssm_dt->data);
  const float *a = static_cast<const float *>(ssm_a->data);
  const float *nw = static_cast<const float *>(ssm_norm->data);
  float *m = static_cast<float *>(state->data);
  if (S == 128)
    gdn_core_kernel<128><<<p->n_vh, kRowGroups * 128, 0, st>>>(
        core, qkv, z, beta, alpha, dt, a, nw, m, p->n_kh, p->norm_eps);
  else
    gdn_core_kernel<64><<<p->n_vh, kRowGroups * 64, 0, st>>>(
        core, qkv, z, beta, alpha, dt, a, nw, m, p->n_kh, p->norm_eps);

  if (q35_gemv(w_out, core, static_cast<float *>(out->data), true, st))
    return -1;
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
