/*
 * kernels/qwen/qwen3_5/nvidia/attn.cu — attention_ex (ABI v7) for every NVIDIA
 * GPU (portable CUDA C++, sm_75 .. sm_120): the full-attention layers of hybrid
 * Qwen3.5 decoders. Per-head gated Q projection, per-head q/k RMSNorm, partial
 * NEOX RoPE (first rope_dim dims), KV write at ctx->pos, causal GQA softmax
 * attention over F32 or F16 KV cache rows, sigmoid output gate, out += wo . att.
 * With gated_q = 0 and rope_dim == head_dim this is exactly the plain attention
 * op. Semantics: SpiteAttentionExFn in core/abi.h. head_dim in {64, 128, 256}
 * (lane layout below); anything else returns -1.
 *
 * Launch sequence (all on ctx->gpu_stream):
 *   q35_gemv_multi  wq.x (-> q|gate, interleaved per head), wk.x, wv.x: ONE
 *                   launch when the three weights share a type
 *   attn_prep       one warp per head: q/k RMSNorm + RoPE (q in place, k into the
 *                   cache row), V row into the cache (F32 or F16); zeroes the
 *                   split-K arrival counters
 *   attn_split      flash-decoding. Grid (KV chunk, kv head x query-head
 *                   subgroup of <= 4 heads). Each warp streams its own tokens of
 *                   the chunk straight from global memory — lane l owns dims
 *                   {VEC*l + 32*VEC*j} of every row, so a row is read as
 *                   coalesced 8/16-byte lanes with no shared-memory staging — and
 *                   keeps an online-softmax state for the block's GS query heads
 *                   (K/V is loaded once and used for all of them). Warps merge in
 *                   shared memory; the block writes one (m, l, acc) partial per
 *                   head, and the LAST block to finish a (kv head, subgroup)
 *                   (atomic ticket) folds all chunks and applies the sigmoid gate.
 *                   If the whole context is one chunk the block writes the gated
 *                   head output directly (no partials, no ticket).
 *   q35_gemv        out += wo . att
 *
 * Scratchpad (floats; spite_attn_ex_scratch_floats is the guaranteed minimum):
 *   q|gate [nh*hd*(gated?2:1)] | att [nh*hd] | k [nkv*hd] | v [nkv*hd] | workspace
 * The workspace holds the split-K partials ((hd+2) floats per chunk and head) and
 * the tickets. Chunks: as many as ~0.7 blocks per SM, never finer than 64 tokens,
 * and no more than the workspace fits (ctx/(hd+2) at the guaranteed minimum scratch:
 * 15 at hd=256, ctx=4096). Measured: finer splits are slower, the per-block fixed
 * cost dominates (see sm_count() use below).
 */
#include "common.h"

#include <math.h>

#include <algorithm>

namespace {

constexpr int kWarps = 8; // warps per attn_split block; also the smem merge width
constexpr int kTokensPerStep = 4; // tokens a warp keeps in flight (K and V each)
constexpr int kMinChunk = 64;   // never split the KV axis finer than this
constexpr int kMaxGroupSub = 4; // query heads per block (register budget)

// ── prep: q/k RMSNorm + partial NEOX RoPE, KV row write ──────────────────

template <typename T> __device__ __forceinline__ void store_kv(T *dst, float v);
template <>
__device__ __forceinline__ void store_kv<float>(float *dst, float v) {
  *dst = v;
}
template <>
__device__ __forceinline__ void store_kv<__half>(__half *dst, float v) {
  *dst = __float2half_rn(v);
}

/* One WARP per head (no block barriers, everything between two global trips is
 * shuffles and __syncwarp): blocks [0, nh) normalise+rotate query head b in place
 * (row stride qstride); blocks [nh, nh+nkv) do key head b-nh into the cache row and
 * copy value head b-nh into the value cache row. Block 0 also zeroes the split-K
 * arrival counters that attn_split_kernel's last-block combine uses. */
template <typename T>
__global__ void __launch_bounds__(32)
    attn_prep_kernel(float *__restrict__ qg, const float *__restrict__ kraw,
                     const float *__restrict__ vraw, T *__restrict__ krow,
                     T *__restrict__ vrow, const void *qn, int qnk,
                     const void *kn, int knk, float eps, int hd, int rd,
                     int qstride, int pos, double log2_base, int nh,
                     unsigned *counters, int ncounters) {
  __shared__ float sx[256];
  __shared__ float s_cos[128], s_sin[128];
  const int lane = threadIdx.x;
  const int b = blockIdx.x;
  const bool isq = b < nh;
  const int kh = b - nh;
  if (b == 0)
    for (int i = lane; i < ncounters; i += 32)
      counters[i] = 0u;
  const float *src = isq ? qg + static_cast<size_t>(b) * qstride
                         : kraw + static_cast<size_t>(kh) * hd;
  const void *nw = isq ? qn : kn;
  const int nk = isq ? qnk : knk;

  // issue every global load first (activations and the norm weight, which is cold),
  // so the rotation-table math below overlaps their latency
  constexpr int kPer = 256 / 32; // elements per lane at the largest head_dim
  float v[kPer], w[kPer], vv[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = lane + 32 * j;
    v[j] = i < hd ? src[i] : 0.0f;
    w[j] = (nw && i < hd) ? q35_load_w(nw, nk, i) : 1.0f;
    vv[j] = (!isq && i < hd) ? vraw[static_cast<size_t>(kh) * hd + i] : 0.0f;
  }

  // rotation table for this position: angle_i = pos * base^(-2i/rd). Evaluated in
  // double and reduced mod 2*pi there, so --use_fast_math cannot cost bits at long
  // context; only the reduced angle (|a| <= pi) goes through float sincospi.
  const int half = rd / 2;
  for (int i = lane; i < half; i += 32) {
    const double ang = static_cast<double>(pos) *
                       exp2(-(2.0 * i / rd) * log2_base);
    const double red =
        ang - rint(ang * 0.15915494309189535) * 6.283185307179586;
    sincospif(static_cast<float>(red * 0.3183098861837907), &s_sin[i],
              &s_cos[i]);
  }

  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < kPer; ++j)
    ss += v[j] * v[j];
  ss = q35_warp_sum(ss);
  const float scale = nw ? rsqrtf(ss / hd + eps) : 1.0f;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = lane + 32 * j;
    if (i < hd)
      sx[i] = v[j] * scale * w[j];
  }
  __syncwarp();
  for (int i = lane; i < hd; i += 32) {
    float r;
    if (i < half)
      r = sx[i] * s_cos[i] - sx[i + half] * s_sin[i];
    else if (i < rd)
      r = sx[i - half] * s_sin[i - half] + sx[i] * s_cos[i - half];
    else
      r = sx[i];
    if (isq)
      qg[static_cast<size_t>(b) * qstride + i] = r;
    else
      store_kv<T>(krow + static_cast<size_t>(kh) * hd + i, r);
  }
  if (!isq)
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (lane + 32 * j < hd)
        store_kv<T>(vrow + static_cast<size_t>(kh) * hd + lane + 32 * j, vv[j]);
}

// ── split-KV attention core ──────────────────────────────────────────────

/* Load VEC consecutive KV elements as floats. */
template <int VEC, typename T> struct LdVec;
template <int VEC> struct LdVec<VEC, float> {
  static __device__ __forceinline__ void load(const float *p, float *o) {
    if constexpr (VEC == 4) {
      const float4 f = __ldg(reinterpret_cast<const float4 *>(p));
      o[0] = f.x;
      o[1] = f.y;
      o[2] = f.z;
      o[3] = f.w;
    } else {
      const float2 f = __ldg(reinterpret_cast<const float2 *>(p));
      o[0] = f.x;
      o[1] = f.y;
    }
  }
};
template <int VEC> struct LdVec<VEC, __half> {
  static __device__ __forceinline__ void load(const __half *p, float *o) {
    if constexpr (VEC == 4) {
      const uint2 u = __ldg(reinterpret_cast<const uint2 *>(p));
      const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u.x));
      const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
      o[0] = a.x;
      o[1] = a.y;
      o[2] = b.x;
      o[3] = b.y;
    } else {
      const unsigned u = __ldg(reinterpret_cast<const unsigned *>(p));
      const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u));
      o[0] = a.x;
      o[1] = a.y;
    }
  }
};

/* Block = (chunk, kv head x subgroup). Subgroup `sub` covers query heads
 * [sub*GS, sub*GS+GS) of the kv head's group (heads >= G are dead lanes of the
 * last subgroup). grid.x == 1 writes the final gated output into att. */
template <int HD, int GS, typename T>
__global__ void __launch_bounds__(kWarps * 32)
    attn_split_kernel(const float *__restrict__ qbuf, int qstride, int gated,
                      const uint8_t *__restrict__ kc,
                      const uint8_t *__restrict__ vc, size_t knb, size_t vnb,
                      int nh, int G, int nsub, int n_tok, int per, float scale,
                      float *__restrict__ wm, float *__restrict__ wl,
                      float *__restrict__ wacc, float *__restrict__ att,
                      unsigned *__restrict__ counters) {
  constexpr int VEC = HD >= 128 ? 4 : 2; // elements per lane per load
  constexpr int NJ = HD / (32 * VEC);    // loads per row per lane
  constexpr int U = kTokensPerStep;
  static_assert(NJ * 32 * VEC == HD, "head_dim must be 64, 128 or 256");

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int chunk = blockIdx.x;
  const int kh = blockIdx.y / nsub;
  const int g0 = (blockIdx.y % nsub) * GS;
  const int t_begin = chunk * per;
  const int t_end = min(n_tok, t_begin + per);

  __shared__ float s_m[kWarps][GS];
  __shared__ float s_l[kWarps][GS];
  __shared__ float s_acc[kWarps][GS][HD];

  // q fragments: lane owns dims VEC*lane + 32*VEC*j + e of each head
  float q[GS][NJ][VEC];
#pragma unroll
  for (int gi = 0; gi < GS; ++gi) {
    const int g = g0 + gi;
    const float *qh =
        qbuf + static_cast<size_t>(kh * G + (g < G ? g : 0)) * qstride;
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < VEC; ++e)
        q[gi][j][e] = g < G ? qh[VEC * lane + 32 * VEC * j + e] * scale : 0.0f;
  }

  float m[GS], l[GS], acc[GS][NJ][VEC];
#pragma unroll
  for (int gi = 0; gi < GS; ++gi) {
    m[gi] = -INFINITY;
    l[gi] = 0.0f;
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < VEC; ++e)
        acc[gi][j][e] = 0.0f;
  }

  const T *kb = reinterpret_cast<const T *>(kc) + static_cast<size_t>(kh) * HD +
                VEC * lane;
  const T *vb = reinterpret_cast<const T *>(vc) + static_cast<size_t>(kh) * HD +
                VEC * lane;
  const size_t knb_e = knb / sizeof(T), vnb_e = vnb / sizeof(T);

  for (int tb = t_begin + warp * U; tb < t_end; tb += kWarps * U) {
    float kk[U][NJ][VEC], vv[U][NJ][VEC];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int t = tb + u;
      if (t < t_end) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          LdVec<VEC, T>::load(
              kb + static_cast<size_t>(t) * knb_e + 32 * VEC * j, kk[u][j]);
          LdVec<VEC, T>::load(
              vb + static_cast<size_t>(t) * vnb_e + 32 * VEC * j, vv[u][j]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
          for (int e = 0; e < VEC; ++e) {
            kk[u][j][e] = 0.0f;
            vv[u][j][e] = 0.0f;
          }
      }
    }

    float s[GS][U];
#pragma unroll
    for (int gi = 0; gi < GS; ++gi)
#pragma unroll
      for (int u = 0; u < U; ++u) {
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
          for (int e = 0; e < VEC; ++e)
            a = fmaf(q[gi][j][e], kk[u][j][e], a);
        s[gi][u] = q35_warp_sum(a); // every lane now holds the full score
      }

#pragma unroll
    for (int gi = 0; gi < GS; ++gi) {
      float mx = -INFINITY;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (tb + u >= t_end)
          s[gi][u] = -INFINITY;
        mx = fmaxf(mx, s[gi][u]);
      }
      const float mn = fmaxf(m[gi], mx); // finite: token tb is always valid
      const float alpha = __expf(m[gi] - mn);
      float ps = 0.0f;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        s[gi][u] = __expf(s[gi][u] - mn);
        ps += s[gi][u];
      }
      l[gi] = l[gi] * alpha + ps;
      m[gi] = mn;
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
          float a = acc[gi][j][e] * alpha;
#pragma unroll
          for (int u = 0; u < U; ++u)
            a = fmaf(s[gi][u], vv[u][j][e], a);
          acc[gi][j][e] = a;
        }
    }
  }

  // merge the warps of this block
#pragma unroll
  for (int gi = 0; gi < GS; ++gi) {
    if (lane == 0) {
      s_m[warp][gi] = m[gi];
      s_l[warp][gi] = l[gi];
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int e = 0; e < VEC; ++e)
        s_acc[warp][gi][VEC * lane + 32 * VEC * j + e] = acc[gi][j][e];
  }
  __syncthreads();

  const bool direct = gridDim.x == 1;
  for (int idx = threadIdx.x; idx < GS * HD; idx += blockDim.x) {
    const int gi = idx / HD, d = idx % HD;
    const int g = g0 + gi;
    if (g >= G)
      continue;
    const int h = kh * G + g;
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
      M = fmaxf(M, s_m[w][gi]);
    float num = 0.0f, den = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const float e =
          __expf(s_m[w][gi] - M); // warps with no tokens: m = -inf -> 0
      num = fmaf(e, s_acc[w][gi][d], num);
      den = fmaf(e, s_l[w][gi], den);
    }
    if (direct) {
      float o = num / den;
      if (gated)
        o *= 1.0f /
             (1.0f + __expf(-qbuf[static_cast<size_t>(h) * qstride + HD + d]));
      att[static_cast<size_t>(h) * HD + d] = o;
    } else {
      wacc[(static_cast<size_t>(chunk) * nh + h) * HD + d] = num;
      if (d == 0) {
        wm[static_cast<size_t>(chunk) * nh + h] = M;
        wl[static_cast<size_t>(chunk) * nh + h] = den;
      }
    }
  }
  if (direct)
    return;

  // Split-K fold by the LAST block to finish this (kv head, subgroup): every writer
  // fences its partial, one thread takes a ticket (counters are zeroed by
  // attn_prep_kernel of this call), and the block holding the final ticket reads
  // all chunks' partials through L2 and writes the gated head outputs. Saves a
  // launch and its drain bubble over a separate combine kernel.
  __shared__ unsigned s_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    s_last = atomicAdd(&counters[blockIdx.y], 1u) == gridDim.x - 1;
  __syncthreads();
  if (!s_last)
    return;
  __threadfence();
  const int chunks = gridDim.x;
  for (int idx = threadIdx.x; idx < GS * HD; idx += blockDim.x) {
    const int gi = idx / HD, d = idx % HD;
    const int g = g0 + gi;
    if (g >= G)
      continue;
    const int h = kh * G + g;
    float M = -INFINITY;
    for (int c = 0; c < chunks; ++c)
      M = fmaxf(M, __ldcg(&wm[static_cast<size_t>(c) * nh + h]));
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < chunks; ++c) {
      const float e = __expf(__ldcg(&wm[static_cast<size_t>(c) * nh + h]) - M);
      num = fmaf(e, __ldcg(&wacc[(static_cast<size_t>(c) * nh + h) * HD + d]), num);
      den = fmaf(e, __ldcg(&wl[static_cast<size_t>(c) * nh + h]), den);
    }
    float o = num / den;
    if (gated)
      o *= 1.0f /
           (1.0f + __expf(-qbuf[static_cast<size_t>(h) * qstride + HD + d]));
    att[static_cast<size_t>(h) * HD + d] = o;
  }
}

template <int HD, typename T>
void launch_split(int GS, dim3 grid, const float *q, int qstride, int gated,
                  const uint8_t *kc, const uint8_t *vc, size_t knb, size_t vnb,
                  int nh, int G, int nsub, int n_tok, int per, float scale,
                  float *wm, float *wl, float *wacc, float *att,
                  unsigned *cnt, cudaStream_t st) {
#define SPLIT_CASE(N)                                                          \
  case N:                                                                      \
    attn_split_kernel<HD, N, T><<<grid, kWarps * 32, 0, st>>>(                 \
        q, qstride, gated, kc, vc, knb, vnb, nh, G, nsub, n_tok, per, scale,   \
        wm, wl, wacc, att, cnt);                                               \
    break;
  switch (GS) {
    SPLIT_CASE(1)
    SPLIT_CASE(2)
    SPLIT_CASE(3)
    SPLIT_CASE(4)
  }
#undef SPLIT_CASE
}

template <typename T>
void launch_split_hd(int hd, int GS, dim3 grid, const float *q, int qstride,
                     int gated, const uint8_t *kc, const uint8_t *vc,
                     size_t knb, size_t vnb, int nh, int G, int nsub, int n_tok,
                     int per, float scale, float *wm, float *wl, float *wacc,
                     float *att, unsigned *cnt, cudaStream_t st) {
  switch (hd) {
  case 64:
    launch_split<64, T>(GS, grid, q, qstride, gated, kc, vc, knb, vnb, nh, G,
                        nsub, n_tok, per, scale, wm, wl, wacc, att, cnt, st);
    break;
  case 128:
    launch_split<128, T>(GS, grid, q, qstride, gated, kc, vc, knb, vnb, nh, G,
                         nsub, n_tok, per, scale, wm, wl, wacc, att, cnt, st);
    break;
  default:
    launch_split<256, T>(GS, grid, q, qstride, gated, kc, vc, knb, vnb, nh, G,
                         nsub, n_tok, per, scale, wm, wl, wacc, att, cnt, st);
    break;
  }
}

int sm_count() {
  static int
      cache[64]; // idempotent benign race: every writer stores the same value
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64)
    return 80;
  if (!cache[dev]) {
    int n = 0;
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    cache[dev] = n > 0 ? n : 80;
  }
  return cache[dev];
}

bool norm_ok(const SpiteTensor *t, int hd) {
  return !t || (t->data && t->ne[0] == static_cast<uint32_t>(hd) &&
                (t->kind == SPITE_TYPE_F32 || t->kind == SPITE_TYPE_F16 ||
                 t->kind == SPITE_TYPE_BF16) &&
                (reinterpret_cast<uintptr_t>(t->data) & 3) == 0);
}

} // namespace

extern "C" int
qwen35_cuda_attention_ex(SpiteTensor *out, const SpiteTensor *x,
                         const SpiteTensor *wq, const SpiteTensor *wk,
                         const SpiteTensor *wv, const SpiteTensor *wo,
                         const SpiteTensor *q_norm, const SpiteTensor *k_norm,
                         float norm_eps, SpiteKvCache *kv, float rope_freq_base,
                         const SpiteAttnParams *params, const SpiteCtx *ctx) {
  if (!out || !x || !wq || !wk || !wv || !wo || !kv || !params || !ctx)
    return -1;
  const int nh = ctx->n_heads, nkv = ctx->n_kv_heads;
  const int hd = params->head_dim, rd = params->rope_dim;
  const bool gated = params->gated_q != 0;
  if (nh <= 0 || nkv <= 0 || nh % nkv != 0)
    return -1;
  if (hd != 64 && hd != 128 && hd != 256)
    return -1; // lane layout: 32 lanes x (2|4) elements x 1..2 loads
  if (rd < 0 || rd > hd || (rd & 1))
    return -1;
  if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32 || !x->data ||
      !out->data)
    return -1;
  const int G = nh / nkv;
  const int64_t kv_stride = static_cast<int64_t>(nkv) * hd;
  const int64_t q_rows = static_cast<int64_t>(nh) * hd * (gated ? 2 : 1);
  if (static_cast<int64_t>(wq->ne[1]) != q_rows ||
      static_cast<int64_t>(wk->ne[1]) != kv_stride ||
      static_cast<int64_t>(wv->ne[1]) != kv_stride ||
      static_cast<int64_t>(wo->ne[0]) != static_cast<int64_t>(nh) * hd ||
      static_cast<int64_t>(out->ne[0]) < static_cast<int64_t>(wo->ne[1]))
    return -1;
  if (wk->ne[0] != wq->ne[0] || wv->ne[0] != wq->ne[0] || x->ne[0] != wq->ne[0])
    return -1;
  if (!norm_ok(q_norm, hd) || !norm_ok(k_norm, hd))
    return -1;

  const SpiteType kind = kv->k.kind;
  if ((kind != SPITE_TYPE_F32 && kind != SPITE_TYPE_F16) ||
      kv->v.kind != kind || !kv->k.data || !kv->v.data ||
      static_cast<int64_t>(kv->k.ne[0]) != kv_stride ||
      static_cast<int64_t>(kv->v.ne[0]) != kv_stride)
    return -1;
  const size_t esz = kind == SPITE_TYPE_F32 ? 4 : 2;
  const size_t row_bytes = static_cast<size_t>(kv_stride) * esz;
  const int n_ctx = static_cast<int>(kv->k.ne[1]);
  if (kv->k.nb[1] < row_bytes || kv->v.nb[1] < row_bytes ||
      (kv->k.nb[1] & 15) || (kv->v.nb[1] & 15) ||
      (reinterpret_cast<uintptr_t>(kv->k.data) & 15) ||
      (reinterpret_cast<uintptr_t>(kv->v.data) & 15))
    return -1; // vector loads need 16-byte aligned rows
  const int pos = ctx->pos;
  const int m = x->ne[1] ? static_cast<int>(x->ne[1]) : 1;
  if (pos < 0 || pos + m > n_ctx || static_cast<int>(kv->v.ne[1]) != n_ctx)
    return -2;

  // scratch, m token columns: q|gate [q_rows,m] | att [nh*hd,m] | k | v | workspace
  const size_t q_floats = static_cast<size_t>(nh) * hd * (gated ? 2 : 1);
  const size_t att_floats = static_cast<size_t>(nh) * hd;
  const size_t fixed =
      (q_floats + att_floats + 2 * static_cast<size_t>(kv_stride)) * m;
  if (!ctx->scratchpad || ctx->scratchpad_bytes < sizeof(float) * fixed ||
      (reinterpret_cast<uintptr_t>(ctx->scratchpad) & 15))
    return -2;
  float *qg = static_cast<float *>(ctx->scratchpad);
  float *att = qg + q_floats * m;
  float *kbuf = att + att_floats * m;
  float *vbuf = kbuf + static_cast<size_t>(kv_stride) * m;
  float *ws = vbuf + static_cast<size_t>(kv_stride) * m;
  const size_t ws_floats = ctx->scratchpad_bytes / sizeof(float) - fixed;

  const cudaStream_t st = q35_stream(ctx);
  const float *xin = static_cast<const float *>(x->data);
  const int qstride = gated ? 2 * hd : hd;
  if (m == 1) {
    const Q35GemvJob proj[3] = {{wq, qg}, {wk, kbuf}, {wv, vbuf}};
    if (q35_gemv_multi(proj, 3, xin, false, st))
      return -1;
  } else {
    if (q35_gemv_batch(wq, xin, qg, m, false, st) ||
        q35_gemv_batch(wk, xin, kbuf, m, false, st) ||
        q35_gemv_batch(wv, xin, vbuf, m, false, st))
      return -1;
  }

  // query-head subgroups (<= kMaxGroupSub heads per block) and KV chunking
  const int nsub = (G + kMaxGroupSub - 1) / kMaxGroupSub;
  const int GS = (G + nsub - 1) / nsub;
  const int ncnt = nkv * nsub; // split-K arrival counters live behind the partials
  const int64_t cmax_ws = static_cast<int64_t>(
      (ws_floats > static_cast<size_t>(ncnt) ? ws_floats - ncnt : 0) /
      (static_cast<size_t>(nh) * (hd + 2)));
  // ~0.7 blocks per SM: measured on an RTX 5090 (170 SMs, hd=256, 4k context) the split
  // kernel takes 55/34/29/29/37/39 us at 24/48/80/120/170/340 blocks — one 8-warp block
  // per SM is already latency-limited by its own merge/ticket tail, and finer splits
  // pay that fixed cost more often.
  const int want_blocks = std::max(1, sm_count() * 7 / 10);
  // workspace sized for the widest token (the last position)
  int chunks_max = (pos + m + kMinChunk - 1) / kMinChunk;
  chunks_max = std::min(chunks_max, std::max(1, (want_blocks + ncnt - 1) / ncnt));
  chunks_max = static_cast<int>(std::min<int64_t>(chunks_max, std::max<int64_t>(1, cmax_ws)));
  // workspace: m[chunks*nh] l[chunks*nh] acc[chunks*nh*hd] | counters[ncnt]
  float *wm = ws;
  float *wl = wm + static_cast<size_t>(chunks_max) * nh;
  float *wacc = wl + static_cast<size_t>(chunks_max) * nh;
  unsigned *cnt = reinterpret_cast<unsigned *>(wacc + static_cast<size_t>(chunks_max) * nh * hd);
  if (chunks_max > 1 && static_cast<size_t>(chunks_max) * nh * (hd + 2) + ncnt > ws_floats)
    return -2; // cannot happen (cmax_ws), kept as a guard against future edits

  const void *qn = q_norm ? q_norm->data : nullptr,
             *kn = k_norm ? k_norm->data : nullptr;
  const int qnk = q_norm ? q_norm->kind : 0, knk = k_norm ? k_norm->kind : 0;
  const float scale = rsqrtf(static_cast<float>(hd));
  const uint8_t *kc = static_cast<const uint8_t *>(kv->k.data);
  const uint8_t *vc = static_cast<const uint8_t *>(kv->v.data);
  const double log2base = log2(static_cast<double>(rope_freq_base));

  // One token column at a time: its KV row at `p`, attending causally over
  // every row up to `p`. Only the projections above are batched.
  for (int t = 0; t < m; ++t) {
    const int p = pos + t;
    const int n_tok = p + 1;
    int chunks = (n_tok + kMinChunk - 1) / kMinChunk;
    chunks = std::min(chunks, std::max(1, (want_blocks + ncnt - 1) / ncnt));
    chunks = static_cast<int>(std::min<int64_t>(chunks, std::max<int64_t>(1, cmax_ws)));
    const int per = (n_tok + chunks - 1) / chunks;
    chunks = (n_tok + per - 1) / per; // chunks actually populated

    uint8_t *k_row = static_cast<uint8_t *>(kv->k.data) +
                     static_cast<size_t>(p) * kv->k.nb[1];
    uint8_t *v_row = static_cast<uint8_t *>(kv->v.data) +
                     static_cast<size_t>(p) * kv->v.nb[1];
    float *qg_t = qg + static_cast<size_t>(t) * q_floats;
    float *kb_t = kbuf + static_cast<size_t>(t) * kv_stride;
    float *vb_t = vbuf + static_cast<size_t>(t) * kv_stride;
    const int n_zero = chunks > 1 ? ncnt : 0;
    if (kind == SPITE_TYPE_F32)
      attn_prep_kernel<float><<<nh + nkv, 32, 0, st>>>(
          qg_t, kb_t, vb_t, reinterpret_cast<float *>(k_row),
          reinterpret_cast<float *>(v_row), qn, qnk, kn, knk, norm_eps, hd, rd,
          qstride, p, log2base, nh, cnt, n_zero);
    else
      attn_prep_kernel<__half><<<nh + nkv, 32, 0, st>>>(
          qg_t, kb_t, vb_t, reinterpret_cast<__half *>(k_row),
          reinterpret_cast<__half *>(v_row), qn, qnk, kn, knk, norm_eps, hd, rd,
          qstride, p, log2base, nh, cnt, n_zero);

    const dim3 grid(chunks, ncnt);
    float *att_t = att + static_cast<size_t>(t) * att_floats;
    if (kind == SPITE_TYPE_F32)
      launch_split_hd<float>(hd, GS, grid, qg_t, qstride, gated, kc, vc,
                             kv->k.nb[1], kv->v.nb[1], nh, G, nsub, n_tok, per,
                             scale, wm, wl, wacc, att_t, cnt, st);
    else
      launch_split_hd<__half>(hd, GS, grid, qg_t, qstride, gated, kc, vc,
                              kv->k.nb[1], kv->v.nb[1], nh, G, nsub, n_tok, per,
                              scale, wm, wl, wacc, att_t, cnt, st);
  }

  if (m == 1) {
    if (q35_gemv(wo, att, static_cast<float *>(out->data), true, st))
      return -1;
  } else if (q35_gemv_batch(wo, att, static_cast<float *>(out->data), m, true, st)) {
    return -1;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
