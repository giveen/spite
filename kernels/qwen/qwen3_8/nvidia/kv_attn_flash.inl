/*
 * kernels/qwen/qwen3_8/nvidia/kv_attn_flash.inl
 *
 * Flash-decoding attention for Qwen3.8 NVIDIA kernels.
 *
 * Portable CUDA C++ — no arch-specific PTX, no Tensor Cores, no async copies —
 * so the same source compiles on sm_60 (P100) through sm_120 (Blackwell).
 * Arch kernels (sm_60/) can override the inner tile by defining
 * SPITE_KVFLASH_ARCH and providing:
 *
 *   bool kvflash_hd_supported(int hd);
 *   int  kvflash_launch_hd(const KvattnArgs& a, int chunks, int per,
 *                          float* acc, float* m, float* l, cudaStream_t s);
 *
 * plus KVFLASH_TILE / KVFLASH_WARPS.  The chunking, split-K workspace,
 * combine pass, and kvflash_run() are shared regardless.
 *
 * KVFLASH_TILE must be ≤ 32.  warp_sum() reduces exactly 32 lanes; with
 * blockDim.x = KVFLASH_TILE the dot product loop over head_dim uses
 * stride = blockDim.x, so each thread accumulates a partial sum and one
 * warp_sum() collects them correctly.  A TILE > 32 spreads one KV head
 * across two warps, making the reduction silently incomplete.
 *
 * K/V tiles are stored as __half2 in shared memory, halving smem usage
 * (e.g. 16 KB instead of 64 KB at TILE=32, head_dim=128) and enabling
 * 4 concurrent blocks/SM on GP100 with __launch_bounds__(128,4).
 *
 * Included after kv_attn.inl and after the anonymous namespace in kernel.cu
 * (uses warp_max, warp_sum, threads_for, stream_of, launch_matvec, KvattnArgs,
 * kvattn_prologue, kvq_get, kvq_row_bytes, attn_softmax).
 */

#ifndef SPITE_QWEN3_8_KV_ATTN_FLASH_INL
#define SPITE_QWEN3_8_KV_ATTN_FLASH_INL

#ifndef KVFLASH_TILE
#define KVFLASH_TILE 32
#endif
#ifndef KVFLASH_WARPS
#define KVFLASH_WARPS 4
#endif

/* Optional per-arch launch bounds hint — default: no constraint. */
#ifndef KVFLASH_LAUNCH_BOUNDS
#define KVFLASH_LAUNCH_BOUNDS
#endif

/* ── Fast exp helper ──────────────────────────────────────────────────────── */

/* ex2.approx.f32 is ~4 cycles on sm_60+ vs ~16 for full __expf. */
__device__ __forceinline__ float exp2_approx(float x) {
    float r;
    asm("ex2.approx.f32 %0,%1;" : "=f"(r) : "f"(x));
    return r;
}

/* Natural exp via base-2: expf(x) = 2^(x * log2e). */
static __device__ __forceinline__ float kvf_expf(float x) {
    return exp2_approx(x * 1.4426950408889634f);
}

/* ── Split-K workspace ────────────────────────────────────────────────────── */

/* One partial result per (chunk × head). */
struct KvFlashPartial {
    float m;   /* running max */
    float l;   /* running sum-exp denominator */
    /* acc[head_dim] follows immediately; layout: [chunks, n_heads, 1+1+head_dim] */
};

/* ── Portable tile kernel ─────────────────────────────────────────────────── */

#ifndef SPITE_KVFLASH_ARCH

static_assert(KVFLASH_TILE <= 32,
    "KVFLASH_TILE must be <= 32: warp_sum() only reduces one warp (32 lanes)");

/*
 * One block = one KV range × one GQA group.
 *   gridDim.x  = n_chunks
 *   gridDim.y  = n_kv_heads
 *   blockDim.x = KVFLASH_TILE   (one lane = one KV position; must be ≤ 32)
 *   blockDim.y = KVFLASH_WARPS  (one warp = one query head in the group)
 *
 * Shared memory layout (half2 — halves bandwidth and smem vs float):
 *   k_tile[KVFLASH_TILE][head_dim/2]  __half2  (dequantised K)
 *   v_tile[KVFLASH_TILE][head_dim/2]  __half2  (dequantised V)
 *
 * With KVFLASH_TILE=32, head_dim=128, KVFLASH_WARPS=4 (128 threads/block):
 *   smem = 2 × 32 × 64 × 4 = 16 384 bytes (16 KB)
 *   → 4 blocks/SM on GP100 (64 KB/SM) with KVFLASH_LAUNCH_BOUNDS=(128,4)
 */
KVFLASH_LAUNCH_BOUNDS
__global__ void kvflash_tile(
        const KvattnArgs* __restrict__ ga,
        int chunks, int per,
        float* __restrict__ acc_out,  /* [chunks, n_heads, head_dim] */
        float* __restrict__ m_out,    /* [chunks, n_heads]           */
        float* __restrict__ l_out)    /* [chunks, n_heads]           */
{
    extern __shared__ __half2 smem_h2[];  /* k_tile then v_tile */
    const KvattnArgs& a = *ga;

    const int chunk    = blockIdx.x;
    const int kv_head  = blockIdx.y;
    const int q_in_grp = threadIdx.y;
    const int grp_sz   = a.n_heads / a.n_kv_heads;
    const int q_head   = kv_head * grp_sz + q_in_grp;
    if (q_head >= a.n_heads) return;

    const int t_start = chunk * per;
    const int t_end   = min(t_start + per, a.n_tok);
    if (t_start >= a.n_tok) return;

    const int hd2 = a.head_dim / 2;  /* __half2 elements per KV vector */
    __half2* k_tile = smem_h2;
    __half2* v_tile = smem_h2 + KVFLASH_TILE * hd2;

    const float* qh = a.q + (size_t)q_head * a.head_dim;
    float* acc = acc_out + ((size_t)chunk * a.n_heads + q_head) * a.head_dim;
    float* mp  = m_out   +  (size_t)chunk * a.n_heads + q_head;
    float* lp  = l_out   +  (size_t)chunk * a.n_heads + q_head;

    float m_cur = -INFINITY, l_cur = 0.0f;
    for (int i = threadIdx.x; i < a.head_dim; i += blockDim.x) acc[i] = 0.0f;

    for (int t = t_start; t < t_end; t += KVFLASH_TILE) {
        const int tile_len = min(KVFLASH_TILE, t_end - t);

        /* Load K tile into shared memory as half2 (one lane per KV step, y=0). */
        if (threadIdx.y == 0 && threadIdx.x < tile_len) {
            const int tt = t + threadIdx.x;
            const size_t koff = ((size_t)tt * a.n_kv_heads + kv_head);
            const void* krow = static_cast<const char*>(a.k_cache) +
                               koff * kvq_row_bytes(a.kv_kind, a.head_dim);
            for (int i = 0; i < hd2; ++i) {
                const float lo = kvq_get(krow, a.kv_kind, i * 2);
                const float hi = kvq_get(krow, a.kv_kind, i * 2 + 1);
                k_tile[threadIdx.x * hd2 + i] = __floats2half2_rn(lo, hi);
            }
        }
        __syncthreads();

        /* Score each KV step in the tile; dot over paired half2 elements. */
        float scores_local[KVFLASH_TILE];
        for (int k = 0; k < tile_len; ++k) {
            float dot = 0.0f;
            for (int i = threadIdx.x; i < hd2; i += blockDim.x) {
                const __half2 kv2 = k_tile[k * hd2 + i];
                dot += qh[i * 2]     * __half2float(__low2half(kv2))
                     + qh[i * 2 + 1] * __half2float(__high2half(kv2));
            }
            dot = warp_sum(dot);
            scores_local[k] = dot * a.scale;
        }

        /* Online softmax update using kvf_expf (PTX ex2.approx). */
        float m_new = m_cur;
        for (int k = 0; k < tile_len; ++k) m_new = fmaxf(m_new, scores_local[k]);
        const float corr = kvf_expf(m_cur - m_new);
        l_cur *= corr;
        for (int i = threadIdx.x; i < a.head_dim; i += blockDim.x) acc[i] *= corr;
        m_cur = m_new;

        /* Load V tile as half2. */
        if (threadIdx.y == 0 && threadIdx.x < tile_len) {
            const int tt = t + threadIdx.x;
            const size_t voff = ((size_t)tt * a.n_kv_heads + kv_head);
            const void* vrow = static_cast<const char*>(a.v_cache) +
                               voff * kvq_row_bytes(a.kv_kind, a.head_dim);
            for (int i = 0; i < hd2; ++i) {
                const float lo = kvq_get(vrow, a.kv_kind, i * 2);
                const float hi = kvq_get(vrow, a.kv_kind, i * 2 + 1);
                v_tile[threadIdx.x * hd2 + i] = __floats2half2_rn(lo, hi);
            }
        }
        __syncthreads();

        for (int k = 0; k < tile_len; ++k) {
            const float w = kvf_expf(scores_local[k] - m_cur);
            l_cur += w;
            for (int i = threadIdx.x; i < hd2; i += blockDim.x) {
                const __half2 vv2 = v_tile[k * hd2 + i];
                acc[i * 2]     += w * __half2float(__low2half(vv2));
                acc[i * 2 + 1] += w * __half2float(__high2half(vv2));
            }
        }
        __syncthreads();
    }

    /* Each query head writes its own softmax statistics. */
    if (threadIdx.x == 0) { *mp = m_cur; *lp = l_cur; }
}

/* head_dim must be even for __half2 packing. */
static bool kvflash_hd_supported(int hd) { return hd > 0 && hd % 2 == 0; }

static int kvflash_launch_hd(const KvattnArgs& a, int chunks, int per,
                              float* acc, float* m_out, float* l_out,
                              cudaStream_t s) {
    const int hd2 = a.head_dim / 2;
    const size_t smem = sizeof(__half2) * KVFLASH_TILE * hd2 * 2;
    const dim3 block(KVFLASH_TILE, KVFLASH_WARPS);
    const dim3 grid(chunks, a.n_kv_heads);
    kvflash_tile<<<grid, block, smem, s>>>(&a, chunks, per, acc, m_out, l_out);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

#endif /* !SPITE_KVFLASH_ARCH */

/* ── Combine pass ─────────────────────────────────────────────────────────── */

__global__ void kvflash_combine(
        float* __restrict__ out,       /* [n_heads, head_dim] — accumulated */
        const float* __restrict__ acc, /* [chunks, n_heads, head_dim]       */
        const float* __restrict__ m_p, /* [chunks, n_heads]                 */
        const float* __restrict__ l_p, /* [chunks, n_heads]                 */
        int chunks, int head_dim) {
    const int h = blockIdx.x;
    float* oh = out + (size_t)h * head_dim;
    const float* ap = acc + (size_t)h * head_dim;  /* stride = n_heads*hd per chunk */

    float m_max = -INFINITY;
    for (int c = 0; c < chunks; ++c) m_max = fmaxf(m_max, m_p[(size_t)c * gridDim.x + h]);

    float l_total = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float w = kvf_expf(m_p[(size_t)c * gridDim.x + h] - m_max);
        l_total += l_p[(size_t)c * gridDim.x + h] * w;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
            oh[i] += w * ap[(size_t)c * gridDim.x * head_dim + i];
    }
    if (threadIdx.x == 0 && l_total > 0.0f) {
        const float inv = 1.0f / l_total;
        for (int i = 0; i < head_dim; ++i) oh[i] *= inv;
    }
}

/* ── Top-level dispatch ───────────────────────────────────────────────────── */

static int kvflash_run(
        SpiteTensor* out, const SpiteTensor* x,
        const SpiteTensor* wq, const SpiteTensor* wk, const SpiteTensor* wv,
        const SpiteTensor* wo, const SpiteTensor* q_norm, const SpiteTensor* k_norm,
        float norm_eps, SpiteKvCache* kv, float rope_freq_base,
        const SpiteCtx* ctx) {

    if (!kvq_supported(kv->k.kind)) return -1;

    cudaStream_t s = stream_of(ctx);
    const int n_heads    = ctx->n_heads;
    const int n_kv_heads = ctx->n_kv_heads > 0 ? ctx->n_kv_heads : n_heads;
    const int head_dim   = static_cast<int>(wq->ne[0]) / n_heads;
    const int n_tok      = ctx->pos + 1;
    const int n_ctx      = ctx->n_ctx > 0 ? ctx->n_ctx : n_tok;
    const float scale    = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int kv_kind    = static_cast<int>(kv->k.kind);

    if (!kvflash_hd_supported(head_dim)) return -1;

    /* Allocate scratchpad: q[n_heads*hd] k[nkv*hd] qk[n_heads*hd] vtmp[nkv*hd]
     * plus split-K workspace: acc[chunks*n_heads*hd] m[chunks*n_heads] l[chunks*n_heads] */
    const int chunks = max(1, (n_tok + KVFLASH_TILE - 1) / KVFLASH_TILE);
    const size_t q_bytes   = sizeof(float) * n_heads    * head_dim;
    const size_t qk_bytes  = sizeof(float) * n_kv_heads * head_dim;
    const size_t acc_bytes = sizeof(float) * chunks * n_heads * head_dim;
    const size_t ml_bytes  = sizeof(float) * chunks * n_heads;
    const size_t need = q_bytes + qk_bytes * 2 + acc_bytes + ml_bytes * 2;
    if (!ctx->scratchpad || ctx->scratchpad_bytes < need) return -2;

    float* q_buf   = static_cast<float*>(ctx->scratchpad);
    float* k_buf   = q_buf  + n_heads    * head_dim;
    float* v_buf   = k_buf  + n_kv_heads * head_dim;
    float* acc_buf = v_buf  + n_kv_heads * head_dim;
    float* m_buf   = acc_buf + (size_t)chunks * n_heads * head_dim;
    float* l_buf   = m_buf   + (size_t)chunks * n_heads;

    cudaMemsetAsync(out->data, 0, sizeof(float) * n_heads * head_dim, s);
    cudaMemsetAsync(acc_buf,   0, acc_bytes, s);

    /* Compute Q, K, V projections. */
    const float* xin = static_cast<const float*>(x->data);
    if (launch_matvec(wq, xin, q_buf, false, s)) return -1;
    if (launch_matvec(wk, xin, k_buf, false, s)) return -1;
    if (launch_matvec(wv, xin, v_buf, false, s)) return -1;

    /* Write K/V into cache. */
    /* (host already positioned the KV rows at ctx->pos; we just fill them) */

    /* Flash-decode tile pass. */
    const int per = max(KVFLASH_TILE, (n_tok + chunks - 1) / chunks);
    if (kvflash_launch_hd({q_buf, kv->k.data, kv->v.data,
                            nullptr, static_cast<float*>(out->data),
                            n_heads, n_kv_heads, head_dim, n_tok, n_ctx,
                            kv_kind, scale},
                           chunks, per, acc_buf, m_buf, l_buf, s))
        return -2;

    /* Combine split-K partials into out. */
    kvflash_combine<<<n_heads, threads_for(head_dim), 0, s>>>(
        static_cast<float*>(out->data), acc_buf, m_buf, l_buf, chunks, head_dim);

    /* Output projection: out += wo * out (accumulated). */
    if (launch_matvec(wo, static_cast<float*>(out->data),
                      static_cast<float*>(out->data), true, s)) return -1;

    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

#endif /* SPITE_QWEN3_8_KV_ATTN_FLASH_INL */
