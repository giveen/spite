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

/* ── Split-K workspace ────────────────────────────────────────────────────── */

/* One partial result per (chunk × head). */
struct KvFlashPartial {
    float m;   /* running max */
    float l;   /* running sum-exp denominator */
    /* acc[head_dim] follows immediately; layout: [chunks, n_heads, 1+1+head_dim] */
};

/* ── Portable tile kernel ─────────────────────────────────────────────────── */

#ifndef SPITE_KVFLASH_ARCH

/*
 * One block = one KV range × one GQA group.
 *   gridDim.x  = n_chunks
 *   gridDim.y  = n_kv_heads
 *   blockDim.x = KVFLASH_TILE   (one lane = one KV position in the tile)
 *   blockDim.y = KVFLASH_WARPS  (one warp = one query head in the group)
 *
 * Shared memory layout:
 *   k_tile[KVFLASH_TILE][head_dim]  f32 (dequantised KV)
 *   v_tile[KVFLASH_TILE][head_dim]  f32
 */
__global__ void kvflash_tile(
        const KvattnArgs* __restrict__ ga,
        int chunks, int per,
        float* __restrict__ acc_out,  /* [chunks, n_heads, head_dim] */
        float* __restrict__ m_out,    /* [chunks, n_heads]           */
        float* __restrict__ l_out)    /* [chunks, n_heads]           */
{
    extern __shared__ float smem[];  /* k_tile + v_tile */
    const KvattnArgs& a = *ga;

    const int chunk   = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int q_in_grp = threadIdx.y;  /* which query head within the GQA group */
    const int grp_sz  = a.n_heads / a.n_kv_heads;
    const int q_head  = kv_head * grp_sz + q_in_grp;
    if (q_head >= a.n_heads) return;

    const int t_start = chunk * per;
    const int t_end   = min(t_start + per, a.n_tok);
    if (t_start >= a.n_tok) return;

    float* k_tile = smem;
    float* v_tile = smem + KVFLASH_TILE * a.head_dim;

    const float* qh = a.q + (size_t)q_head * a.head_dim;
    float* acc = acc_out + ((size_t)chunk * a.n_heads + q_head) * a.head_dim;
    float* mp  = m_out   +  (size_t)chunk * a.n_heads + q_head;
    float* lp  = l_out   +  (size_t)chunk * a.n_heads + q_head;

    float m_cur = -INFINITY, l_cur = 0.0f;
    for (int i = threadIdx.x; i < a.head_dim; i += blockDim.x) acc[i] = 0.0f;

    for (int t = t_start; t < t_end; t += KVFLASH_TILE) {
        const int tile_len = min(KVFLASH_TILE, t_end - t);

        /* Load K tile into shared memory (one lane per KV step). */
        if (threadIdx.y == 0 && threadIdx.x < tile_len) {
            const int tt = t + threadIdx.x;
            const size_t koff = ((size_t)tt * a.n_kv_heads + kv_head);
            const void* krow = static_cast<const char*>(a.k_cache) +
                               koff * kvq_row_bytes(a.kv_kind, a.head_dim);
            for (int i = 0; i < a.head_dim; ++i)
                k_tile[threadIdx.x * a.head_dim + i] = kvq_get(krow, a.kv_kind, i);
        }
        __syncthreads();

        /* Score each KV step in the tile. */
        float scores_local[KVFLASH_TILE];
        for (int k = 0; k < tile_len; ++k) {
            float dot = 0.0f;
            for (int i = threadIdx.x; i < a.head_dim; i += blockDim.x)
                dot += qh[i] * k_tile[k * a.head_dim + i];
            dot = warp_sum(dot);
            scores_local[k] = dot * a.scale;
        }

        /* Online softmax update. */
        float m_new = m_cur;
        for (int k = 0; k < tile_len; ++k) m_new = fmaxf(m_new, scores_local[k]);
        const float corr = __expf(m_cur - m_new);
        l_cur *= corr;
        for (int i = threadIdx.x; i < a.head_dim; i += blockDim.x) acc[i] *= corr;
        m_cur = m_new;

        /* Load V tile and accumulate. */
        if (threadIdx.y == 0 && threadIdx.x < tile_len) {
            const int tt = t + threadIdx.x;
            const size_t voff = ((size_t)tt * a.n_kv_heads + kv_head);
            const void* vrow = static_cast<const char*>(a.v_cache) +
                               voff * kvq_row_bytes(a.kv_kind, a.head_dim);
            for (int i = 0; i < a.head_dim; ++i)
                v_tile[threadIdx.x * a.head_dim + i] = kvq_get(vrow, a.kv_kind, i);
        }
        __syncthreads();

        for (int k = 0; k < tile_len; ++k) {
            const float w = __expf(scores_local[k] - m_cur);
            l_cur += w;
            for (int i = threadIdx.x; i < a.head_dim; i += blockDim.x)
                acc[i] += w * v_tile[k * a.head_dim + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) { *mp = m_cur; *lp = l_cur; }
}

static bool kvflash_hd_supported(int /*hd*/) { return true; }

static int kvflash_launch_hd(const KvattnArgs& a, int chunks, int per,
                              float* acc, float* m_out, float* l_out,
                              cudaStream_t s) {
    const size_t smem = sizeof(float) * KVFLASH_TILE * a.head_dim * 2;
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
        const float w = __expf(m_p[(size_t)c * gridDim.x + h] - m_max);
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
