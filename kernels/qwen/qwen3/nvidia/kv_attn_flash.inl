/*
 * kernels/qwen/qwen3/nvidia/kv_attn_flash.inl
 *
 * Flash-decoding attention for the Qwen3 NVIDIA kernels — the portable back end
 * for the vendor root, i.e. every CUDA architecture this tree builds for
 * (sm_75 Turing through sm_120 Blackwell).  Nothing here is arch-specific: no
 * 128-bit vector types, no async copies, no dynamic shared memory, no
 * assumption beyond what CUDA C++ guarantees on all of them, so the same
 * source is what every NVIDIA card runs.
 *
 * Why this exists (measured on RTX 5090, nh=32/nkv=8/hd=128, n_tok=4096):
 * the portable VBR back end in kv_attn.inl spends 72% of the op in
 * `kvattn_weighted_v_f32` — one block per query head (32 blocks on 170 SMs),
 * each thread walking the whole KV history one dependent FMA at a time.  Two
 * properties of the decode shape are worth exploiting instead:
 *
 *   1. Every query head in a GQA group reads the same K/V rows, so the VBR
 *      cache is currently dequantized `group` times over.  KV is staged in
 *      shared memory once per KV head and shared by the whole group.
 *   2. The score vector does not need to be materialized.  Scores are consumed
 *      where they are produced (online softmax), which removes the
 *      nh*n_ctx-float buffer and the three passes over it.
 *
 * Decode-only model — the host issues one token per call — so this is
 * FlashDecoding rather than a q-tiled FlashAttention: the KV axis is split
 * across blocks (one block = one KV head x one KV range), each block writes its
 * partial (m, l, acc) and kvflash_combine() folds them together.  Splitting the
 * KV axis is what buys the parallelism; sharing the tile across the group is
 * what buys the bandwidth.
 *
 * Included at global scope from a kernel.cu, after kv_attn.inl (it uses
 * KvattnArgs/kvattn_prologue/kvq_get/kvq_row_bytes) and after the file's
 * anonymous namespace (it uses warp_max, warp_sum, threads_for, stream_of and
 * launch_matvec, exactly like kv_attn.inl does).
 *
 * An arch kernel may supply its own tile kernel instead: define
 * SPITE_KVFLASH_ARCH, then provide (before including this file)
 *
 *   bool kvflash_hd_supported(int hd);
 *   int  kvflash_launch_hd(const KvattnArgs& a, int chunks, int per,
 *                          float* acc, float* m, float* l, cudaStream_t s);
 *
 * plus KVFLASH_TILE / KVFLASH_WARPS for its own tile geometry (the scaffolding
 * reads those macros for its chunking heuristic).  The portable tile kernel and
 * dispatch below are then not emitted, and the chunking, the split-K workspace,
 * the combine pass and kvflash_run() stay shared — so the levels cannot drift
 * on anything except the tile kernel itself.
 */

#ifndef SPITE_QWEN3_KV_ATTN_FLASH_INL
#define SPITE_QWEN3_KV_ATTN_FLASH_INL

/* KV rows staged per tile.  A tile is one warp-wide score step (lane == KV
 * index inside the tile), so it cannot be smaller than a warp. */
#ifndef KVFLASH_TILE
#define KVFLASH_TILE 32
#endif
/* Query heads handled per block (one warp each).  Qwen3-8B has 4 query heads
 * per KV head, so the default covers a whole GQA group in one pass. */
#ifndef KVFLASH_WARPS
#define KVFLASH_WARPS 4
#endif
/* Keep every chunk at least this many tiles wide, so splitting the KV axis for
 * occupancy does not degenerate into one row per block. */
constexpr int KVFLASH_TILES_MIN = 4;

/* The split-K workspace is carved out of the region kvattn_prologue() reserved
 * for the VBR score vector: that is nh*n_ctx floats, which is what let this fit
 * without changing the host's scratchpad budget.  Layout (floats), `acc` first
 * because a tile kernel may store it with 128-bit writes:
 *   acc [chunks*nh*hd]  m [chunks*nh]  l [chunks*nh]
 * chunks <= n_ctx/(hd+2) keeps the sum inside the scores reservation. */

#ifndef SPITE_KVFLASH_ARCH
/* One warp per query head of the GQA group; the block loads a KV tile once for
 * the whole group, so the VBR decode cost is paid once per KV head instead of
 * once per query head. */
template <int HD>
__global__ void kvflash_core_portable(const float* __restrict__ q,
                                      const uint8_t* __restrict__ kc,
                                      const uint8_t* __restrict__ vc, int row_bytes, int kind,
                                      int nh, int group, int n_tok, int per, float scale,
                                      float* __restrict__ wm, float* __restrict__ wl,
                                      float* __restrict__ wacc) {
    constexpr int NLANE = (HD + 31) / 32;  /* output dims owned by one lane */
    const int lane = threadIdx.x;
    const int w = threadIdx.y;
    const int nwarps = blockDim.y;
    const int gchunks = (group + KVFLASH_WARPS - 1) / KVFLASH_WARPS;

    /* +1 word of row padding: the score phase reads one column across 32 rows,
     * and an unpadded row stride is a multiple of 32 words, i.e. a 32-way bank
     * conflict for every single load. */
    __shared__ float s_q[KVFLASH_WARPS][HD];
    __shared__ float s_k[KVFLASH_TILE][HD + 1];
    __shared__ float s_v[KVFLASH_TILE][HD + 1];
    __shared__ float s_p[KVFLASH_WARPS][KVFLASH_TILE];

    const int kh = blockIdx.y / gchunks;
    const int hw = blockIdx.y % gchunks * (int)nwarps + w;
    const bool live = hw < group;
    const int h = kh * group + hw;

    const int t_begin = blockIdx.x * per;
    const int t_end = min(n_tok, t_begin + per);

    if (live)
        for (int i = lane; i < HD; i += 32) s_q[w][i] = q[static_cast<size_t>(h) * HD + i];
    __syncwarp();

    float m = -INFINITY, l = 0.0f;
    float acc[NLANE];
#pragma unroll
    for (int e = 0; e < NLANE; ++e) acc[e] = 0.0f;

    for (int t0 = t_begin; t0 < t_end; t0 += KVFLASH_TILE) {
        /* Stage K and V for this tile, decoded to float, once for the group.
         * Warp w owns rows w, w+nwarps, ... and the lanes walk a row, so the
         * global side is coalesced and the smem side is conflict-free. */
        for (int t = w; t < KVFLASH_TILE; t += nwarps) {
            const int gt = t0 + t;
            const bool ok = gt < t_end;
            const uint8_t* kr = kc + static_cast<size_t>(gt) * row_bytes;
            const uint8_t* vr = vc + static_cast<size_t>(gt) * row_bytes;
#pragma unroll 4
            for (int i = lane; i < HD; i += 32) {
                s_k[t][i] = ok ? kvq_get(kr, kind, kh * HD + i) : 0.0f;
                s_v[t][i] = ok ? kvq_get(vr, kind, kh * HD + i) : 0.0f;
            }
        }
        __syncthreads();

        /* Scores: lane == position inside the tile, so q.k_t is a per-lane FMA
         * chain with no shuffle tree (the VBR back end needs 5 shuffles per
         * 4 FMAs here).  Four accumulators keep the chain off the critical
         * path. */
        const int gt = t0 + lane;
        float s = -INFINITY;
        if (live && gt < t_end) {
            const float* krow = s_k[lane];
            const float* qrow = s_q[w];
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            int i = 0;
#pragma unroll 4
            for (; i + 4 <= HD; i += 4) {
                a0 = fmaf(qrow[i + 0], krow[i + 0], a0);
                a1 = fmaf(qrow[i + 1], krow[i + 1], a1);
                a2 = fmaf(qrow[i + 2], krow[i + 2], a2);
                a3 = fmaf(qrow[i + 3], krow[i + 3], a3);
            }
            for (; i < HD; ++i) a0 = fmaf(qrow[i], krow[i], a0);
            s = (a0 + a1 + a2 + a3) * scale;
        }

        /* Online softmax: rescale the running accumulator instead of writing
         * the scores out and walking them again. */
        const float m_new = fmaxf(m, warp_max(s));
        const float alpha = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * alpha + warp_sum(p);
        m = m_new;
#pragma unroll
        for (int e = 0; e < NLANE; ++e) acc[e] *= alpha;
        if (live) s_p[w][lane] = p;
        __syncwarp();

        /* P.V: the lane owns output dims now, so every staged V element is used
         * once per (head, tile) rather than once per token. */
        if (live) {
            for (int t = 0; t < KVFLASH_TILE; ++t) {
                if (t0 + t >= t_end) break;
                const float pv = s_p[w][t];
                const float* vrow = s_v[t];
#pragma unroll
                for (int e = 0; e < NLANE; ++e)
                    acc[e] = fmaf(pv, vrow[lane + 32 * e], acc[e]);
            }
        }
        __syncthreads();
    }

    /* Hand the partial (m, l, acc) to the combine pass.  A chunk that falls
     * past the end of the KV history contributes nothing: m stays -inf, so its
     * combine weight exp(m - M) is 0. */
    if (live) {
        if (lane == 0) {
            wm[static_cast<size_t>(blockIdx.x) * nh + h] = m;
            wl[static_cast<size_t>(blockIdx.x) * nh + h] = l;
        }
#pragma unroll
        for (int e = 0; e < NLANE; ++e) {
            const int d = lane + 32 * e;
            if (d < HD)
                wacc[(static_cast<size_t>(blockIdx.x) * nh + h) * HD + d] = acc[e];
        }
    }
}

inline bool kvflash_hd_supported(int hd) {
    return hd == 32 || hd == 64 || hd == 80 || hd == 96 || hd == 128;
}

template <int HD>
inline int kvflash_launch_tile(const KvattnArgs& a, int chunks, int per, float* acc, float* m,
                               float* l, cudaStream_t s) {
    const int gchunks = (a.group + KVFLASH_WARPS - 1) / KVFLASH_WARPS;
    const dim3 grid(chunks, a.nkv * gchunks);
    const dim3 block(32, KVFLASH_WARPS);
    kvflash_core_portable<HD><<<grid, block, 0, s>>>(a.q, a.kc, a.vc, a.row_bytes, a.kind, a.nh,
                                                     a.group, a.n_tok, per, a.scale, m, l, acc);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

inline int kvflash_launch_hd(const KvattnArgs& a, int chunks, int per, float* acc, float* m,
                             float* l, cudaStream_t s) {
    switch (a.hd) {
    case 32: return kvflash_launch_tile<32>(a, chunks, per, acc, m, l, s);
    case 64: return kvflash_launch_tile<64>(a, chunks, per, acc, m, l, s);
    case 80: return kvflash_launch_tile<80>(a, chunks, per, acc, m, l, s);
    case 96: return kvflash_launch_tile<96>(a, chunks, per, acc, m, l, s);
    case 128: return kvflash_launch_tile<128>(a, chunks, per, acc, m, l, s);
    default: return -1;
    }
}
#endif  /* !SPITE_KVFLASH_ARCH */

/* Fold the per-(chunk, head) partials into the attention output.
 *
 * out[h] = sum_c exp(m_c - M) * acc_c[h] / sum_c exp(m_c - M) * l_c,  M = max_c m_c
 * One block per query head; the chunk axis is walked twice (max, then the
 * weighted sum), which is cheap next to the tile kernels. */
__global__ void kvflash_combine(float* __restrict__ att, const float* __restrict__ wm,
                                const float* __restrict__ wl, const float* __restrict__ wacc,
                                int chunks, int nh, int hd) {
    const int h = blockIdx.x;
    float M = -INFINITY;
    for (int c = 0; c < chunks; ++c) M = fmaxf(M, wm[static_cast<size_t>(c) * nh + h]);

    float den = 0.0f;
    for (int c = 0; c < chunks; ++c)
        den += __expf(wm[static_cast<size_t>(c) * nh + h] - M) * wl[static_cast<size_t>(c) * nh + h];
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;

    const float* acch = wacc + static_cast<size_t>(h) * hd;
    float* outh = att + static_cast<size_t>(h) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float num = 0.0f;
        for (int c = 0; c < chunks; ++c)
            num += __expf(wm[static_cast<size_t>(c) * nh + h] - M) *
                   acch[static_cast<size_t>(c) * nh * hd + i];
        outh[i] = num * inv;
    }
}

/*
 * Flash-decoding attention for one token at `ctx->pos`.
 *
 * Shapes the tile kernel does not cover (head_dim outside the arch dispatch,
 * or a context too short to hold even one chunk of the split-K workspace) fall
 * back to kvattn_run(), which is why the pre-check runs before the prologue
 * rather than after it: falling back later would re-run the projections only to
 * throw the result away.
 */
inline int kvflash_run(SpiteTensor* out, const SpiteTensor* x, const SpiteTensor* wq,
                       const SpiteTensor* wk, const SpiteTensor* wv, const SpiteTensor* wo,
                       const SpiteTensor* q_norm, const SpiteTensor* k_norm, float norm_eps,
                       SpiteKvCache* kv, float rope_freq_base, const SpiteCtx* ctx) {
    if (!ctx || !kv || ctx->n_heads <= 0 || ctx->n_kv_heads <= 0) return -1;
    const int hd = static_cast<int>(wq->ne[1]) / ctx->n_heads;
    const int n_ctx = static_cast<int>(kv->k.ne[1]);
    /* The workspace below is carved out of the scores region, which is nh*n_ctx
     * floats, and costs chunks*nh*(hd+2) of it. */
    if (!kvflash_hd_supported(hd) || n_ctx < hd + 2)
        return kvattn_run(out, x, wq, wk, wv, wo, q_norm, k_norm, norm_eps, kv, rope_freq_base,
                          ctx);

    KvattnArgs a;
    const int rc = kvattn_prologue(&a, x, wq, wk, wv, q_norm, k_norm, norm_eps, kv,
                                   rope_freq_base, ctx);
    if (rc) return rc;

    const int nh = a.nh;
    /* The tile kernel writes acc with 128-bit stores, so it starts on a 16-byte
     * boundary; the 0..3 floats of slack come out of the budget below. */
    const uintptr_t raw = reinterpret_cast<uintptr_t>(a.scores);
    const uintptr_t aligned = (raw + 15u) & ~static_cast<uintptr_t>(15u);
    const int pad = static_cast<int>((aligned - raw) / sizeof(float));
    const int avail = static_cast<int>(ctx->scratchpad_bytes / sizeof(float)) -
                      (2 * nh * a.hd + 2 * a.kv_stride) - pad;
    const int cmax = avail / (nh * (a.hd + 2));
    int chunks = (a.n_tok + KVFLASH_TILES_MIN * KVFLASH_TILE - 1) / (KVFLASH_TILES_MIN * KVFLASH_TILE);
    if (chunks > cmax) chunks = cmax;
    if (chunks < 1) chunks = 1;
    const int per = (a.n_tok + chunks - 1) / chunks;

    float* wacc = reinterpret_cast<float*>(aligned);
    float* wm = wacc + static_cast<size_t>(chunks) * nh * a.hd;
    float* wl = wm + static_cast<size_t>(chunks) * nh;

    cudaStream_t s = stream_of(ctx);
    if (kvflash_launch_hd(a, chunks, per, wacc, wm, wl, s)) return -2;

    kvflash_combine<<<nh, threads_for(a.hd), 0, s>>>(a.att, wm, wl, wacc, chunks, nh, a.hd);
    if (launch_matvec(wo, a.att, static_cast<float*>(out->data), true, s)) return -1;
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

#endif  /* SPITE_QWEN3_KV_ATTN_FLASH_INL */
