/*
 * kernels/qwen/qwen3_8/nvidia/kv_attn.inl
 *
 * Shared Variable-Bit-Rate (VBR) attention for the Qwen3.8 NVIDIA kernels.
 *
 * The KV cache may be stored at any of the GGML tiers this project supports:
 * F32, F16, Q8_0, Q5_1, Q4_0. The calling kernel passes the tier in
 * `SpiteKvCache::k.kind` / `::v.kind`; the host re-encodes the whole cache when
 * it degrades, so every row in the buffer is always at one tier.
 *
 * The block layouts here MUST match `crates/spite-kvcache/src/quant.rs`
 * exactly — the host requantizes with those codecs and the device reads/writes
 * with these.
 *
 * Included at global scope from a kernel.cu after `BlockQ8_0`, `WARP`,
 * `ROWS_PER_BLOCK`, `warp_sum`, `block_sum`, `load_w` and `launch_matvec` are
 * defined. All names are `kvq_`/`kvattn_`-prefixed to avoid collisions.
 */

#ifndef SPITE_QWEN3_8_KV_ATTN_INL
#define SPITE_QWEN3_8_KV_ATTN_INL

constexpr int KVQ_BLOCK = 32;

struct KvqQ5_1 {
    __half d;
    __half m;
    uint32_t qh;
    uint8_t qs[16];
};
static_assert(sizeof(KvqQ5_1) == 24, "Q5_1 block must be 24 bytes");

struct KvqQ4_0 {
    __half d;
    uint8_t qs[16];
};
static_assert(sizeof(KvqQ4_0) == 18, "Q4_0 block must be 18 bytes");

__host__ __device__ __forceinline__ int kvq_block_bytes(int kind) {
    switch (kind) {
    case SPITE_TYPE_Q8_0: return 34;
    case SPITE_TYPE_Q5_1: return 24;
    case SPITE_TYPE_Q4_0: return 18;
    default: return 0;
    }
}

__host__ __device__ __forceinline__ bool kvq_supported(int kind) {
    return kind == SPITE_TYPE_F32 || kind == SPITE_TYPE_F16 || kind == SPITE_TYPE_Q8_0 ||
           kind == SPITE_TYPE_Q5_1 || kind == SPITE_TYPE_Q4_0;
}

/* Bytes occupied by one KV row of `n_elem` elements at `kind`. */
__host__ __device__ __forceinline__ size_t kvq_row_bytes(int kind, int n_elem) {
    if (kind == SPITE_TYPE_F32) return (size_t)n_elem * 4;
    if (kind == SPITE_TYPE_F16) return (size_t)n_elem * 2;
    const int bb = kvq_block_bytes(kind);
    return (size_t)((n_elem + KVQ_BLOCK - 1) / KVQ_BLOCK) * (size_t)bb;
}

__device__ __forceinline__ float kvq_warp_min(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float kvq_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Read element `i` of a packed KV row as float. */
__device__ __forceinline__ float kvq_get(const uint8_t* row, int kind, int i) {
    if (kind == SPITE_TYPE_F32) return reinterpret_cast<const float*>(row)[i];
    if (kind == SPITE_TYPE_F16) return __half2float(reinterpret_cast<const __half*>(row)[i]);
    const int lane = i & (KVQ_BLOCK - 1);
    const int blk = i >> 5;
    const int j = lane & 15;
    if (kind == SPITE_TYPE_Q8_0) {
        const BlockQ8_0* b = reinterpret_cast<const BlockQ8_0*>(row + (size_t)blk * 34);
        return static_cast<float>(b->qs[lane]) * __half2float(b->d);
    }
    if (kind == SPITE_TYPE_Q5_1) {
        const KvqQ5_1* b = reinterpret_cast<const KvqQ5_1*>(row + (size_t)blk * 24);
        const uint8_t byte = b->qs[j];
        const uint32_t lo = (lane < 16) ? (byte & 0x0F) : (byte >> 4);
        const uint32_t q = lo | (((b->qh >> lane) & 1u) << 4);
        return static_cast<float>(q) * __half2float(b->d) + __half2float(b->m);
    }
    const KvqQ4_0* b = reinterpret_cast<const KvqQ4_0*>(row + (size_t)blk * 18);
    const uint8_t byte = b->qs[j];
    const int lo = (lane < 16) ? (byte & 0x0F) : (byte >> 4);
    return static_cast<float>(lo - 8) * __half2float(b->d);
}

/* Quantize `x[0..n_elem]` into the packed `row`. One 32-thread block per block. */
__global__ void kvq_quantize_row(const float* __restrict__ x, uint8_t* __restrict__ row,
                                 int kind, int n_elem) {
    const int lane = threadIdx.x;
    const int base = blockIdx.x * KVQ_BLOCK;
    const int i0 = base + lane;
    const float v = (i0 < n_elem) ? x[i0] : 0.0f;

    if (kind == SPITE_TYPE_F16) {
        if (i0 < n_elem) reinterpret_cast<__half*>(row)[i0] = __float2half(v);
        return;
    }
    uint8_t* blk = row + (size_t)blockIdx.x * (size_t)kvq_block_bytes(kind);

    if (kind == SPITE_TYPE_Q8_0) {
        const float amax = kvq_warp_max(fabsf(v));
        const float d = amax / 127.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        int q = __float2int_rn(v * inv);
        q = max(-128, min(127, q));
        BlockQ8_0* b = reinterpret_cast<BlockQ8_0*>(blk);
        if (lane == 0) b->d = __float2half(d);
        b->qs[lane] = static_cast<int8_t>(q);
        return;
    }
    if (kind == SPITE_TYPE_Q5_1) {
        const float mx = kvq_warp_max(i0 < n_elem ? v : -INFINITY);
        const float mn = kvq_warp_min(i0 < n_elem ? v : INFINITY);
        const float d = (mx - mn) / 31.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        KvqQ5_1* b = reinterpret_cast<KvqQ5_1*>(blk);
        if (lane == 0) {
            b->d = __float2half(d);
            b->m = __float2half(mn);
            b->qh = 0u;
        }
        __syncwarp();
        if (lane < 16) {
            auto qz = [&](int j) -> unsigned {
                if (j >= n_elem) return 0u;
                int q = __float2int_rn((x[j] - mn) * inv);
                return static_cast<unsigned>(max(0, min(31, q)));
            };
            const unsigned q0 = qz(base + lane);
            const unsigned q1 = qz(base + lane + 16);
            b->qs[lane] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            const unsigned bits = (((q0 >> 4) & 1u) << lane) | (((q1 >> 4) & 1u) << (lane + 16));
            if (bits) atomicOr(reinterpret_cast<unsigned*>(&b->qh), bits);
        }
        return;
    }

    // Q4_0
    const float amax = kvq_warp_max(fabsf(v));
    const float d = amax / 8.0f;
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    KvqQ4_0* b = reinterpret_cast<KvqQ4_0*>(blk);
    if (lane == 0) b->d = __float2half(d);
    if (lane < 16) {
        auto qz = [&](int j) -> unsigned {
            if (j >= n_elem) return 8u;
            int q = __float2int_rn(x[j] * inv);
            q = max(-8, min(7, q));
            return static_cast<unsigned>(q + 8);
        };
        const unsigned q0 = qz(base + lane);
        const unsigned q1 = qz(base + lane + 16);
        b->qs[lane] = static_cast<uint8_t>(q0 | (q1 << 4));
    }
}

/* Per-head RMSNorm + NEOX RoPE; `dst` may alias `src`. */
__global__ void kvattn_qk_norm_rope(float* dst, const float* src, const void* __restrict__ nw,
                                    int nkind, float eps, int hd, int pos, float theta) {
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

/* scores[h, t] = q_h . k_t[kvh] * scale. One warp per (h, t). */
__global__ void kvattn_scores(float* __restrict__ scores, const float* __restrict__ q,
                              const uint8_t* __restrict__ kc, int n_tok, int n_ctx, int hd,
                              int group, int krow_bytes, int kind, float scale) {
    const int h = blockIdx.y;
    const int t = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (t >= n_tok) return;
    const int kh = h / group;
    const float* qh = q + static_cast<size_t>(h) * hd;
    const uint8_t* kr = kc + static_cast<size_t>(t) * krow_bytes;
    float acc = 0.0f;
    if (kind == SPITE_TYPE_F32 && (hd % 4) == 0) {
        const float4* q4 = reinterpret_cast<const float4*>(qh);
        const float4* k4 = reinterpret_cast<const float4*>(kr + static_cast<size_t>(kh) * hd * 4);
        const int hd4 = hd / 4;
        #pragma unroll 4
        for (int i = threadIdx.x; i < hd4; i += WARP) {
            const float4 a = q4[i];
            const float4 b4 = k4[i];
            acc = fmaf(a.x, b4.x, acc);
            acc = fmaf(a.y, b4.y, acc);
            acc = fmaf(a.z, b4.z, acc);
            acc = fmaf(a.w, b4.w, acc);
        }
    } else {
        #pragma unroll 4
        for (int i = threadIdx.x; i < hd; i += WARP)
            acc = fmaf(qh[i], kvq_get(kr, kind, kh * hd + i), acc);
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) scores[static_cast<size_t>(h) * n_ctx + t] = acc * scale;
}

/* att[h, i] = sum_t p[h, t] * v_t[kvh, i], F32 tier. Block per head, thread per
 * dim, one row of the cache per `t`.
 *
 * This stays a separate kernel from the block-tier version below on purpose: a
 * single kernel covering both tiers makes ptxas fold the block-decode address
 * math into this loop and costs ~1.8x here (this op is ~11% of decode time). */
__global__ void kvattn_weighted_v_f32(float* __restrict__ att, const float* __restrict__ scores,
                                      const float* __restrict__ vc, int n_tok, int n_ctx,
                                      int hd, int group, int kv_stride) {
    const int h = blockIdx.x;
    const float* p = scores + static_cast<size_t>(h) * n_ctx;
    const float* vb = vc + static_cast<size_t>(h / group) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_tok; ++t)
            acc += p[t] * vb[static_cast<size_t>(t) * kv_stride + i];
        att[static_cast<size_t>(h) * hd + i] = acc;
    }
}

/* att[h, i] = sum_t p[h, t] * v_t[kvh, i] for the block-quantized tiers
 * (q8_0/q5_1/q4_0), decoding each V element on read. */
__global__ void kvattn_weighted_v(float* __restrict__ att, const float* __restrict__ scores,
                                  const uint8_t* __restrict__ vc, int n_tok, int n_ctx, int hd,
                                  int group, int vrow_bytes, int kind) {
    const int h = blockIdx.x;
    const int vh = h / group;
    const float* p = scores + static_cast<size_t>(h) * n_ctx;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int t = 0; t < n_tok; ++t)
            acc = fmaf(p[t], kvq_get(vc + static_cast<size_t>(t) * vrow_bytes, kind, vh * hd + i),
                       acc);
        att[static_cast<size_t>(h) * hd + i] = acc;
    }
}

/* Everything both attention back ends need: shape validation, the scratchpad
 * budget, and this token's Q/K/V projections, per-head QK-norm/RoPE and KV-row
 * write.  kvattn_prologue() fills it; kvattn_run() (below) and the flash
 * kernels in kv_attn_flash.inl both start from it, so neither can drift on
 * validation or on what the host was asked to reserve. */
struct KvattnArgs {
    int nh, nkv, hd, kv_stride, n_ctx, n_tok, pos, group, kind;
    int row_bytes;   /* bytes per K (== V) row of the cache */
    float scale;     /* 1 / sqrt(hd) */
    const uint8_t* kc;  /* KV cache row 0, key side   */
    const uint8_t* vc;  /* KV cache row 0, value side */
    float* q;        /* [nh*hd]    */
    float* k;        /* [nkv*hd]   */
    float* att;      /* [nh*hd]    */
    float* scores;   /* [nh*n_ctx] VBR score vector; the flash back end reuses
                      *            this region as its split-K workspace */
    float* vtmp;     /* [nkv*hd]   */
};

/*
 * Scratchpad (floats): q[nh*hd] k[nkv*hd] att[nh*hd] scores[nh*n_ctx] vtmp[nkv*hd]
 * Returns 0, -1 (bad shape/kind) or -2 (bad scratchpad / launch failure).
 */
inline int kvattn_prologue(KvattnArgs* a, const SpiteTensor* x, const SpiteTensor* wq,
                           const SpiteTensor* wk, const SpiteTensor* wv,
                           const SpiteTensor* q_norm, const SpiteTensor* k_norm, float norm_eps,
                           SpiteKvCache* kv, float rope_freq_base, const SpiteCtx* ctx) {
    if (!ctx || !kv || ctx->n_heads <= 0 || ctx->n_kv_heads <= 0) return -1;
    const int nh = ctx->n_heads, nkv = ctx->n_kv_heads;
    const int hd = static_cast<int>(wq->ne[1]) / nh;
    const int kv_stride = nkv * hd;
    const int pos = ctx->pos;
    if (hd <= 0 || hd % 2 || nh % nkv || static_cast<int>(kv->k.ne[0]) != kv_stride ||
        static_cast<int>(kv->v.ne[0]) != kv_stride)
        return -1;
    const int kind = kv->k.kind;
    if (kind != kv->v.kind || !kvq_supported(kind)) return -1;
    const int n_ctx = static_cast<int>(kv->k.ne[1]);
    if (pos < 0 || pos >= n_ctx) return -2;

    const size_t need = sizeof(float) * (static_cast<size_t>(nh) * hd * 2 +
                                         static_cast<size_t>(kv_stride) * 2 +
                                         static_cast<size_t>(nh) * n_ctx);
    if (!ctx->scratchpad || ctx->scratchpad_bytes < need) return -2;
    float* q = static_cast<float*>(ctx->scratchpad);
    float* k = q + static_cast<size_t>(nh) * hd;
    float* att = k + kv_stride;
    float* scores = att + static_cast<size_t>(nh) * hd;
    float* vtmp = scores + static_cast<size_t>(nh) * n_ctx;

    cudaStream_t s = stream_of(ctx);
    const float* xin = static_cast<const float*>(x->data);
    uint8_t* k_row = static_cast<uint8_t*>(kv->k.data) + static_cast<size_t>(pos) * kv->k.nb[1];
    uint8_t* v_row = static_cast<uint8_t*>(kv->v.data) + static_cast<size_t>(pos) * kv->v.nb[1];
    const int rt = threads_for(hd / 2);
    const void* qn = q_norm ? q_norm->data : nullptr;
    const void* kn = k_norm ? k_norm->data : nullptr;

    if (kind == SPITE_TYPE_F32) {
        float* k_row_f = reinterpret_cast<float*>(k_row);
        float* v_row_f = reinterpret_cast<float*>(v_row);
        if (launch_matvec(wq, xin, q, false, s) || launch_matvec(wk, xin, k, false, s) ||
            launch_matvec(wv, xin, v_row_f, false, s))
            return -1;
        kvattn_qk_norm_rope<<<nh, rt, 0, s>>>(q, q, qn, q_norm ? q_norm->kind : 0, norm_eps, hd,
                                              pos, rope_freq_base);
        kvattn_qk_norm_rope<<<nkv, rt, 0, s>>>(k_row_f, k, kn, k_norm ? k_norm->kind : 0, norm_eps,
                                               hd, pos, rope_freq_base);
    } else {
        if (launch_matvec(wq, xin, q, false, s) || launch_matvec(wk, xin, k, false, s) ||
            launch_matvec(wv, xin, vtmp, false, s))
            return -1;
        kvattn_qk_norm_rope<<<nh, rt, 0, s>>>(q, q, qn, q_norm ? q_norm->kind : 0, norm_eps, hd,
                                              pos, rope_freq_base);
        kvattn_qk_norm_rope<<<nkv, rt, 0, s>>>(k, k, kn, k_norm ? k_norm->kind : 0, norm_eps, hd,
                                               pos, rope_freq_base);
        kvq_quantize_row<<<(kv_stride + KVQ_BLOCK - 1) / KVQ_BLOCK, KVQ_BLOCK, 0, s>>>(k, k_row,
                                                                                       kind,
                                                                                       kv_stride);
        kvq_quantize_row<<<(kv_stride + KVQ_BLOCK - 1) / KVQ_BLOCK, KVQ_BLOCK, 0, s>>>(
            vtmp, v_row, kind, kv_stride);
    }

    a->nh = nh;
    a->nkv = nkv;
    a->hd = hd;
    a->kv_stride = kv_stride;
    a->n_ctx = n_ctx;
    a->n_tok = pos + 1;
    a->pos = pos;
    a->group = nh / nkv;
    a->kind = kind;
    a->row_bytes = static_cast<int>(kvq_row_bytes(kind, kv_stride));
    a->scale = rsqrtf(static_cast<float>(hd));
    a->kc = static_cast<const uint8_t*>(kv->k.data);
    a->vc = static_cast<const uint8_t*>(kv->v.data);
    a->q = q;
    a->k = k;
    a->att = att;
    a->scores = scores;
    a->vtmp = vtmp;
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

/*
 * Full attention op for one token at `ctx->pos`, KV stored at `kv->k.kind` /
 * `kv->v.kind` (must match). F32 keeps the vectorized path; every other tier
 * quantizes the new row and dequantizes the history in the read kernels.
 *
 * This is the portable VBR back end: it materializes the score vector and then
 * walks it three times (write, softmax, weighted-V).  kv_attn_flash.inl fuses
 * those passes; this one remains the fallback for shapes the flash kernel does
 * not cover, and the differential reference the flash kernel is tested against.
 */
inline int kvattn_run(SpiteTensor* out, const SpiteTensor* x, const SpiteTensor* wq,
                      const SpiteTensor* wk, const SpiteTensor* wv, const SpiteTensor* wo,
                      const SpiteTensor* q_norm, const SpiteTensor* k_norm, float norm_eps,
                      SpiteKvCache* kv, float rope_freq_base, const SpiteCtx* ctx) {
    KvattnArgs a;
    const int rc = kvattn_prologue(&a, x, wq, wk, wv, q_norm, k_norm, norm_eps, kv,
                                  rope_freq_base, ctx);
    if (rc) return rc;

    cudaStream_t s = stream_of(ctx);
    const int nh = a.nh, hd = a.hd, n_tok = a.n_tok, n_ctx = a.n_ctx, group = a.group;
    kvattn_scores<<<dim3((n_tok + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, nh),
                    dim3(WARP, ROWS_PER_BLOCK), 0, s>>>(
        a.scores, a.q, static_cast<const uint8_t*>(kv->k.data), n_tok, n_ctx, hd, group,
        a.row_bytes, a.kind, a.scale);
    attn_softmax<<<nh, threads_for(n_tok > 256 ? 256 : n_tok), 0, s>>>(a.scores, n_tok, n_ctx);
    if (a.kind == SPITE_TYPE_F32) {
        kvattn_weighted_v_f32<<<nh, threads_for(hd), 0, s>>>(
            a.att, a.scores, static_cast<const float*>(kv->v.data), n_tok, n_ctx, hd, group,
            a.kv_stride);
    } else {
        kvattn_weighted_v<<<nh, threads_for(hd), 0, s>>>(
            a.att, a.scores, static_cast<const uint8_t*>(kv->v.data), n_tok, n_ctx, hd, group,
            a.row_bytes, a.kind);
    }

    if (launch_matvec(wo, a.att, static_cast<float*>(out->data), true, s)) return -1;
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

#endif  // SPITE_QWEN3_8_KV_ATTN_INL
