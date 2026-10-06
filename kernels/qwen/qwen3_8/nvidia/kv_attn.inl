/*
 * kernels/qwen/qwen3_8/nvidia/kv_attn.inl
 *
 * Variable-Bit-Rate KV attention for Qwen3.8 NVIDIA kernels.
 * KV cache may be stored at F32, F16, Q8_0, Q5_1 or Q4_0.
 *
 * Block layouts MUST match crates/spite-kvcache/src/quant.rs.
 * Included from kernel.cu after BlockQ8_0, WARP, ROWS_PER_BLOCK,
 * warp_sum, block_sum, load_w and launch_matvec are defined.
 * All public names are kvq_/kvattn_-prefixed.
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
    return kind == SPITE_TYPE_F32 || kind == SPITE_TYPE_F16 ||
           kind == SPITE_TYPE_Q8_0 || kind == SPITE_TYPE_Q5_1 ||
           kind == SPITE_TYPE_Q4_0;
}

__host__ __device__ __forceinline__ size_t kvq_row_bytes(int kind, int n_elem) {
    if (kind == SPITE_TYPE_F32) return (size_t)n_elem * 4;
    if (kind == SPITE_TYPE_F16) return (size_t)n_elem * 2;
    const int bb = kvq_block_bytes(kind);
    return (size_t)((n_elem + KVQ_BLOCK - 1) / KVQ_BLOCK) * (size_t)bb;
}

/* Dequantize element i of a KV row stored at ptr (kind). */
__device__ __forceinline__ float kvq_get(const void* ptr, int kind, int i) {
    if (kind == SPITE_TYPE_F32) return static_cast<const float*>(ptr)[i];
    if (kind == SPITE_TYPE_F16) return __half2float(static_cast<const __half*>(ptr)[i]);
    if (kind == SPITE_TYPE_Q8_0) {
        const auto* b = reinterpret_cast<const BlockQ8_0*>(ptr) + i / KVQ_BLOCK;
        return __half2float(b->d) * static_cast<float>(b->qs[i % KVQ_BLOCK]);
    }
    if (kind == SPITE_TYPE_Q5_1) {
        const auto* b = reinterpret_cast<const KvqQ5_1*>(ptr) + i / KVQ_BLOCK;
        const int j = i % KVQ_BLOCK;
        const int lo = (b->qs[j / 2] >> (4 * (j & 1))) & 0xf;
        const int hi = (b->qh >> j) & 1;
        const float d = __half2float(b->d), m = __half2float(b->m);
        return d * static_cast<float>(lo | (hi << 4)) + m;
    }
    /* Q4_0 */
    const auto* b = reinterpret_cast<const KvqQ4_0*>(ptr) + i / KVQ_BLOCK;
    const int j = i % KVQ_BLOCK;
    const int q4 = (b->qs[j / 2] >> (4 * (j & 1))) & 0xf;
    return __half2float(b->d) * static_cast<float>(q4 - 8);
}

/* Arguments common to all attention paths. */
struct KvattnArgs {
    const float*  q;          /* [n_heads, head_dim]           */
    const void*   k_cache;    /* [n_ctx,   n_kv_heads, hd]  quantised */
    const void*   v_cache;    /* [n_ctx,   n_kv_heads, hd]  quantised */
    float*        scores;     /* scratch [n_heads, n_ctx]      */
    float*        out;        /* [n_heads, head_dim]  (+=)     */
    int           n_heads;
    int           n_kv_heads;
    int           head_dim;
    int           n_tok;      /* context length written so far */
    int           n_ctx;      /* max context (stride)          */
    int           kv_kind;    /* SpiteType of KV cache         */
    float         scale;      /* 1/sqrt(head_dim)              */
};

/* Build KvattnArgs from the ABI tensors. Returns false on unsupported config. */
__host__ __forceinline__ bool kvattn_prologue(
        KvattnArgs& a,
        const SpiteTensor* wq, const SpiteKvCache* kv, const SpiteCtx* ctx,
        float* q_buf, float* scores_buf, float* out_buf) {
    if (!kvq_supported(kv->k.kind)) return false;
    a.n_heads    = ctx->n_heads;
    a.n_kv_heads = ctx->n_kv_heads > 0 ? ctx->n_kv_heads : ctx->n_heads;
    a.head_dim   = wq ? static_cast<int>(wq->ne[0]) / a.n_heads : 128;
    a.n_tok      = ctx->pos + 1;
    a.n_ctx      = ctx->n_ctx > 0 ? ctx->n_ctx : a.n_tok;
    a.kv_kind    = static_cast<int>(kv->k.kind);
    a.scale      = 1.0f / sqrtf(static_cast<float>(a.head_dim));
    a.k_cache    = kv->k.data;
    a.v_cache    = kv->v.data;
    a.scores     = scores_buf;
    a.out        = out_buf;
    a.q          = q_buf;
    return true;
}

/* Score kernel: scores[h, t] = scale * dot(q[h], k_cache[t, kv_h]) */
__global__ void kvattn_scores(const KvattnArgs* __restrict__ ga) {
    const KvattnArgs& a = *ga;
    const int h = blockIdx.x, t = blockIdx.y;
    if (h >= a.n_heads || t >= a.n_tok) return;
    const int kv_h = h / (a.n_heads / a.n_kv_heads);
    const size_t kv_row = (static_cast<size_t>(t) * a.n_kv_heads + kv_h) * a.n_ctx;
    const void* krow = static_cast<const char*>(a.k_cache) +
                       kv_row * kvq_row_bytes(a.kv_kind, a.head_dim) / a.n_ctx;
    /* Simpler: row t, head kv_h, stored as [n_ctx, n_kv_heads, head_dim]. */
    const size_t off = ((size_t)t * a.n_kv_heads + kv_h);
    krow = static_cast<const char*>(a.k_cache) + off * kvq_row_bytes(a.kv_kind, a.head_dim);

    const float* qh = a.q + static_cast<size_t>(h) * a.head_dim;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < a.head_dim; i += blockDim.x)
        dot += qh[i] * kvq_get(krow, a.kv_kind, i);
    dot = block_sum(dot);
    if (threadIdx.x == 0)
        a.scores[static_cast<size_t>(h) * a.n_ctx + t] = a.scale * dot;
}

/* Weighted-V accumulation: out[h] += sum_t softmax[h,t] * v_cache[t, kv_h] */
__global__ void kvattn_weighted_v(const KvattnArgs* __restrict__ ga) {
    const KvattnArgs& a = *ga;
    const int h = blockIdx.x;
    if (h >= a.n_heads) return;
    const int kv_h = h / (a.n_heads / a.n_kv_heads);
    const float* sc = a.scores + static_cast<size_t>(h) * a.n_ctx;
    float* oh = a.out + static_cast<size_t>(h) * a.head_dim;
    for (int i = threadIdx.x; i < a.head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < a.n_tok; ++t) {
            const size_t off = ((size_t)t * a.n_kv_heads + kv_h);
            const void* vrow = static_cast<const char*>(a.v_cache) +
                               off * kvq_row_bytes(a.kv_kind, a.head_dim);
            acc += sc[t] * kvq_get(vrow, a.kv_kind, i);
        }
        oh[i] += acc;
    }
}

#endif /* SPITE_QWEN3_8_KV_ATTN_INL */
