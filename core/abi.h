/*
 * spite/core/abi.h  —  stable C ABI contract for kernels
 *
 * C++23 / C11 compatible. Kernel .cu files include this directly.
 * The Rust host mirrors every type in crates/spite-abi/src/lib.rs.
 * Both sides must be updated together; ABI_VERSION is the enforcement.
 *
 * Rule: never change existing structs. Add new ones and bump ABI_VERSION.
 */

#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
#  define SPITE_NORETURN  [[noreturn]]
#  define SPITE_NODISCARD [[nodiscard]]
#else
#  define SPITE_NORETURN  _Noreturn
#  define SPITE_NODISCARD
#endif

#define SPITE_ABI_VERSION 7

/* ── Quant type tag ───────────────────────────────────────────────────── */

/*
 * Values are the GGUF / ggml tensor type ids, so the loader hands the file's
 * type id straight through and block layouts match ggml-common.h exactly
 * (vendored as core/ggml-common.h). ABI v6: ids changed from the earlier ad-hoc
 * numbering (Q4_0=10, Q4_K=12 meant a different type in a GGUF file).
 */
typedef enum {
    SPITE_TYPE_F32      = 0,
    SPITE_TYPE_F16      = 1,
    SPITE_TYPE_Q4_0     = 2,
    SPITE_TYPE_Q4_1     = 3,
    SPITE_TYPE_Q5_0     = 6,
    SPITE_TYPE_Q5_1     = 7,
    SPITE_TYPE_Q8_0     = 8,
    SPITE_TYPE_Q2_K     = 10,
    SPITE_TYPE_Q3_K     = 11,
    SPITE_TYPE_Q4_K     = 12,
    SPITE_TYPE_Q5_K     = 13,
    SPITE_TYPE_Q6_K     = 14,
    SPITE_TYPE_IQ2_XXS  = 16,
    SPITE_TYPE_IQ2_XS   = 17,
    SPITE_TYPE_IQ3_XXS  = 18,
    SPITE_TYPE_IQ1_S    = 19,
    SPITE_TYPE_IQ4_NL   = 20,
    SPITE_TYPE_IQ3_S    = 21,
    SPITE_TYPE_IQ2_S    = 22,
    SPITE_TYPE_IQ4_XS   = 23,
    SPITE_TYPE_IQ1_M    = 29,
    SPITE_TYPE_BF16     = 30,
    SPITE_TYPE_TQ1_0    = 34,
    SPITE_TYPE_TQ2_0    = 35,
    SPITE_TYPE_MXFP4    = 39,
    SPITE_TYPE_NVFP4    = 40,
    SPITE_TYPE_Q1_0     = 41,
    SPITE_TYPE_Q2_0     = 42,
} SpiteType;

/*
 * Byte size of the atomic storage unit for each type (0 = unknown type).
 *
 * For full-precision types this is bytes-per-element.
 * For block-quantised types this is bytes-per-block; the number of
 * elements per block is given by spite_type_block_elements().
 * Use byte strides (nb), never element strides — block-quant elements
 * do not have an integer byte size.
 */
static inline uint64_t spite_type_block_bytes(SpiteType t) {
    switch (t) {
        case SPITE_TYPE_F32: return 4;
        case SPITE_TYPE_F16: return 2;
        case SPITE_TYPE_Q4_0: return 18;
        case SPITE_TYPE_Q4_1: return 20;
        case SPITE_TYPE_Q5_0: return 22;
        case SPITE_TYPE_Q5_1: return 24;
        case SPITE_TYPE_Q8_0: return 34;
        case SPITE_TYPE_Q2_K: return 84;
        case SPITE_TYPE_Q3_K: return 110;
        case SPITE_TYPE_Q4_K: return 144;
        case SPITE_TYPE_Q5_K: return 176;
        case SPITE_TYPE_Q6_K: return 210;
        case SPITE_TYPE_IQ2_XXS: return 66;
        case SPITE_TYPE_IQ2_XS: return 74;
        case SPITE_TYPE_IQ3_XXS: return 98;
        case SPITE_TYPE_IQ1_S: return 50;
        case SPITE_TYPE_IQ4_NL: return 18;
        case SPITE_TYPE_IQ3_S: return 110;
        case SPITE_TYPE_IQ2_S: return 82;
        case SPITE_TYPE_IQ4_XS: return 136;
        case SPITE_TYPE_IQ1_M: return 56;
        case SPITE_TYPE_BF16: return 2;
        case SPITE_TYPE_TQ1_0: return 54;
        case SPITE_TYPE_TQ2_0: return 66;
        case SPITE_TYPE_MXFP4: return 17;
        case SPITE_TYPE_NVFP4: return 36;
        case SPITE_TYPE_Q1_0: return 18;
        case SPITE_TYPE_Q2_0: return 18;
        default: return 0;
    }
}

/* Elements per atomic storage block (1 for full-precision types, 0 = unknown). */
static inline uint32_t spite_type_block_elements(SpiteType t) {
    switch (t) {
        case SPITE_TYPE_F32: return 1;
        case SPITE_TYPE_F16: return 1;
        case SPITE_TYPE_Q4_0: return 32;
        case SPITE_TYPE_Q4_1: return 32;
        case SPITE_TYPE_Q5_0: return 32;
        case SPITE_TYPE_Q5_1: return 32;
        case SPITE_TYPE_Q8_0: return 32;
        case SPITE_TYPE_Q2_K: return 256;
        case SPITE_TYPE_Q3_K: return 256;
        case SPITE_TYPE_Q4_K: return 256;
        case SPITE_TYPE_Q5_K: return 256;
        case SPITE_TYPE_Q6_K: return 256;
        case SPITE_TYPE_IQ2_XXS: return 256;
        case SPITE_TYPE_IQ2_XS: return 256;
        case SPITE_TYPE_IQ3_XXS: return 256;
        case SPITE_TYPE_IQ1_S: return 256;
        case SPITE_TYPE_IQ4_NL: return 32;
        case SPITE_TYPE_IQ3_S: return 256;
        case SPITE_TYPE_IQ2_S: return 256;
        case SPITE_TYPE_IQ4_XS: return 256;
        case SPITE_TYPE_IQ1_M: return 256;
        case SPITE_TYPE_BF16: return 1;
        case SPITE_TYPE_TQ1_0: return 256;
        case SPITE_TYPE_TQ2_0: return 256;
        case SPITE_TYPE_MXFP4: return 32;
        case SPITE_TYPE_NVFP4: return 64;
        case SPITE_TYPE_Q1_0: return 128;
        case SPITE_TYPE_Q2_0: return 64;
        default: return 0;
    }
}

/* ── Tensor ───────────────────────────────────────────────────────────── */

/*
 * Tensor view. `data` may point into a mmap'd GGUF buffer (weights) or a
 * GPU/CPU activation buffer. Kernels must not free it.
 *
 * nb[] are BYTE strides per dimension:
 *   nb[0] = spite_type_block_bytes(kind)          (bytes per block/element)
 *   nb[1] = nb[0] * (ne[0] / block_elements)      (bytes per row)
 *   nb[2] = nb[1] * ne[1]                         (bytes per matrix)
 *   nb[3] = nb[2] * ne[2]                         (bytes per batch)
 *
 * A contiguous tensor satisfies the above equalities for every dimension
 * where ne[i] > 1. The executor guarantees that all tensors passed to
 * external kernel .so files are contiguous; use spite_tensor_is_contiguous()
 * to assert this at kernel entry during development.
 */
typedef struct {
    void*     data;     /* pointer into GGUF buffer or activation buffer */
    uint32_t  ne[4];    /* ne[0]=cols, ne[1]=rows, ne[2]=matrices, ne[3]=batch */
    uint64_t  nb[4];    /* byte strides — see comment above */
    SpiteType kind;
} SpiteTensor;

/* True iff the tensor's strides are tightly packed (no padding, no transposition). */
static inline bool spite_tensor_is_contiguous(const SpiteTensor* t) {
    if (t->nb[0] == 0) return true;
    uint64_t blk  = (uint64_t)spite_type_block_elements(t->kind);
    uint64_t row  = t->nb[0] * ((uint64_t)t->ne[0] / blk);
    if (t->ne[1] > 1 && t->nb[1] != row)                            return false;
    if (t->ne[2] > 1 && t->nb[2] != t->nb[1] * (uint64_t)t->ne[1]) return false;
    if (t->ne[3] > 1 && t->nb[3] != t->nb[2] * (uint64_t)t->ne[2]) return false;
    return true;
}

/* Compute contiguous strides for a freshly-allocated tensor. */
static inline void spite_contiguous_strides(SpiteType kind, const uint32_t ne[4],
                                             uint64_t nb_out[4]) {
    uint64_t blk = (uint64_t)spite_type_block_elements(kind);
    nb_out[0] = spite_type_block_bytes(kind);
    nb_out[1] = nb_out[0] * ((uint64_t)ne[0] / blk > 0 ? (uint64_t)ne[0] / blk : 1);
    nb_out[2] = nb_out[1] * (uint64_t)ne[1];
    nb_out[3] = nb_out[2] * (uint64_t)ne[2];
}

/* ── Inference context ────────────────────────────────────────────────── */

typedef struct {
    int     n_ctx;
    int     n_batch;
    int     n_threads;
    int     pos;              /* current token position in the sequence (0-based) */
    int     n_heads;          /* total query heads */
    int     n_kv_heads;       /* KV heads — may be < n_heads for GQA/MQA */
    void*   gpu_stream;       /* CUDA stream / HIP stream / MTLCommandBuffer */
    void*   scratchpad;
    size_t  scratchpad_bytes;
} SpiteCtx;

/* ── KV cache ─────────────────────────────────────────────────────────── */

typedef struct {
    SpiteTensor k;
    SpiteTensor v;
    int         layer;
} SpiteKvCache;

/* ── Op signatures ────────────────────────────────────────────────────── */
/* Return 0 on success, -1 if not implemented (dispatcher uses fallback). */

typedef int (*SpiteRmsNormFn)(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* weight,
    float              eps,
    const SpiteCtx*    ctx
);

/*
 * Attention for one token at ctx->pos.
 * ABI v4: result is ACCUMULATED into out (out += attn(x)) — residual fused.
 * q_norm / k_norm: optional per-head RMSNorm weights [head_dim] (Qwen3);
 * NULL when absent. head_dim = wq->ne[1] / ctx->n_heads.
 */
typedef int (*SpiteAttentionFn)(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* wq,
    const SpiteTensor* wk,
    const SpiteTensor* wv,
    const SpiteTensor* wo,
    const SpiteTensor* q_norm,
    const SpiteTensor* k_norm,
    float              norm_eps,
    SpiteKvCache*      kvcache,
    float              rope_freq_base,
    const SpiteCtx*    ctx              /* pos, n_heads, n_kv_heads live in ctx */
);

/* FFN activation function selector. */
typedef enum {
    SPITE_FFN_SILU_GATE = 0,  /* SwiGLU — LLaMA, Mistral, Qwen */
    SPITE_FFN_GELU_GATE = 1,  /* GeGLU  — Gemma */
    SPITE_FFN_GELU      = 2,  /* standard GELU — BERT-family, Phi */
    SPITE_FFN_RELU      = 3,  /* ReLU²  — GPT-NeoX variants */
} SpiteFfnActivation;

/* ABI v4: result is ACCUMULATED into out (out += ffn(x)). */
typedef int (*SpiteFfnFn)(
    SpiteTensor*        out,
    const SpiteTensor*  x,
    const SpiteTensor*  w_gate,
    const SpiteTensor*  w_up,
    const SpiteTensor*  w_down,
    SpiteFfnActivation  activation,
    const SpiteCtx*     ctx
);

/* Dense projection out[r] = sum_c w[r,c] * x[c] (overwrites out). ABI v4. */
typedef int (*SpiteMatmulFn)(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* w,
    const SpiteCtx*    ctx
);

/*
 * Multi-head Latent Attention (DeepSeek MLA).
 *
 * KV is compressed through low-rank projections before caching. The
 * compressed latent is stored in the KV cache; up-projection happens
 * during the attention score computation.
 */
typedef int (*SpiteMlaFn)(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* w_dq,       /* query down-projection (absorbs W_Q) */
    const SpiteTensor* w_uq,       /* query up-projection */
    const SpiteTensor* w_dkv,      /* KV down-projection (shared compress) */
    const SpiteTensor* w_ukv,      /* KV up-projection */
    const SpiteTensor* wo,         /* output projection */
    SpiteKvCache*      kvcache,
    float              rope_freq_base,
    const SpiteCtx*    ctx
);

/* Optional: fuse rms_norm + attention + ffn for one full layer. */
typedef int (*SpiteLayerFn)(
    SpiteTensor*       out,
    const SpiteTensor* x,
    int                layer_idx,
    SpiteKvCache*      kvcache,
    int                pos,
    const SpiteCtx*    ctx
);

/* ── Speculative decoding ops ─────────────────────────────────────────── */

/*
 * Verify N draft tokens in parallel.
 *
 * Compares draft_logits[i] against main_logits[i] using the standard
 * speculative sampling accept/reject rule and writes the accept mask.
 * The first rejected position and all positions after it are set false.
 *
 *   accept_mask : bool output [n_draft]
 *   draft_logits: float input  [n_draft, vocab_size]  (from draft model)
 *   main_logits : float input  [n_draft, vocab_size]  (from main model)
 *   temperature : sampling temperature (applied to both before comparison)
 *
 * Returning -1 falls back to the generic scalar implementation.
 * A GPU-specific kernel can fuse the softmax + comparison + sampling
 * into a single pass — especially valuable at large vocab sizes.
 */
typedef int (*SpiteSpecVerifyFn)(
    bool*              accept_mask,
    const SpiteTensor* draft_logits,
    const SpiteTensor* main_logits,
    float              temperature,
    uint32_t           n_draft,
    const SpiteCtx*    ctx
);

/* ── Model capability declaration ─────────────────────────────────────── */

/*
 * Describes what a model supports. Loaded from GGUF metadata by the
 * runtime — kernel authors do not fill this; it comes from the model file.
 * Exposed here so kernels can query it if needed.
 */
typedef struct {
    /* Speculative decoding */
    bool     can_verify;          /* model can act as the verifier */
    bool     can_draft;           /* model can act as the draft     */
    uint32_t max_draft_tokens;    /* 0 = speculative not supported  */
    /* Compatible draft architectures, NULL-terminated list of strings. */
    const char* const* draft_archs;

    /* Future capability flags live here — add fields, bump ABI_VERSION. */
} SpiteModelCaps;

/* ── Extended attention (ABI v7) ──────────────────────────────────────── */

/*
 * Attention variant used by hybrid Qwen3.5-style full-attention layers.
 *   head_dim  per-head width (no longer derivable from wq when the Q
 *             projection is gated)
 *   rope_dim  only the first rope_dim dims of every q/k head are rotated
 *             (NEOX pairing i <-> i+rope_dim/2, freq_i = base^(-2i/rope_dim));
 *             rope_dim == head_dim reproduces plain attention
 *   gated_q   0: wq has n_heads*head_dim rows.
 *             1: wq has 2*n_heads*head_dim rows laid out per head as
 *                [q(head_dim) | gate(head_dim)]; after attention each head's
 *                output is multiplied by sigmoid(gate) before wo.
 * Everything else is exactly SpiteAttentionFn (norms, KV cache, ctx->pos,
 * out += wo . attn). Return 0 / -1 (unsupported geometry or type) / -2 (scratch).
 *
 * Scratch (ctx->scratchpad, floats): spite_attn_ex_scratch_floats().
 */
typedef struct {
    int32_t head_dim;
    int32_t rope_dim;
    int32_t gated_q;
} SpiteAttnParams;

static inline uint64_t spite_attn_ex_scratch_floats(const SpiteAttnParams* p, int n_heads,
                                                    int n_kv_heads, int n_ctx) {
    /* q+gate(2) + att + k + vtmp + per-head score workspace */
    return (uint64_t)3 * n_heads * p->head_dim + 2ull * n_kv_heads * p->head_dim +
           (uint64_t)n_heads * n_ctx;
}

typedef int (*SpiteAttentionExFn)(
    SpiteTensor*          out,
    const SpiteTensor*    x,
    const SpiteTensor*    wq,
    const SpiteTensor*    wk,
    const SpiteTensor*    wv,
    const SpiteTensor*    wo,
    const SpiteTensor*    q_norm,
    const SpiteTensor*    k_norm,
    float                 norm_eps,
    SpiteKvCache*         kvcache,
    float                 rope_freq_base,
    const SpiteAttnParams* params,
    const SpiteCtx*       ctx
);

/* ── Linear attention: Gated Delta Net layer (ABI v7) ─────────────────── */

/*
 *   n_kh      key/query heads;  n_vh value heads (n_vh % n_kh == 0)
 *   head_dim  S: key, query and value head width (state is S x S per value head)
 *   d_conv    K: depthwise conv taps (history holds K-1 previous inputs)
 *   norm_eps  epsilon of the q/k L2 normalisation and of the gated RMS norm
 */
typedef struct {
    int32_t n_kh;
    int32_t n_vh;
    int32_t head_dim;
    int32_t d_conv;
    float   norm_eps;
} SpiteGdnParams;

/* Scratch floats the layer op needs in ctx->scratchpad: qkv | z | core | beta | alpha. */
static inline uint64_t spite_gdn_scratch_floats(const SpiteGdnParams* p) {
    const uint64_t V = (uint64_t)p->n_vh * p->head_dim;
    return 2ull * p->n_kh * p->head_dim + V /*qkv*/ + V /*z*/ + V /*core*/ + 2ull * p->n_vh;
}

/*
 * One decode token of a complete GDN layer: out += W_out . gated_norm(core(x)).
 * `x` is the already RMS-normalised layer input [d_model] (F32). Weights w_*
 * may be any SpiteType (GGUF [cols=d_model, rows] layout); the small vectors
 * are F32. All state is caller-owned device memory, updated in place.
 *
 *   qkv[C]   = w_qkv . x                  C = 2*n_kh*S + n_vh*S   (q | k | v)
 *   z[V]     = w_gate . x                 V = n_vh*S
 *   beta[h]  = sigmoid(w_beta . x)        alpha[h] = w_alpha . x      h < n_vh
 *   g[h]     = softplus(alpha[h] + ssm_dt[h]) * ssm_a[h]
 *   conv_w   [K, C] ggml layout, ne[0]=K: tap k of channel c at conv_w[c*K + k];
 *            conv[c] = sum_{k<K-1} hist[k][c]*w[c][k] + qkv[c]*w[c][K-1]
 *            (hist oldest first), then SiLU.  hist <- shift, append qkv.
 *   conv_hist storage is private to the kernel, (K-1)*C floats, zero-initialised.
 *   per value head vh (kh = vh % n_kh):
 *     q,k = l2norm(conv q[kh]), l2norm(conv k[kh]);  l2norm(x) =
 *           x / sqrt(mean(x^2) + eps/S) / sqrt(S);  q *= 1/sqrt(S)
 *     M *= exp(g); d = (v - M^T k) * beta; M += k (x) d; o = M^T q
 *   y[vh*S + s] = rms_norm(o[vh], ssm_norm[S], eps)[s] * silu(z[vh*S + s])
 *   out += w_out . y
 * state: [n_vh, S, S] F32, M[r][s] row-major, zero-initialised by the host.
 * Return 0 / -1 (unsupported geometry or weight type) / -2 (scratch too small).
 */
typedef int (*SpiteGdnFn)(
    SpiteTensor*          out,
    const SpiteTensor*    x,
    const SpiteTensor*    w_qkv,
    const SpiteTensor*    w_gate,
    const SpiteTensor*    w_beta,
    const SpiteTensor*    w_alpha,
    const SpiteTensor*    w_out,
    const SpiteTensor*    conv_w,
    const SpiteTensor*    ssm_dt,
    const SpiteTensor*    ssm_a,
    const SpiteTensor*    ssm_norm,
    SpiteTensor*          conv_hist,
    SpiteTensor*          state,
    const SpiteGdnParams* params,
    const SpiteCtx*       ctx
);

/* ── Kernel descriptor ────────────────────────────────────────────────── */

/*
 * Optional: KV-cache tiers this kernel's attention op can read and write, as a
 * bitmask with bit (SpiteType) set. A kernel that leaves this NULL predates
 * VBR and is taken to accept F32 KV only — the conservative reading. The host
 * clamps the VBR start tier and the degrade ladder to this set, so a kernel
 * that supports fewer tiers still runs rather than failing the attention op.
 */
typedef uint64_t (*SpiteKvCacheKindsFn)(void);

typedef struct {
    uint32_t    abi_version;
    const char* model_arch;       /* e.g. "llama3" */
    const char* gpu_arch;         /* e.g. "sm_89"  */
    const char* author;

    uint32_t supported_quants[8]; /* 0-terminated list of SpiteType values */

    /* NULL = not implemented; dispatcher uses fallback. */
    SpiteRmsNormFn   rms_norm;
    SpiteAttentionFn attention;
    SpiteMlaFn       mla;
    SpiteFfnFn       ffn;
    SpiteLayerFn     layer;
    SpiteSpecVerifyFn speculative_verify; /* NULL if no optimized impl */
    /* Chunked prefill: process prompt in chunks; caller passes chunk_idx via pos. */
    SpiteLayerFn     prefill;
    /* Dense projection (LM head). Added in ABI v4. */
    SpiteMatmulFn    matmul;
    /* KV-cache tiers this attention op accepts; NULL = F32 only. */
    SpiteKvCacheKindsFn kv_cache_kinds;
    /* Gated-delta-net layer (hybrid archs). ABI v7 (layer-level; v5/v6 had a core-only op). */
    SpiteGdnFn       linear_attn;
    /* Extended attention: partial RoPE + gated Q. ABI v7. */
    SpiteAttentionExFn attention_ex;
} SpiteKernelInfo;

/* Every kernel .so must export this symbol. */
typedef const SpiteKernelInfo* (*SpiteKernelInfoFn)(void);
