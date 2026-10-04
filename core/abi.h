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

#define SPITE_ABI_VERSION 4

/* ── Quant type tag ───────────────────────────────────────────────────── */

typedef enum {
    SPITE_TYPE_F32   = 0,
    SPITE_TYPE_F16   = 1,
    SPITE_TYPE_BF16  = 2,
    SPITE_TYPE_Q8_0  = 8,
    SPITE_TYPE_Q5_1  = 11,
    SPITE_TYPE_Q4_0  = 10,
    SPITE_TYPE_Q4_K  = 12,
    SPITE_TYPE_Q5_K  = 13,
    SPITE_TYPE_Q6_K  = 14,
} SpiteType;

/*
 * Byte size of the atomic storage unit for each type.
 *
 * For full-precision types this is bytes-per-element.
 * For block-quantised types this is bytes-per-block; the number of
 * elements per block is given by spite_type_block_elements().
 * Use byte strides (nb), never element strides — block-quant elements
 * do not have an integer byte size.
 */
static inline uint64_t spite_type_block_bytes(SpiteType t) {
    switch (t) {
        case SPITE_TYPE_F32:  return 4;
        case SPITE_TYPE_F16:  return 2;
        case SPITE_TYPE_BF16: return 2;
        case SPITE_TYPE_Q8_0: return 34;   /* 32-elem block: f16 scale + 32×i8 */
        case SPITE_TYPE_Q5_1: return 24;   /* 32-elem block: f16 d + f16 m + u32 qh + 16×u8 */
        case SPITE_TYPE_Q4_0: return 18;   /* 32-elem block: f16 scale + 16×u8 */
        case SPITE_TYPE_Q4_K: return 144;  /* 256-elem super-block */
        case SPITE_TYPE_Q5_K: return 176;  /* 256-elem super-block */
        case SPITE_TYPE_Q6_K: return 210;  /* 256-elem super-block */
        default:              return 0;
    }
}

/* Elements per atomic storage block (1 for full-precision types). */
static inline uint32_t spite_type_block_elements(SpiteType t) {
    switch (t) {
        case SPITE_TYPE_Q8_0:
        case SPITE_TYPE_Q5_1:
        case SPITE_TYPE_Q4_0: return 32;
        case SPITE_TYPE_Q4_K:
        case SPITE_TYPE_Q5_K:
        case SPITE_TYPE_Q6_K: return 256;
        default:              return 1;
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

typedef int (*SpiteAttentionFn)(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* wq,
    const SpiteTensor* wk,
    const SpiteTensor* wv,
    const SpiteTensor* wo,
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

typedef int (*SpiteFfnFn)(
    SpiteTensor*        out,
    const SpiteTensor*  x,
    const SpiteTensor*  w_gate,
    const SpiteTensor*  w_up,
    const SpiteTensor*  w_down,
    SpiteFfnActivation  activation,
    const SpiteCtx*     ctx
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

/* ── Kernel descriptor ────────────────────────────────────────────────── */

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
} SpiteKernelInfo;

/* Every kernel .so must export this symbol. */
typedef const SpiteKernelInfo* (*SpiteKernelInfoFn)(void);
