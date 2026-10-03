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

#define SPITE_ABI_VERSION 2

/* ── Quant type tag ───────────────────────────────────────────────────── */

typedef enum {
    SPITE_TYPE_F32   = 0,
    SPITE_TYPE_F16   = 1,
    SPITE_TYPE_BF16  = 2,
    SPITE_TYPE_Q8_0  = 8,
    SPITE_TYPE_Q4_0  = 10,
    SPITE_TYPE_Q4_K  = 12,
    SPITE_TYPE_Q5_K  = 13,
    SPITE_TYPE_Q6_K  = 14,
} SpiteType;

/* ── Tensor ───────────────────────────────────────────────────────────── */

typedef struct {
    void*    data;      /* pointer into mmap'd GGUF buffer — do not free */
    uint32_t ne[4];     /* ne[0]=cols, ne[1]=rows, ... */
    SpiteType kind;
} SpiteTensor;

/* ── Inference context ────────────────────────────────────────────────── */

typedef struct {
    int     n_ctx;
    int     n_batch;
    int     n_threads;
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
    int                pos,
    float              rope_freq_base,
    const SpiteCtx*    ctx
);

typedef int (*SpiteFfnFn)(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* w_gate,
    const SpiteTensor* w_up,
    const SpiteTensor* w_down,
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
    SpiteFfnFn       ffn;
    SpiteLayerFn     layer;
    SpiteSpecVerifyFn speculative_verify; /* NULL if no optimized impl */
    /* Chunked prefill: process prompt in chunks; caller passes chunk_idx via pos. */
    SpiteLayerFn     prefill;
} SpiteKernelInfo;

/* Every kernel .so must export this symbol. */
typedef const SpiteKernelInfo* (*SpiteKernelInfoFn)(void);
