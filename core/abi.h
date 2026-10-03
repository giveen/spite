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

#ifdef __cplusplus
#  define SPITE_NORETURN  [[noreturn]]
#  define SPITE_NODISCARD [[nodiscard]]
#else
#  define SPITE_NORETURN  _Noreturn
#  define SPITE_NODISCARD
#endif

#define SPITE_ABI_VERSION 1

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

/* ── Kernel descriptor ────────────────────────────────────────────────── */

typedef struct {
    uint32_t    abi_version;
    const char* model_arch;       /* e.g. "llama3" */
    const char* gpu_arch;         /* e.g. "sm_89"  */
    const char* author;           /* optional credit */

    uint32_t supported_quants[8]; /* 0-terminated list of SpiteType values */

    /* NULL = not implemented; dispatcher uses fallback. */
    SpiteRmsNormFn  rms_norm;
    SpiteAttentionFn attention;
    SpiteFfnFn      ffn;
    SpiteLayerFn    layer;
} SpiteKernelInfo;

/* Every kernel .so must export this symbol. */
typedef const SpiteKernelInfo* (*SpiteKernelInfoFn)(void);
