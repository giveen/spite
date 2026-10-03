/*
 * spite/core/abi.h  —  the contract every kernel must implement
 *
 * Every file under kernels/<model>/<gpu>/ implements these functions.
 * The signatures here are frozen. Adding a parameter here breaks every
 * downstream kernel. Think carefully before touching this file.
 *
 * Version history is tracked in ABI_VERSION below. Kernels declare
 * which version they were written against; the dispatcher rejects
 * mismatches and falls back to the generic kernel.
 */

#pragma once
#include <stdint.h>
#include <stddef.h>

#define SPITE_ABI_VERSION 1

/* ── Tensor ───────────────────────────────────────────────────────────── */

typedef enum {
    SPITE_TYPE_F32   = 0,
    SPITE_TYPE_F16   = 1,
    SPITE_TYPE_BF16  = 2,
    SPITE_TYPE_Q8_0  = 8,
    SPITE_TYPE_Q4_0  = 10,
    SPITE_TYPE_Q4_K  = 12,   /* covers Q4_K_S and Q4_K_M */
    SPITE_TYPE_Q5_K  = 13,
    SPITE_TYPE_Q6_K  = 14,
} spite_type_t;

typedef struct {
    void       *data;        /* pointer into mmap'd GGUF buffer */
    uint32_t    ne[4];       /* dimensions: ne[0]=cols, ne[1]=rows, ... */
    spite_type_t type;
} spite_tensor_t;

/* ── Context passed to every kernel op ───────────────────────────────── */

typedef struct {
    int     n_ctx;           /* current context length */
    int     n_batch;         /* tokens in this forward pass */
    int     n_threads;       /* CPU threads available (ignored by GPU kernels) */
    void   *gpu_stream;      /* CUDA stream / HIP stream / Metal command buffer */
    void   *scratchpad;      /* pre-allocated GPU scratch memory */
    size_t  scratchpad_bytes;
} spite_ctx_t;

/* ── KV cache layout ─────────────────────────────────────────────────── */

typedef struct {
    spite_tensor_t k;        /* [n_layers, n_ctx, n_kv_heads, head_dim] */
    spite_tensor_t v;
    int            layer;    /* which layer this cache entry is for */
} spite_kvcache_t;

/* ── Ops every kernel file must export ───────────────────────────────── */
/*
 * All functions return 0 on success, negative on error.
 * Kernels that don't support an op leave the symbol absent;
 * the dispatcher falls back to the generic implementation.
 */

/* required: single-token or batched RMS norm */
typedef int (*spite_rms_norm_fn)(
    spite_tensor_t       *out,
    const spite_tensor_t *x,
    const spite_tensor_t *weight,
    float                 eps,
    const spite_ctx_t    *ctx
);

/* required: QKV projection + rope + attention + output projection */
typedef int (*spite_attention_fn)(
    spite_tensor_t       *out,
    const spite_tensor_t *x,
    const spite_tensor_t *wq,
    const spite_tensor_t *wk,
    const spite_tensor_t *wv,
    const spite_tensor_t *wo,
    spite_kvcache_t      *kvcache,
    int                   pos,      /* current token position */
    float                 rope_freq_base,
    const spite_ctx_t    *ctx
);

/* required: gate + up projection, SiLU activation, down projection */
typedef int (*spite_ffn_fn)(
    spite_tensor_t       *out,
    const spite_tensor_t *x,
    const spite_tensor_t *w_gate,
    const spite_tensor_t *w_up,
    const spite_tensor_t *w_down,
    const spite_ctx_t    *ctx
);

/* optional: fused rms_norm + attention + ffn for a single layer */
typedef int (*spite_layer_fn)(
    spite_tensor_t       *out,
    const spite_tensor_t *x,
    int                   layer_idx,
    spite_kvcache_t      *kvcache,
    int                   pos,
    const spite_ctx_t    *ctx
);

/* ── Kernel descriptor — returned by spite_kernel_info() ─────────────── */

typedef struct {
    uint32_t    abi_version;    /* must equal SPITE_ABI_VERSION */
    const char *model_arch;     /* e.g. "llama3", "mistral" */
    const char *gpu_arch;       /* e.g. "sm_89", "rdna3", "metal" */
    const char *author;         /* optional, for credit */

    /* which quant types this kernel handles (0-terminated list) */
    spite_type_t supported_quants[8];

    /* function pointers — NULL means "not implemented, use fallback" */
    spite_rms_norm_fn  rms_norm;
    spite_attention_fn attention;
    spite_ffn_fn       ffn;
    spite_layer_fn     layer;   /* NULL unless the kernel fuses the full layer */
} spite_kernel_info_t;

/*
 * Every kernel shared library must export this function.
 * The dispatcher calls it to validate the kernel before use.
 */
typedef const spite_kernel_info_t *(*spite_kernel_info_fn)(void);
