/*
 * kernels/generic/generic/kernel.c
 *
 * Generic fallback kernel — always compiled, always present.
 * Exports `spite_kernel_info()` so the dispatcher can dlopen this .so.
 *
 * This kernel handles:
 *   rms_norm  — scalar C, any F32/Q8_0/Q4_K weight
 *   ffn       — scalar C, any F32/Q8_0/Q4_K weight
 *   attention — returns -1 (Rust spite-compute scalar GQA takes over)
 *   layer     — NULL  (individual ops are dispatched separately)
 *   prefill   — NULL  (Rust handles chunked prefill)
 */

#include "../../../core/abi.h"

/* forward-declare ops from ops.c */
int spite_generic_rms_norm(
    SpiteTensor*, const SpiteTensor*, const SpiteTensor*, float, const SpiteCtx*);
int spite_generic_ffn(
    SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, SpiteFfnActivation, const SpiteCtx*);
int spite_generic_attention(
    SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    SpiteKvCache*, float, const SpiteCtx*);

static const SpiteKernelInfo GENERIC_KERNEL_INFO = {
    .abi_version = SPITE_ABI_VERSION,
    .model_arch  = "generic",
    .gpu_arch    = "generic",
    .author      = "spite project",

    .supported_quants = {
        SPITE_TYPE_F32,
        SPITE_TYPE_Q8_0,
        SPITE_TYPE_Q4_K,
        0
    },

    .rms_norm            = spite_generic_rms_norm,
    .attention           = spite_generic_attention,
    .mla                 = NULL,
    .ffn                 = spite_generic_ffn,
    .layer               = NULL,
    .speculative_verify  = NULL,
    .prefill             = NULL,
};

const SpiteKernelInfo *spite_kernel_info(void) {
    return &GENERIC_KERNEL_INFO;
}
