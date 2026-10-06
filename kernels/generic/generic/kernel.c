/*
 * kernels/generic/generic/kernel.c
 *
 * Generic fallback kernel — always compiled, always present.
 * Exports `spite_kernel_info()` so the dispatcher can dlopen this .so.
 *
 * This kernel handles:
 *   rms_norm  — scalar C, any weight type (spite_dequantize_row)
 *   ffn       — scalar C, any weight type (spite_dequantize_row)
 *   attention — scalar C, F32 KV only (the numeric reference for GPU
 *               attention; see the note in ops.c). `kv_cache_kinds` stays
 *               NULL, which the host reads as "F32 tiers only".
 *   matmul    — scalar C, any weight type (spite_dequantize_row) (LM head)
 *   attention_ex — scalar C attention with partial RoPE + gated Q (ABI v7);
 *               `attention` is the same code with gated_q=0, rope_dim=head_dim
 *   linear_attn — scalar C Gated Delta Net layer, projections included (ABI v7,
 *               double accumulation)
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
    const SpiteTensor*, const SpiteTensor*, float,
    SpiteKvCache*, float, const SpiteCtx*);
int spite_generic_attention_ex(
    SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, float,
    SpiteKvCache*, float, const SpiteAttnParams*, const SpiteCtx*);
int spite_generic_matmul(
    SpiteTensor*, const SpiteTensor*, const SpiteTensor*, const SpiteCtx*);

int spite_generic_linear_attn(
    SpiteTensor*, const SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    SpiteTensor*, SpiteTensor*, const SpiteGdnParams*, const SpiteCtx*);

int spite_generic_mtp_stem(
    SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, float, const SpiteCtx*);

int spite_generic_moe_ffn(
    SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteTensor*, const SpiteTensor*, const SpiteTensor*,
    const SpiteMoeParams*, const SpiteCtx*);

static const SpiteKernelInfo GENERIC_KERNEL_INFO = {
    .abi_version = SPITE_ABI_VERSION,
    .model_arch  = "generic",
    .gpu_arch    = "generic",
    .author      = "spite project",

    /* The ops handle every type spite_dequantize_row() knows (all GGUF quants),
     * but the array has only 8 slots (0-terminated => at most 7 entries), so
     * advertise the commonly shipped ones. F32 (== 0) cannot be listed because 0
     * terminates the list; it is always supported. */
    .supported_quants = {
        SPITE_TYPE_F16,
        SPITE_TYPE_BF16,
        SPITE_TYPE_Q8_0,
        SPITE_TYPE_Q4_0,
        SPITE_TYPE_Q4_K,
        SPITE_TYPE_Q6_K,
        0
    },

    .rms_norm            = spite_generic_rms_norm,
    .attention           = spite_generic_attention,
    .mla                 = NULL,
    .ffn                 = spite_generic_ffn,
    .layer               = NULL,
    .speculative_verify  = NULL,
    .prefill             = NULL,
    .matmul              = spite_generic_matmul,
    .kv_cache_kinds      = NULL,
    .linear_attn         = spite_generic_linear_attn,
    .attention_ex        = spite_generic_attention_ex,
    .mtp_stem            = spite_generic_mtp_stem,
    .moe_ffn             = spite_generic_moe_ffn,
};

const SpiteKernelInfo *spite_kernel_info(void) {
    return &GENERIC_KERNEL_INFO;
}
