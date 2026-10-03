/*
 * spite kernel — llama3 / sm_89 (RTX 4070, 4080, 4090, RTX 4000 Ada)
 *
 * HOW TO USE THIS FILE
 * ────────────────────
 * 1. Copy this file to kernels/<model>/<your_gpu_arch>/<op_name>.cu
 * 2. Fill in the ops you want to optimize. Leave the rest returning -1
 *    (the dispatcher will fall back to the generic implementation).
 * 3. Run:  spite verify kernels/<model>/<arch>/your_file.cu
 * 4. Run:  spite bench   kernels/<model>/<arch>/your_file.cu
 * 5. Include your bench output in the PR description.
 *
 * GPU ARCHITECTURE NOTES — sm_89 (Ada Lovelace)
 * ───────────────────────────────────────────────
 * - Tensor cores: FP8, FP16, BF16, INT8, INT4
 *   mma shape: m16n8k16 (FP16), m16n8k32 (INT8), m16n8k32 (FP8)
 * - L2 cache: 72 MB (4090) — large enough to cache Q/K/V projections
 * - Shared memory per SM: 100 KB
 * - Register file: 65536 x 32-bit per SM
 * - Warp size: 32 threads
 * - Max threads per block: 1024
 * - Good for: fused attention (FlashAttention-style), Q4_K matmul
 *
 * Q4_K BLOCK LAYOUT REMINDER (see core/quant.h)
 * ──────────────────────────────────────────────
 * Each block_q4_K covers 256 values:
 *   d, dmin  — two fp16 super-scales
 *   scales[] — 12 bytes encoding 8 x 6-bit sub-block scales + mins
 *   qs[]     — 128 bytes of packed 4-bit quants (2 per byte)
 *
 * To dequantize value i within a block:
 *   sub_block = i / 32
 *   scale = decode_scale(block.scales, sub_block)   // 6-bit
 *   min   = decode_min  (block.scales, sub_block)
 *   q     = (qs[i/2] >> (4*(i%2))) & 0xF
 *   val   = float(d) * (scale * q - min * float(dmin))
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../../core/abi.h"
#include "../../../core/quant.h"

/* ── Helper: decode 6-bit sub-block scale from block_q4_K.scales[] ──── */
static __device__ __forceinline__
float decode_scale(const uint8_t *scales, int sub_block) {
    /* TODO: implement — see GGUF spec or core/quant.h comments */
    (void)scales; (void)sub_block;
    return 1.0f;
}

/* ── RMS Norm ──────────────────────────────────────────────────────────
 *
 * out[i] = x[i] / rms(x) * weight[i]
 * rms(x) = sqrt( mean(x^2) + eps )
 *
 * One block per row. Threads in a warp reduce over the hidden dim.
 * Typical hidden dim for llama3-8b: 4096.
 */
static __global__ void rms_norm_kernel(
    float        *out,
    const float  *x,
    const float  *weight,
    int           hidden_dim,
    float         eps
) {
    /* TODO: implement — this is a good first kernel to write */
    (void)out; (void)x; (void)weight; (void)hidden_dim; (void)eps;
}

static int rms_norm(
    spite_tensor_t       *out,
    const spite_tensor_t *x,
    const spite_tensor_t *weight,
    float                 eps,
    const spite_ctx_t    *ctx
) {
    /* TODO: launch rms_norm_kernel */
    (void)out; (void)x; (void)weight; (void)eps; (void)ctx;
    return -1; /* -1 = not implemented, dispatcher uses fallback */
}

/* ── FFN: gate/up/down with SiLU ───────────────────────────────────────
 *
 * out = down( silu(gate(x)) * up(x) )
 *
 * For Q4_K weights on sm_89, the inner loop is:
 *   1. Load a tile of x into shared memory (FP16 or FP32)
 *   2. Dequantize a tile of the weight matrix from Q4_K blocks
 *   3. Accumulate with tensor cores (INT8 or FP16 path)
 *
 * Llama3-8b dimensions:
 *   hidden_dim  = 4096
 *   ffn_dim     = 14336  (intermediate)
 */
static int ffn(
    spite_tensor_t       *out,
    const spite_tensor_t *x,
    const spite_tensor_t *w_gate,
    const spite_tensor_t *w_up,
    const spite_tensor_t *w_down,
    const spite_ctx_t    *ctx
) {
    /* TODO: implement Q4_K matmul + SiLU fusion */
    (void)out; (void)x; (void)w_gate; (void)w_up; (void)w_down; (void)ctx;
    return -1;
}

/* ── Attention ─────────────────────────────────────────────────────────
 *
 * Llama3-8b:
 *   n_heads    = 32, n_kv_heads = 8  (grouped query attention)
 *   head_dim   = 128
 *   rope_theta = 500000
 *
 * A FlashAttention-2 style kernel is ideal here:
 *   - Tiled Q/K/V computation, never materializes full attention matrix
 *   - sm_89 L2 is large enough to keep K/V tiles hot across queries
 *
 * For a first contribution, a naive but correct implementation is fine.
 */
static int attention(
    spite_tensor_t       *out,
    const spite_tensor_t *x,
    const spite_tensor_t *wq,
    const spite_tensor_t *wk,
    const spite_tensor_t *wv,
    const spite_tensor_t *wo,
    spite_kvcache_t      *kvcache,
    int                   pos,
    float                 rope_freq_base,
    const spite_ctx_t    *ctx
) {
    /* TODO: implement */
    (void)out; (void)x; (void)wq; (void)wk; (void)wv; (void)wo;
    (void)kvcache; (void)pos; (void)rope_freq_base; (void)ctx;
    return -1;
}

/* ── Kernel descriptor ─────────────────────────────────────────────────
 *
 * Edit author, supported_quants. Leave ops NULL if not implemented.
 */
static const spite_kernel_info_t kernel_info = {
    .abi_version      = SPITE_ABI_VERSION,
    .model_arch       = "llama3",
    .gpu_arch         = "sm_89",
    .author           = "your name / handle here",

    .supported_quants = { SPITE_TYPE_Q4_K, SPITE_TYPE_Q8_0, 0 },

    .rms_norm         = NULL,    /* set to rms_norm once implemented */
    .attention        = NULL,
    .ffn              = NULL,
    .layer            = NULL,    /* optional: fuse all ops into one kernel */
};

const spite_kernel_info_t *spite_kernel_info(void) {
    return &kernel_info;
}
