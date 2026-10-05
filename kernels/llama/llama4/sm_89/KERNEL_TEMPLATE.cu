// spite kernel — llama4 / sm_89 (RTX 4070, 4080, 4090, RTX 4000 Ada)
// C++23 · CUDA 12.3+
//
// HOW TO USE THIS FILE
// ────────────────────
// 1. cp kernels/llama/llama4/sm_89/KERNEL_TEMPLATE.cu \
//       kernels/llama/llama4/<your_gpu>/<op_name>.cu
// 2. Change gpu_arch in KERNEL_INFO at the bottom.
// 3. Implement one op. Leave the rest returning -1 — the dispatcher
//    uses the fallback for those.
// 4. Build the kernels for your card:
//      cmake -B build -DSPITE_MODELS="llama/llama4" -DSPITE_GPU_ARCHS="<your_gpu>" \
//        && cmake --build build
// 5. Verify the built .so against the generic reference:
//      python3 tools/verify/verify.py \
//        build/kernels/llama/llama4/<your_gpu>/libkernel_llama_llama4_<your_gpu>.so
// 6. Benchmark:
//      cargo run --release -p spite-bench -- --model path/to/model.gguf
// 7. Paste bench output in your PR description.
//
// GPU ARCHITECTURE NOTES — sm_89 (Ada Lovelace)
// ───────────────────────────────────────────────
//  Tensor cores  FP8, FP16, BF16, INT8, INT4
//  MMA shape     m16n8k16 (FP16)  m16n8k32 (INT8 / FP8)
//  Shared mem    100 KB / SM (stay ≤ 50 KB for 2-block occupancy)
//  L2 cache      72 MB (4090)  36 MB (4070)
//  Registers     65536 × 32-bit / SM
//  Bandwidth     1008 GB/s (4090)  504 GB/s (4070)
//
// Q4_K BLOCK LAYOUT (see core/quant.h)
// ──────────────────────────────────────
//  block_q4_K covers 256 values:
//    d, dmin    — two fp16 super-scales
//    scales[12] — 6-bit sub-block scales + mins, packed
//    qs[128]    — packed 4-bit quants (2 per byte)
//
//  To dequantize value i:
//    sb    = i / 32                           // sub-block index (0..7)
//    scale = decode_scale(block.scales, sb)   // 6-bit
//    min   = decode_min  (block.scales, sb)
//    q     = (qs[i/2] >> (4*(i%2))) & 0xF
//    val   = float(d) * (scale * q - min * float(dmin))

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/std/span>
#include <cuda/std/expected>
#include <cuda/std/cstdint>
#include "../../../../core/abi.h"
#include "../../../../core/quant.h"

// ── C++23 / libcudacxx conveniences ───────────────────────────────────────

using cuda::std::span;
using cuda::std::expected;
using cuda::std::unexpected;

// Immutable view over a tensor's raw bytes — no ownership, no copy.
template<typename T>
using TensorView = span<const T>;

// ── Q4_K decode helpers ───────────────────────────────────────────────────

// Extract the 6-bit scale for sub-block `sb` from block_q4_K.scales[].
// See GGUF spec §3.4 for the exact bit packing.
[[nodiscard]] static __device__ __forceinline__
float decode_scale(const uint8_t* scales, int sb) {
    // TODO: implement — two possible bit positions depending on sb parity
    (void)scales; (void)sb;
    return 1.0f;
}

[[nodiscard]] static __device__ __forceinline__
float decode_min(const uint8_t* scales, int sb) {
    (void)scales; (void)sb;
    return 0.0f;
}

// ── RMS Norm ──────────────────────────────────────────────────────────────
//
// out[i] = x[i] / sqrt(mean(x²) + eps) * weight[i]
//
// Launch: one block per row, 128 threads.
// Each thread handles hidden_dim/128 elements.
// Warp-level reduction via __shfl_xor_sync (no shared mem needed).
//
// Example 8B: hidden_dim = 4096  →  32 elements per thread at 128 threads.

static __global__ void rms_norm_kernel(
    float* __restrict__       out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int   hidden_dim,
    float eps
) {
    // TODO: implement
    (void)out; (void)x; (void)weight; (void)hidden_dim; (void)eps;
}

static int rms_norm(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* weight,
    float              eps,
    const SpiteCtx*    ctx
) {
    // TODO: launch rms_norm_kernel
    (void)out; (void)x; (void)weight; (void)eps; (void)ctx;
    return -1; // -1 → not implemented, dispatcher falls back
}

// ── FFN: gate/up projections + SiLU + down projection ────────────────────
//
//  out = down_proj( silu(gate_proj(x)) ⊙ up_proj(x) )
//
//  For Q4_K weights on sm_89:
//    1. Load tile of x into shared memory (FP16)
//    2. Dequantize weight tile from Q4_K blocks → INT8 or FP16
//    3. Accumulate with m16n8k32 (INT8) tensor cores
//
//  Example 8B dims:
//    hidden_dim = 4096
//    ffn_dim    = 14336

static int ffn(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* w_gate,
    const SpiteTensor* w_up,
    const SpiteTensor* w_down,
    SpiteFfnActivation activation,
    const SpiteCtx*    ctx
) {
    // TODO: implement Q4_K matmul + SiLU fusion
    (void)out; (void)x; (void)w_gate; (void)w_up; (void)w_down; (void)activation; (void)ctx;
    return -1;
}

// ── Attention ─────────────────────────────────────────────────────────────
//
//  Grouped query attention (GQA):
//    n_heads    = 32,  n_kv_heads = 8,  head_dim = 128
//    rope_theta = 500000  (modern LLaMA-family base)
//
//  Recommended approach: FlashAttention-2 style tiled kernel.
//  The 72 MB L2 on the 4090 fits all K/V for 4k context at FP16 —
//  structure your tile sizes to exploit that.
//
//  A correct-but-naive implementation is fine for a first contribution.

static int attention(
    SpiteTensor*       out,
    const SpiteTensor* x,
    const SpiteTensor* wq,
    const SpiteTensor* wk,
    const SpiteTensor* wv,
    const SpiteTensor* wo,
    const SpiteTensor* q_norm,     // optional per-head QK norm (NULL if none)
    const SpiteTensor* k_norm,
    float              norm_eps,
    SpiteKvCache*      kvcache,
    float              rope_freq_base,
    const SpiteCtx*    ctx
) {
    // TODO: implement. ABI v4: accumulate into out (out += attn(x)).
    (void)out; (void)x; (void)wq; (void)wk; (void)wv; (void)wo;
    (void)q_norm; (void)k_norm; (void)norm_eps;
    (void)kvcache; (void)rope_freq_base; (void)ctx;
    return -1;
}

// ── Kernel descriptor ─────────────────────────────────────────────────────
//
// Edit: author, gpu_arch if porting to another card, supported_quants.
// Set op pointers to the function once you've implemented them.

static constexpr SpiteKernelInfo KERNEL_INFO {
    .abi_version      = SPITE_ABI_VERSION,
    .model_arch       = "llama4",
    .gpu_arch         = "sm_89",
    .author           = "your name / handle here",

    .supported_quants = {
        static_cast<uint32_t>(SPITE_TYPE_Q4_K),
        static_cast<uint32_t>(SPITE_TYPE_Q8_0),
        0
    },

    .rms_norm         = nullptr,   // set to rms_norm once implemented
    .attention        = nullptr,
    .mla              = nullptr,   // DeepSeek-style latent attention (optional)
    .ffn              = nullptr,
    .layer            = nullptr,   // optional: fuse the whole layer
    .speculative_verify = nullptr,
    .prefill          = nullptr,
    .matmul           = nullptr,   // dense projection (LM head), ABI v4
};

extern "C" [[nodiscard]]
const SpiteKernelInfo* spite_kernel_info() {
    return &KERNEL_INFO;
}
