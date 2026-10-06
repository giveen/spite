/*
 * kernels/qwen/qwen3_5/nvidia/sm_60/tesla_p100/kernel.cu
 *
 * Tesla P100 (GP100, sm_60) card layer for Qwen3.5-architecture models
 * (hybrid Gated DeltaNet + gated full attention, incl. Qwen3.8 fine-tunes).
 *
 * Every compute op is left NULL on purpose: the dispatcher resolves each slot
 * down the chain tesla_p100/ -> nvidia/, and the vendor Qwen3.5 kernel
 * (rms_norm, attention_ex, linear_attn, ffn, moe_ffn, matmul, mtp_stem)
 * compiles for sm_60 unchanged.  A card-level copy of those ops would only be
 * a second place for them to drift; Pascal-specific tuning belongs in an
 * sm_60/ kernel once it has P100 benchmark numbers behind it.
 *
 * What IS card-specific is NVLink (SXM2 only), so this library exports the
 * tensor-parallel all-reduce helpers from multi_gpu.cuh for the host:
 *
 *   int spite_p100_enable_peer_access(const int* devs, int n);
 *   int spite_p100_allreduce_f32(float* const* bufs, const int* devs,
 *                                const cudaEvent_t* ready, int n, int root,
 *                                int n_elems, float* scratch,
 *                                cudaStream_t stream, cudaEvent_t done);
 *
 * Both return a cudaError_t value (0 = success).  The host-side topology and
 * shard-count policy is crates/spite-parallel/src/p100_multi.rs.
 *
 * Memory budget (one P100, 16 GB HBM2): the 27B Qwen3.8 Q5_K_P build used
 * 19 GB on an RTX 5090 (../../sm_120/rtx_5090/qwen3_5.bench), so one P100 needs
 * roughly <= 4.25 bits/weight (IQ4_XS ~14.3 GB, Q3_K_M ~13 GB of weights) plus
 * KV/state; anything larger needs two or more cards.  Q6_K is ~20.9 GiB and
 * never fits one card — run 2x or 4x P100-PCIE with pipeline splitting
 * (docs/multi-gpu.md).  These PCIe cards have no NVLink, so tensor parallelism
 * is refused and only the layer-wise pipeline is used.
 */

#include "core/abi.h"
#include "kernels/qwen/qwen3_5/nvidia/sm_60/tesla_p100/multi_gpu.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" int spite_p100_enable_peer_access(const int* devs, int n) {
    return static_cast<int>(p100_enable_peer_access(devs, n));
}

extern "C" int spite_p100_allreduce_f32(float* const* bufs, const int* devs,
                                        const cudaEvent_t* ready, int n, int root, int n_elems,
                                        float* scratch, cudaStream_t stream, cudaEvent_t done) {
    return static_cast<int>(
        p100_allreduce_f32(bufs, devs, ready, n, root, n_elems, scratch, stream, done));
}

static const SpiteKernelInfo KERNEL_INFO = {
    SPITE_ABI_VERSION,
    "qwen3_5",
    "cuda",
    "spite project (qwen3.5 tesla p100: NVLink TP helpers; ops resolve from nvidia/)",
    {0},
    nullptr, /* rms_norm   -> nvidia */
    nullptr, /* attention (qwen3.5 uses attention_ex -> nvidia) */
    nullptr, /* mla */
    nullptr, /* ffn        -> nvidia */
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    nullptr, /* matmul     -> nvidia */
    nullptr, /* kv_cache_kinds: travels with the attention slot */
    nullptr, /* linear_attn -> nvidia */
    nullptr, /* attention_ex -> nvidia */
    nullptr, /* mtp_stem   -> nvidia */
    nullptr, /* moe_ffn    -> nvidia */
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
