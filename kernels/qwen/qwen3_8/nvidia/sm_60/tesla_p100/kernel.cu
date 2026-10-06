/*
 * kernels/qwen/qwen3_8/nvidia/sm_60/tesla_p100/kernel.cu
 *
 * Tesla P100 (GP100, sm_60) card layer for Qwen3.8.
 *
 * Every compute op is left NULL on purpose: the dispatcher resolves each slot
 * down the chain tesla_p100/ -> sm_60/ -> nvidia/, and nothing about the
 * decode ops is P100-specific beyond GP100 itself (sm_60 is GP100-only, so the
 * sm_60 kernel already is the P100 tuning).  A card-level copy of those ops
 * would only be a second place for them to drift.
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
 * Memory budget (one P100, 16 GB HBM2): a 27B dense model at Q4_0 is ~14.5 GB
 * of weights — it fits, with little room for KV cache.  F16 (~54 GB) needs
 * 4-way tensor parallelism across SXM2 cards.
 */

#include "core/abi.h"
#include "kernels/qwen/qwen3_8/nvidia/sm_60/tesla_p100/multi_gpu.cuh"

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
    "qwen38",
    "cuda",
    "spite project (qwen3.8 tesla p100: NVLink TP helpers; ops resolve from sm_60/)",
    {0},
    nullptr, /* rms_norm   -> sm_60 */
    nullptr, /* attention  -> sm_60 */
    nullptr, /* mla */
    nullptr, /* ffn        -> sm_60 */
    nullptr, /* layer */
    nullptr, /* speculative_verify */
    nullptr, /* prefill */
    nullptr, /* matmul     -> sm_60 */
    nullptr, /* kv_cache_kinds: travels with the attention slot */
    nullptr, /* linear_attn */
    nullptr, /* attention_ex */
    nullptr, /* mtp_stem   -> nvidia */
    nullptr, /* moe_ffn */
};

extern "C" const SpiteKernelInfo* spite_kernel_info() { return &KERNEL_INFO; }
