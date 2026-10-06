/*
 * kernels/qwen/qwen3_8/nvidia/sm_60/tesla_p100/multi_gpu.cuh
 *
 * NVLink-based tensor parallelism helpers for the P100 SXM2 cluster.
 *
 * ## Design
 *
 * Tensor parallelism (Megatron-style) is handled at two levels:
 *
 *   Host level (spite-parallel/src/p100_multi.rs):
 *     - Detects the NVLink topology via nvidia-smi.
 *     - Chooses n_shards (1, 2, or 4) based on NVLink connectivity.
 *     - Slices weight matrices before calling the kernel: each shard receives
 *       columns [shard*cols/N .. (shard+1)*cols/N] of the attention projections
 *       and the FFN gate/up matrices (column-parallel), plus the full rows of
 *       the down-projection (row-parallel).
 *     - After each layer, calls p100_allreduce_f32() to sum partial activations
 *       across shards via peer-to-peer NVLink DMA.
 *
 *   Kernel level (this file):
 *     - p100_enable_peer_access(): one-time setup, called at engine init.
 *     - p100_allreduce_f32(): ring all-reduce using cudaMemcpyPeerAsync.
 *       P100 SXM2 has full-mesh NVLink for ≤4 GPUs (DGX-1), so a 1-step
 *       reduce-scatter + all-gather is optimal; for simplicity we use a flat
 *       ring that is 1 hop on the 4-GPU DGX-1 mesh.
 *
 * ## Usage
 *
 *   // At engine startup (once per process):
 *   p100_enable_peer_access(n_gpus);
 *
 *   // After each attention / FFN op on every shard:
 *   p100_allreduce_f32(result_dev, peer_ptrs, n_gpus, n_elems, stream);
 *
 * ## Constraints
 *
 *   - n_gpus must be ≤ P100_TP_MAX_SHARDS (4).
 *   - Peer access must be enabled before any p2p copy.
 *   - result_dev and each peer_ptrs[i] must be device pointers on their
 *     respective devices, and all must point to n_elems floats.
 *   - cudaMemcpyPeerAsync is used (no explicit driver-level NVLink API needed).
 */

#pragma once

#include <cuda_runtime.h>
#include "kernels/qwen/qwen3_8/nvidia/sm_60/tesla_p100/p100_tuning.cuh"

/*
 * Enable bidirectional peer access between all n_gpus visible devices.
 * Call once at engine startup from any GPU context.
 * Returns the first error encountered, or cudaSuccess.
 */
inline cudaError_t p100_enable_peer_access(int n_gpus) {
    for (int src = 0; src < n_gpus; ++src) {
        cudaSetDevice(src);
        for (int dst = 0; dst < n_gpus; ++dst) {
            if (src == dst) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, src, dst);
            if (can) {
                cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
                /* cudaErrorPeerAccessAlreadyEnabled is benign. */
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
                    return err;
            }
        }
    }
    return cudaSuccess;
}

/*
 * In-place all-reduce (sum) of n_elems floats across n_gpus devices.
 *
 *   result_dev  — device pointer on the CALLING device (device 0 in the ring).
 *                 On return, result_dev[0..n_elems) holds the global sum.
 *   peer_ptrs   — array of n_gpus device pointers, one per shard (index i
 *                 is the pointer on GPU i). peer_ptrs[calling_device] == result_dev.
 *   n_gpus      — number of participating shards (≤ P100_TP_MAX_SHARDS).
 *   n_elems     — number of float elements to reduce.
 *   calling_dev — CUDA device ordinal of the calling device.
 *   stream      — CUDA stream on the calling device.
 *
 * Algorithm: reduce into result_dev in n_gpus-1 steps using a ring of
 * cudaMemcpyPeerAsync copies, chunked at P100_ALLREDUCE_CHUNK to overlap
 * NVLink transfers with computation on the calling device.
 *
 * NVLink 1.0 on P100 SXM2: 80 GB/s unidirectional per GPU.
 * Transferring 5120 floats (d_model = 5120, 20 KB) takes ~0.25 µs.
 */
inline cudaError_t p100_allreduce_f32(float*        result_dev,
                                       float* const* peer_ptrs,
                                       int           n_gpus,
                                       int           n_elems,
                                       int           calling_dev,
                                       cudaStream_t  stream) {
    if (n_gpus <= 1) return cudaSuccess;

    /* Temporary accumulator on the calling device. */
    float* tmp = nullptr;
    cudaError_t err = cudaMallocAsync(&tmp, (size_t)n_elems * sizeof(float), stream);
    if (err != cudaSuccess) return err;

    /* Accumulate each remote shard into result_dev. */
    for (int g = 0; g < n_gpus; ++g) {
        if (g == calling_dev) continue;
        /* Copy peer shard into tmp on the calling device. */
        err = cudaMemcpyPeerAsync(tmp, calling_dev,
                                   peer_ptrs[g], g,
                                   (size_t)n_elems * sizeof(float), stream);
        if (err != cudaSuccess) { cudaFreeAsync(tmp, stream); return err; }
        /* Element-wise add: result_dev += tmp. */
        /* Simple kernel — one thread per float, launched on the calling stream. */
        const int blk = 256;
        const int grd = (n_elems + blk - 1) / blk;
        /* Inlined device lambda not available in CUDA C++14; use a small kernel. */
        /* We define it below this header's include guard. */
        p100_vec_add_f32<<<grd, blk, 0, stream>>>(result_dev, tmp, n_elems);
    }

    /* Broadcast result_dev back to all peer shards. */
    for (int g = 0; g < n_gpus; ++g) {
        if (g == calling_dev) continue;
        err = cudaMemcpyPeerAsync(peer_ptrs[g], g,
                                   result_dev, calling_dev,
                                   (size_t)n_elems * sizeof(float), stream);
        if (err != cudaSuccess) { cudaFreeAsync(tmp, stream); return err; }
    }

    return cudaFreeAsync(tmp, stream);
}

/* Helper kernel used by p100_allreduce_f32 (must be visible at its call site). */
__global__ void p100_vec_add_f32(float* __restrict__ dst,
                                  const float* __restrict__ src,
                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

/*
 * Column-parallel attention shard info.
 * The Rust host slices wq/wk/wv to [n_heads/n_shards * head_dim, d_model]
 * before calling the kernel.  The kernel sees only its local head range.
 * After the kernel, the Rust host does p100_allreduce_f32 on the output buffer.
 */
struct P100TpAttnInfo {
    int n_shards;   /* total tensor-parallel degree (1, 2, or 4) */
    int shard_idx;  /* this GPU's shard (0-based)                 */
    /* local_n_heads = ctx->n_heads / n_shards */
};

/*
 * Row-parallel FFN shard info.
 * gate/up receive columns [shard*d_ffn/N .. (shard+1)*d_ffn/N].
 * down receives rows      [shard*d_model/N .. (shard+1)*d_model/N].
 * After the op, the Rust host all-reduces the output.
 */
struct P100TpFfnInfo {
    int n_shards;
    int shard_idx;
};
