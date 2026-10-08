/*
 * kernels/qwen/qwen3_5/nvidia/sm_60/tesla_p100/multi_gpu.cuh
 *
 * NVLink tensor-parallel helpers for Tesla P100 SXM2 groups (<= 4 GPUs).
 *
 * The compute ops need nothing special for tensor parallelism: the host hands
 * each GPU its weight shard (column-split wq/wk/wv/gate/up, row-split wo/down,
 * ctx->n_heads / n_kv_heads divided by the shard count) and runs the ordinary
 * sm_60 ops on it.  What sharding adds is one all-reduce per attention and FFN
 * op, which is what this file provides (exported as C from kernel.cu).
 *
 * Residual: attention and ffn ACCUMULATE (out += op(x), ABI v4).  Summing
 * shards that each did `out += partial` would add the residual n_gpus times,
 * so the host must run non-root shards on a zeroed `out` (or a scratch
 * buffer) and let only the root shard carry the residual.
 *
 * Synchronisation: every GPU produces its partial on its own stream, so the
 * reducer waits on one event per peer before reading it, and records `done`
 * after the broadcast; peers must cudaStreamWaitEvent(done) before reading
 * the reduced result.
 *
 * Algorithm: gather-to-root + broadcast over cudaMemcpyPeerAsync.  For decode
 * activations (one d_model row, ~20 KB) the transfers are latency-bound and
 * this is as fast as a ring; it is not meant for large (prefill) tensors.
 */

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

#include "kernels/qwen/qwen3_5/nvidia/sm_60/tesla_p100/p100_tuning.cuh"

static __global__ void p100_vec_add_f32(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

/* Enable peer access between every pair of devs[0..n).  Restores the caller's
 * current device.  Pairs without peer capability are skipped (they would fall
 * back to staged copies, which cudaMemcpyPeerAsync handles transparently). */
inline cudaError_t p100_enable_peer_access(const int* devs, int n) {
    int prev = 0;
    cudaError_t err = cudaGetDevice(&prev);
    if (err != cudaSuccess) return err;
    for (int a = 0; a < n && err == cudaSuccess; ++a) {
        err = cudaSetDevice(devs[a]);
        for (int b = 0; b < n && err == cudaSuccess; ++b) {
            if (a == b) continue;
            int can = 0;
            err = cudaDeviceCanAccessPeer(&can, devs[a], devs[b]);
            if (err != cudaSuccess || !can) continue;
            err = cudaDeviceEnablePeerAccess(devs[b], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();  // clear the sticky benign error
                err = cudaSuccess;
            }
        }
    }
    const cudaError_t rs = cudaSetDevice(prev);
    return err != cudaSuccess ? err : rs;
}

/*
 * In-place sum of n_elems floats across n_gpus shards; on completion every
 * bufs[g] holds the total.
 *
 *   bufs[g]   device pointer on devs[g]
 *   ready[g]  event recorded after shard g's partial was written (NULL entries
 *             or a NULL array skip the wait, e.g. when the root produced it on
 *             `stream`)
 *   root      index (into bufs/devs) of the reducing GPU; `stream` and
 *             `scratch` (n_elems floats) live on devs[root]
 *   done      optional event recorded on `stream` after the broadcast
 */
inline cudaError_t p100_allreduce_f32(float* const* bufs, const int* devs,
                                      const cudaEvent_t* ready, int n_gpus, int root,
                                      int n_elems, float* scratch, cudaStream_t stream,
                                      cudaEvent_t done) {
    if (n_gpus <= 1) return cudaSuccess;
    if (n_gpus > P100_TP_MAX_SHARDS || root < 0 || root >= n_gpus || n_elems < 0 || !scratch)
        return cudaErrorInvalidValue;
    int prev = 0;
    cudaError_t err = cudaGetDevice(&prev);
    if (err != cudaSuccess) return err;
    err = cudaSetDevice(devs[root]);

    const size_t bytes = static_cast<size_t>(n_elems) * sizeof(float);
    const int blk = 256;
    const int grd = (n_elems + blk - 1) / blk;
    for (int g = 0; g < n_gpus && err == cudaSuccess; ++g) {
        if (g == root) continue;
        if (ready && ready[g]) err = cudaStreamWaitEvent(stream, ready[g], 0);
        if (err == cudaSuccess)
            err = cudaMemcpyPeerAsync(scratch, devs[root], bufs[g], devs[g], bytes, stream);
        if (err == cudaSuccess && grd > 0) {
            p100_vec_add_f32<<<grd, blk, 0, stream>>>(bufs[root], scratch, n_elems);
            err = cudaGetLastError();
        }
    }
    for (int g = 0; g < n_gpus && err == cudaSuccess; ++g) {
        if (g == root) continue;
        err = cudaMemcpyPeerAsync(bufs[g], devs[g], bufs[root], devs[root], bytes, stream);
    }
    if (err == cudaSuccess && done) err = cudaEventRecord(done, stream);

    const cudaError_t rs = cudaSetDevice(prev);
    return err != cudaSuccess ? err : rs;
}
