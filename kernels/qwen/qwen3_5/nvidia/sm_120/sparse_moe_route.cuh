/*
 * kernels/qwen/qwen3_5/nvidia/sparse_moe_route.cuh
 *
 * In-register warp-level bitonic top-8 selection and shared expert gating.
 * Ported from NInfer for Qwen3.5 MoE (256 routed experts + 1 shared expert).
 */
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kSparseMoeExperts = 256;
constexpr int kSparseMoeTopK    = 8;
constexpr uint32_t kFullWarpMask = 0xffffffffu;

struct SparseMoeRankedValue {
    float value;
    int id;
};

__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(kFullWarpMask, v, mask);
    }
    return v;
}

// Merges the descending run of each lane with the run of its xor partner and keeps the better half.
// 5 merge steps achieve warp-wide top-8 in registers without global/shared memory roundtrips.
__device__ __forceinline__ void
sparse_moe_merge_ranked_runs(SparseMoeRankedValue (&run)[kSparseMoeTopK]) {
#pragma unroll
    for (int partner = 1; partner < 32; partner <<= 1) {
        SparseMoeRankedValue merged[kSparseMoeTopK];
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            const SparseMoeRankedValue mirror = run[kSparseMoeTopK - 1 - rank];
            SparseMoeRankedValue other;
            other.value  = __shfl_xor_sync(kFullWarpMask, mirror.value, partner);
            other.id     = __shfl_xor_sync(kFullWarpMask, mirror.id, partner);
            merged[rank] = sparse_moe_ranked_better(run[rank], other) ? run[rank] : other;
        }
#pragma unroll
        for (int stride = kSparseMoeTopK / 2; stride > 0; stride >>= 1) {
#pragma unroll
            for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
                if ((rank & stride) != 0) { continue; }
                const int partner_rank = rank | stride;
                if (!sparse_moe_ranked_better(merged[rank], merged[partner_rank])) {
                    const SparseMoeRankedValue swap = merged[rank];
                    merged[rank]                    = merged[partner_rank];
                    merged[partner_rank]            = swap;
                }
            }
        }
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) { run[rank] = merged[rank]; }
    }
}

__device__ __forceinline__ void sparse_moe_select_top8_warp(const float* scores,
                                                            int* ids,
                                                            float* alpha,
                                                            float* shared_scale,
                                                            int has_shared) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[kSparseMoeTopK];
#pragma unroll
    for (int item = 0; item < kSparseMoeTopK; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id};
    }
#pragma unroll
    for (int i = 1; i < kSparseMoeTopK; ++i) {
        const SparseMoeRankedValue value = local[i];
        int position                     = i;
        while (position > 0 && sparse_moe_ranked_better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }
    sparse_moe_merge_ranked_runs(local);

    __shared__ float s_logits[kSparseMoeTopK];
    if (lane < kSparseMoeTopK) {
        ids[lane]      = local[lane].id;
        s_logits[lane] = local[lane].value;
    }
    __syncwarp();

    float exponential = 0.0f;
    if (lane < kSparseMoeTopK) {
        exponential = expf(s_logits[lane] - s_logits[0]);
    }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < kSparseMoeTopK) {
        alpha[lane] = exponential / (denominator > 1e-30f ? denominator : 1e-30f);
    }
    if (lane == 0 && shared_scale) {
        if (has_shared) {
            float s = scores[kSparseMoeExperts];
            *shared_scale = 1.0f / (1.0f + expf(-s));
        } else {
            *shared_scale = 0.0f;
        }
    }
}

} // namespace
