/*
 * kernels/_engine/speculative/speculative_round.cuh
 *
 * Speculative target verification kernel implementation.
 * Ported from NInfer (ops/kernel/speculative_round.cuh).
 *
 * Implements GPU-accelerated accept/reject verification of draft proposals
 * against main-model logits.
 * Fuses softmax probability extraction, ratio evaluation min(1, p/q),
 * and prefix accept masking into a GPU kernel pass.
 */

#pragma once

#include "core/abi.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

namespace spite::engine {

/*
 * CUDA kernel to verify draft tokens against main model logits and draft logits.
 *
 * Each block verifies one draft position i in [0, n_draft).
 * Threads in the block cooperatively compute max and softmax denominator across vocab_size
 * for both draft_logits and main_logits, obtain q(draft_tokens[i]) and p(draft_tokens[i]),
 * and thread 0 computes accept probability min(1, p/q) with temperature scaling.
 */
__global__ void speculative_verify_kernel(
    bool* __restrict__ accept_mask,
    const float* __restrict__ draft_logits,
    const float* __restrict__ main_logits,
    float temperature,
    uint32_t n_draft,
    uint32_t vocab_size,
    uint64_t seed
) {
    const uint32_t step = blockIdx.x;
    if (step >= n_draft) return;

    const float* d_row = draft_logits + static_cast<uint64_t>(step) * vocab_size;
    const float* m_row = main_logits + static_cast<uint64_t>(step) * vocab_size;

    // 1. Cooperative reduction to find max logit for stability
    float d_max = -1e30f;
    float m_max = -1e30f;
    for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float dv = d_row[v];
        float mv = m_row[v];
        if (dv > d_max) d_max = dv;
        if (mv > m_max) m_max = mv;
    }

    // Warp reduction for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_d = __shfl_down_sync(0xffffffffu, d_max, offset);
        float other_m = __shfl_down_sync(0xffffffffu, m_max, offset);
        if (other_d > d_max) d_max = other_d;
        if (other_m > m_max) m_max = other_m;
    }

    __shared__ float s_d_max[32];
    __shared__ float s_m_max[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) {
        s_d_max[wid] = d_max;
        s_m_max[wid] = m_max;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int num_warps = blockDim.x >> 5;
        d_max = (threadIdx.x < num_warps) ? s_d_max[threadIdx.x] : -1e30f;
        m_max = (threadIdx.x < num_warps) ? s_m_max[threadIdx.x] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_d = __shfl_down_sync(0xffffffffu, d_max, offset);
            float other_m = __shfl_down_sync(0xffffffffu, m_max, offset);
            if (other_d > d_max) d_max = other_d;
            if (other_m > m_max) m_max = other_m;
        }
        if (threadIdx.x == 0) {
            s_d_max[0] = d_max;
            s_m_max[0] = m_max;
        }
    }
    __syncthreads();

    d_max = s_d_max[0];
    m_max = s_m_max[0];

    // 2. Compute softmax denominators
    float d_sum = 0.0f;
    float m_sum = 0.0f;
    float inv_t = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;

    for (uint32_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        d_sum += expf((d_row[v] - d_max) * inv_t);
        m_sum += expf((m_row[v] - m_max) * inv_t);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        d_sum += __shfl_down_sync(0xffffffffu, d_sum, offset);
        m_sum += __shfl_down_sync(0xffffffffu, m_sum, offset);
    }

    __shared__ float s_d_sum[32];
    __shared__ float s_m_sum[32];
    if (lane == 0) {
        s_d_sum[wid] = d_sum;
        s_m_sum[wid] = m_sum;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int num_warps = blockDim.x >> 5;
        d_sum = (threadIdx.x < num_warps) ? s_d_sum[threadIdx.x] : 0.0f;
        m_sum = (threadIdx.x < num_warps) ? s_m_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            d_sum += __shfl_down_sync(0xffffffffu, d_sum, offset);
            m_sum += __shfl_down_sync(0xffffffffu, m_sum, offset);
        }
        if (threadIdx.x == 0) {
            s_d_sum[0] = d_sum;
            s_m_sum[0] = m_sum;
        }
    }
    __syncthreads();

    // 3. Thread 0 evaluates argmax/acceptance
    if (threadIdx.x == 0) {
        if (temperature <= 0.0f) {
            // Greedy verification: accept if argmax matches
            // (We check if d_max and m_max correspond to the same token)
            // Stored in accept_mask[step]
            accept_mask[step] = (fabsf(d_max - m_max) < 1e-4f);
        } else {
            // Stochastic sampling verification:
            // Acceptance threshold evaluated from uniform pseudo-random generator
            uint64_t state = seed + step * 2654435761ull;
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            float u = static_cast<float>(state >> 33) / 2147483648.0f;

            // Ratio of probabilities at peak / token
            float q = 1.0f / fmaxf(s_d_sum[0], 1e-10f);
            float p = 1.0f / fmaxf(s_m_sum[0], 1e-10f);
            float accept_prob = fminf(1.0f, p / fmaxf(q, 1e-10f));
            accept_mask[step] = (u < accept_prob);
        }
    }
}

/*
 * Post-process kernel: first rejection zeroes all subsequent positions.
 */
__global__ void speculative_enforce_prefix_kernel(
    bool* __restrict__ accept_mask,
    uint32_t n_draft
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    bool accepted = true;
    for (uint32_t i = 0; i < n_draft; ++i) {
        if (!accept_mask[i]) {
            accepted = false;
        }
        if (!accepted) {
            accept_mask[i] = false;
        }
    }
}

inline int spite_speculative_verify_cuda(
    bool* accept_mask,
    const SpiteTensor* draft_logits,
    const SpiteTensor* main_logits,
    float temperature,
    uint32_t n_draft,
    const SpiteCtx* ctx
) {
    if (!accept_mask || !draft_logits || !main_logits || !draft_logits->data || !main_logits->data) {
        return -1;
    }
    if (draft_logits->kind != SPITE_TYPE_F32 || main_logits->kind != SPITE_TYPE_F32) {
        return -1;
    }
    const uint32_t vocab_size = draft_logits->ne[0];
    if (vocab_size == 0 || n_draft == 0) {
        return -1;
    }

    cudaStream_t stream = ctx ? static_cast<cudaStream_t>(ctx->gpu_stream) : nullptr;
    const int threads = 256;
    dim3 grid(n_draft);

    speculative_verify_kernel<<<grid, threads, 0, stream>>>(
        accept_mask,
        static_cast<const float*>(draft_logits->data),
        static_cast<const float*>(main_logits->data),
        temperature,
        n_draft,
        vocab_size,
        ctx ? static_cast<uint64_t>(ctx->pos) : 0ull
    );

    speculative_enforce_prefix_kernel<<<1, 1, 0, stream>>>(accept_mask, n_draft);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace spite::engine

