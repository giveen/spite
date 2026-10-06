/*
 * kernels/qwen/qwen3_5/nvidia/moe.cu
 *
 * MoE FFN layer execution for Qwen3.5 MoE models (ABI v7).
 * Top-8 bitonic routing over 256 experts + 1 shared expert.
 */
#include "common.h"
#include "sparse_moe_route.cuh"

namespace {

__global__ void moe_route_kernel(const float* __restrict__ scores,
                                 int* __restrict__ ids,
                                 float* __restrict__ alpha,
                                 float* __restrict__ shared_scale,
                                 int has_shared) {
    sparse_moe_select_top8_warp(scores, ids, alpha, shared_scale, has_shared);
}

__global__ void swiglu_scale_kernel(float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    float scale,
                                    int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float u = up[idx];
        float silu_g = g / (1.0f + expf(-g));
        gate[idx] = scale * (silu_g * u);
    }
}

} // namespace

extern "C" int qwen35_cuda_moe_ffn(
    SpiteTensor*          out,
    const SpiteTensor*    x,
    const SpiteTensor*    w_gate_inp,
    const SpiteTensor*    w_up_exps,
    const SpiteTensor*    w_gate_exps,
    const SpiteTensor*    w_down_exps,
    const SpiteTensor*    w_up_shexp,
    const SpiteTensor*    w_gate_shexp,
    const SpiteTensor*    w_down_shexp,
    const SpiteMoeParams* params,
    const SpiteCtx*       ctx) {
    if (!out || !x || !w_gate_inp || !w_up_exps || !w_gate_exps || !w_down_exps ||
        !params || !ctx || !out->data || !x->data || !w_gate_inp->data ||
        !w_up_exps->data || !w_gate_exps->data || !w_down_exps->data ||
        !ctx->scratchpad) {
        return -1;
    }
    if (x->kind != SPITE_TYPE_F32 || out->kind != SPITE_TYPE_F32) {
        return -1;
    }

    const int n_expert = params->num_experts;
    const int n_used = params->num_experts_per_tok;
    const int intermediate = params->intermediate_size;
    const int shared_intermediate = params->shared_intermediate_size;
    const bool has_shared = (w_up_shexp && w_gate_shexp && w_down_shexp &&
                             w_up_shexp->data && w_gate_shexp->data &&
                             w_down_shexp->data && shared_intermediate > 0);

    const int router_rows = static_cast<int>(w_gate_inp->ne[1]);
    if (router_rows < n_expert) {
        return -1;
    }

    cudaStream_t s = q35_stream(ctx);

    // Scratchpad layout:
    // scores: [router_rows] floats
    // ids: [8] ints
    // alpha: [8] floats
    // shared_scale: [1] float
    // d_gate: [intermediate] floats
    // d_up:   [intermediate] floats
    // d_sh_gate: [shared_intermediate] floats
    // d_sh_up:   [shared_intermediate] floats
    size_t offset = 0;
    float* d_scores = reinterpret_cast<float*>(static_cast<char*>(ctx->scratchpad) + offset);
    offset += router_rows * sizeof(float);

    int* d_ids = reinterpret_cast<int*>(static_cast<char*>(ctx->scratchpad) + offset);
    offset += 8 * sizeof(int);

    float* d_alpha = reinterpret_cast<float*>(static_cast<char*>(ctx->scratchpad) + offset);
    offset += 8 * sizeof(float);

    float* d_shared_scale = reinterpret_cast<float*>(static_cast<char*>(ctx->scratchpad) + offset);
    offset += sizeof(float);

    float* d_gate = reinterpret_cast<float*>(static_cast<char*>(ctx->scratchpad) + offset);
    offset += intermediate * sizeof(float);

    float* d_up = reinterpret_cast<float*>(static_cast<char*>(ctx->scratchpad) + offset);
    offset += intermediate * sizeof(float);

    float* d_sh_gate = nullptr;
    float* d_sh_up = nullptr;
    if (has_shared) {
        d_sh_gate = reinterpret_cast<float*>(static_cast<char*>(ctx->scratchpad) + offset);
        offset += shared_intermediate * sizeof(float);

        d_sh_up = reinterpret_cast<float*>(static_cast<char*>(ctx->scratchpad) + offset);
        offset += shared_intermediate * sizeof(float);
    }

    if (offset > ctx->scratchpad_bytes) {
        return -2;
    }

    const float* xin = static_cast<const float*>(x->data);
    float* yout = static_cast<float*>(out->data);

    // 1. Router projection: scores = W_gate_inp . x
    if (q35_gemv(w_gate_inp, xin, d_scores, false, s) != 0) {
        return -1;
    }

    // 2. Warp bitonic top-8 selection and shared gate
    moe_route_kernel<<<1, 32, 0, s>>>(d_scores, d_ids, d_alpha, d_shared_scale, has_shared ? 1 : 0);

    // Synchronize selected expert indices to host
    int host_ids[8];
    float host_alpha[8];
    float host_shared_scale = 0.0f;
    cudaMemcpyAsync(host_ids, d_ids, 8 * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(host_alpha, d_alpha, 8 * sizeof(float), cudaMemcpyDeviceToHost, s);
    if (has_shared) {
        cudaMemcpyAsync(&host_shared_scale, d_shared_scale, sizeof(float), cudaMemcpyDeviceToHost, s);
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return -1;
    }

    // 3. Routed expert evaluation
    const int block = 256;
    const int swiglu_grid = (intermediate + block - 1) / block;

    for (int k = 0; k < n_used && k < 8; ++k) {
        int e = host_ids[k];
        if (e < 0 || e >= n_expert) continue;
        float a = host_alpha[k];

        SpiteTensor ue = *w_up_exps;
        ue.data = static_cast<char*>(w_up_exps->data) + static_cast<size_t>(e) * w_up_exps->nb[2];
        ue.ne[2] = 1; ue.ne[3] = 1;

        SpiteTensor ge = *w_gate_exps;
        ge.data = static_cast<char*>(w_gate_exps->data) + static_cast<size_t>(e) * w_gate_exps->nb[2];
        ge.ne[2] = 1; ge.ne[3] = 1;

        SpiteTensor de = *w_down_exps;
        de.data = static_cast<char*>(w_down_exps->data) + static_cast<size_t>(e) * w_down_exps->nb[2];
        de.ne[2] = 1; de.ne[3] = 1;

        Q35GemvJob jobs[2] = { { &ge, d_gate }, { &ue, d_up } };
        if (q35_gemv_multi(jobs, 2, xin, false, s) != 0) {
            return -1;
        }

        swiglu_scale_kernel<<<swiglu_grid, block, 0, s>>>(d_gate, d_up, a, intermediate);

        // Accumulate into out
        if (q35_gemv(&de, d_gate, yout, true, s) != 0) {
            return -1;
        }
    }

    // 4. Shared expert evaluation (if present)
    if (has_shared && host_shared_scale > 0.0f) {
        const int sh_grid = (shared_intermediate + block - 1) / block;
        Q35GemvJob sh_jobs[2] = { { w_gate_shexp, d_sh_gate }, { w_up_shexp, d_sh_up } };
        if (q35_gemv_multi(sh_jobs, 2, xin, false, s) != 0) {
            return -1;
        }

        swiglu_scale_kernel<<<sh_grid, block, 0, s>>>(d_sh_gate, d_sh_up, host_shared_scale, shared_intermediate);

        // Accumulate into out
        if (q35_gemv(w_down_shexp, d_sh_gate, yout, true, s) != 0) {
            return -1;
        }
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
