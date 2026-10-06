/*
 * kernels/qwen/qwen3_8/nvidia/sm_60/p100_fp16.cuh
 *
 * Pascal (GP100, sm_60) FP16 helpers.
 *
 * GP100 has native 2x-throughput FP16 arithmetic via paired __half2 SIMD ops
 * but NO Tensor Cores (those arrived with Volta, sm_70).  Everything here is
 * plain CUDA C++ that compiles on any sm_60+ target.
 *
 * Exposed to the sm_60/kernel.cu and tesla_p100/kernel.cu via a direct include.
 */

#pragma once

#include <cuda_fp16.h>

/*
 * Packed fp16 dot product: dot(a[0..n), b[0..n)) with f32 accumulation.
 * Processes two elements per step using __half2 SIMD.
 * n must be even; caller ensures alignment.
 */
__device__ __forceinline__ float h2_dot_f32(const __half* __restrict__ a,
                                             const __half* __restrict__ b,
                                             int n) {
    float acc = 0.0f;
    const __half2* a2 = reinterpret_cast<const __half2*>(a);
    const __half2* b2 = reinterpret_cast<const __half2*>(b);
    const int n2 = n / 2;
    for (int i = 0; i < n2; ++i) {
        const __half2 p = __hmul2(a2[i], b2[i]);
        acc += __half2float(p.x) + __half2float(p.y);
    }
    if (n & 1) acc += __half2float(a[n - 1]) * __half2float(b[n - 1]);
    return acc;
}

/*
 * Packed warp-reduction matvec for fp16 weight matrices (column-major rows).
 * Each warp handles one output row; lanes stride over columns using __half2.
 * Accumulates into y[row] in fp32.
 *
 * ROWS_PER_BLOCK_H2: number of output rows per thread block (= blockDim.y).
 */
template <int ROWS_H2>
__global__ void matvec_f16_h2(const __half* __restrict__ w,
                               const float*  __restrict__ x,
                               float*        __restrict__ y,
                               int rows, int cols, int accumulate) {
    const int row = blockIdx.x * ROWS_H2 + threadIdx.y;
    if (row >= rows) return;
    const __half* wr = w + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    /* Each lane owns a stride of 2 elements (packed __half2). */
    const int lane_start = threadIdx.x * 2;
    for (int c = lane_start; c + 1 < cols; c += 32 * 2) {
        const __half2 wv = *reinterpret_cast<const __half2*>(wr + c);
        const float   x0 = x[c], x1 = x[c + 1];
        acc += __half2float(wv.x) * x0 + __half2float(wv.y) * x1;
    }
    /* Warp-reduce */
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (threadIdx.x == 0) y[row] = accumulate ? y[row] + acc : acc;
}

/*
 * RMSNorm using __half2 weight reads (2x bandwidth vs scalar __half load).
 * Activations remain f32.
 */
__global__ void rms_norm_f16w_h2(float* __restrict__ out, const float* __restrict__ x,
                                   const __half* __restrict__ w, int cols, float eps) {
    const float* xr = x   + static_cast<size_t>(blockIdx.x) * cols;
    float*       or_ = out + static_cast<size_t>(blockIdx.x) * cols;
    /* Sum of squares */
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    /* block reduce */
    __shared__ float red[32];
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    float v = ss;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float scale = rsqrtf(red[0] / cols + eps);
    /* Apply weight with __half2 loads */
    const __half2* w2 = reinterpret_cast<const __half2*>(w);
    for (int i = threadIdx.x * 2; i + 1 < cols; i += blockDim.x * 2) {
        const __half2 wv = w2[i / 2];
        or_[i    ] = xr[i    ] * scale * __half2float(wv.x);
        or_[i + 1] = xr[i + 1] * scale * __half2float(wv.y);
    }
    /* Tail (odd cols) */
    if ((cols & 1) && threadIdx.x == 0)
        or_[cols - 1] = xr[cols - 1] * scale * __half2float(w[cols - 1]);
}
