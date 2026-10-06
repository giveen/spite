/*
 * kernels/_engine/kv_quant/int8_g64_codec.cuh
 *
 * Signed INT8, per-token Group-64 KV-cache codec.
 * Ported from NInfer (ops/kv_cache/int8_g64_codec.cuh).
 *
 * HeadDim = 256, Group = 64, Groups = 4.
 * Scale: FP16-RNE(absmax / 127.0f).
 * Codes: signed 8-bit integers clamped to [-127, 127] scaled by reciprocal of represented scale.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace spite::engine {

inline constexpr int kKVCacheInt8HeadDim = 256;
inline constexpr int kKVCacheInt8Group   = 64;
inline constexpr int kKVCacheInt8Groups  = kKVCacheInt8HeadDim / kKVCacheInt8Group;

struct KVCacheInt8QuantParams {
    __half scale;
    float inverse_scale;
};

// Exact persistent group-scale boundary. The stored scale is FP16-RNE(absmax/127);
// codes use the reciprocal of that represented FP16 value.
__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int8_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        scale,
        represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t kv_cache_int8_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ void kv_cache_int8_dequant_f32x8(const std::int8_t* codes8, float scale, float* out8) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out8[i] = static_cast<float>(codes8[i]) * scale;
    }
}

} // namespace spite::engine
