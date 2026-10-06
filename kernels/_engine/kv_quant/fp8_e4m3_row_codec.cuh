/*
 * kernels/_engine/kv_quant/fp8_e4m3_row_codec.cuh
 *
 * E4M3FN row-scaled D256 KV-cache codec.
 * Ported from NInfer (ops/kv_cache/fp8_e4m3_row_codec.cuh).
 *
 * HeadDim = 256, Group = 256 (row-scaled).
 * Scale: FP16-RNE(absmax / 448.0f), bounded by [2^-24, 65504.0].
 * Codes: __nv_fp8_e4m3 via __nv_cvt_float_to_fp8.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace spite::engine {

inline constexpr int kKVCacheFp8HeadDim        = 256;
inline constexpr int kKVCacheFp8Group          = 256;
inline constexpr int kKVCacheFp8Groups         = 1;
inline constexpr float kKVCacheFp8MaxFinite    = 448.0f;
inline constexpr float kKVCacheFp8ScaleMinimum = 0x1p-24f;
inline constexpr float kKVCacheFp8ScaleMaximum = 65504.0f;

struct KVCacheFp8QuantParams {
    __half scale;
    float inverse_scale;
};

__device__ __forceinline__ KVCacheFp8QuantParams kv_cache_fp8_quant_params(float absmax) {
    if (absmax == 0.0f) { return {__float2half_rn(0.0f), 0.0f}; }
    const float raw_scale = absmax / kKVCacheFp8MaxFinite;
    const float bounded   = fminf(kKVCacheFp8ScaleMaximum, fmaxf(kKVCacheFp8ScaleMinimum, raw_scale));
    const __half scale    = __float2half_rn(bounded);
    const float rep_scale = __half2float(scale);
    return {scale, 1.0f / rep_scale};
}

__device__ __forceinline__ std::uint8_t kv_cache_fp8_quant_code(float x, float inverse_scale) {
    if (inverse_scale == 0.0f) { return 0; }
    return __nv_cvt_float_to_fp8(x * inverse_scale, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ std::uint16_t kv_cache_fp8_quant_code2(float x0, float x1, float inverse_scale) {
    if (inverse_scale == 0.0f) { return 0; }
    return __nv_cvt_float2_to_fp8x2(make_float2(x0 * inverse_scale, x1 * inverse_scale),
                                    __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ __half2 kv_cache_fp8_code2_to_half2(std::uint16_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<__half2>(value);
}

__device__ __forceinline__ __half2 kv_cache_fp8_dequant_code2_to_half2(std::uint16_t storage, __half scale) {
    return __hmul2(kv_cache_fp8_code2_to_half2(storage), __halves2half2(scale, scale));
}

} // namespace spite::engine
