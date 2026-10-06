/*
 * kernels/_engine/kv_quant/nvfp4_group16_codec.cuh
 *
 * D256 KV-cache NVFP4 codec with Group-16 quantization.
 * Ported from NInfer (ops/kv_cache/nvfp4_group16_codec.cuh).
 *
 * HeadDim = 256, Group = 16, Groups = 16.
 * Each group: 16 values packed into 8 bytes (E2M1 nibbles) + 1 byte E4M3 scale.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

namespace spite::engine {

inline constexpr int kKVCacheNvfp4HeadDim        = 256;
inline constexpr int kKVCacheNvfp4Group          = 16;
inline constexpr int kKVCacheNvfp4Groups         = 16;
inline constexpr int kKVCacheNvfp4CodeBytes      = 128; // 256 / 2
inline constexpr float kKVCacheNvfp4MaxFinite    = 6.0f;
inline constexpr float kKVCacheNvfp4ScaleMinimum = 0x1p-9f;
inline constexpr float kKVCacheNvfp4ScaleMaximum = 448.0f;

__device__ __forceinline__ float decode_nvfp4_e4m3_scalar(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

__device__ __forceinline__ void
pack_nvfp4_e2m1x16_regs(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
}

struct KVCacheNvfp4QuantizedGroup16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

__device__ __forceinline__ KVCacheNvfp4QuantizedGroup16
kv_cache_nvfp4_quantize_group16(const float* source) {
    float2 values[8];
    float max_abs = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = make_float2(source[2 * pair], source[2 * pair + 1]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    KVCacheNvfp4QuantizedGroup16 result{0, 0, 0};
    if (max_abs == 0.0f) return result;

    const float raw_scale = __fdiv_rn(max_abs, kKVCacheNvfp4MaxFinite);
    const float bounded   = fminf(kKVCacheNvfp4ScaleMaximum, fmaxf(kKVCacheNvfp4ScaleMinimum, raw_scale));
    result.scale          = __nv_cvt_float_to_fp8(bounded, __NV_SATFINITE, __NV_E4M3);
    const float rep_scale = decode_nvfp4_e4m3_scalar(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x, rep_scale);
        values[pair].y = __fdiv_rn(values[pair].y, rep_scale);
    }
    pack_nvfp4_e2m1x16_regs(values, result.codes_lo, result.codes_hi);
    return result;
}

__device__ __forceinline__ void
kv_cache_nvfp4_dequant_f16x8(const std::uint8_t* codes, std::uint8_t scale_code, __half* out8) {
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = static_cast<__half>(encoded_scale);
    const __half2 scale2 = __halves2half2(scale, scale);

#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = codes[pair];
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
        *reinterpret_cast<__half2*>(out8 + 2 * pair) = value;
    }
}

} // namespace spite::engine
