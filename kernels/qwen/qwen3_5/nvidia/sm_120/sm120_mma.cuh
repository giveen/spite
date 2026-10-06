/*
 * kernels/qwen/qwen3_5/nvidia/sm_120/sm120_mma.cuh
 *
 * Blackwell (sm_120) Tensor Core MMA intrinsics and matrix load instructions.
 * Ported from NInfer (ops/common/mma.cuh).
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace sm120 {

__device__ __forceinline__ unsigned smem_u32(const void *ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned &r0, unsigned &r1,
                                            unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned &r0, unsigned &r1,
                                            unsigned &r2, unsigned &r3,
                                            unsigned addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned &r0, unsigned &r1,
                                              unsigned addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned &r0, unsigned &r1,
                                              unsigned &r2, unsigned &r3,
                                              unsigned addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

/* 16x8x16 BF16 MMA: Accumulates D = A * B + C in FP32 */
__device__ __forceinline__ void mma_bf16(float &c0, float &c1, float &c2,
                                        float &c3, unsigned a0, unsigned a1,
                                        unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* 16x8x16 F16 MMA: Accumulates D = A * B + C in FP32 */
__device__ __forceinline__ void mma_f16(float &c0, float &c1, float &c2,
                                       float &c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0,
                                       unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/*
 * Blackwell Native NVFP4 W4A4 MMA:
 * mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
 *
 * M=16, N=8, K=64 with UE4M3 block scaling.
 */
__device__ __forceinline__ void
mma_nvfp4_e4m3(float &c0, float &c1, float &c2, float &c3, unsigned a0,
               unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1,
               unsigned sfa, unsigned sfb) {
  constexpr unsigned short kScaleBlockId = 0;
  constexpr unsigned short kScaleThreadId = 0;
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
               "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, "
               "{%4,%5,%6,%7}, "
               "{%8,%9}, "
               "{%0,%1,%2,%3}, "
               "{%10}, "
               "{%11,%12}, "
               "{%13}, "
               "{%14,%15};\n"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                 "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb),
                 "h"(kScaleBlockId), "h"(kScaleThreadId));
}

} // namespace sm120
