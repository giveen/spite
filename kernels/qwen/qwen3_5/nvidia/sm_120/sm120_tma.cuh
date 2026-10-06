/*
 * kernels/qwen/qwen3_5/nvidia/sm_120/sm120_tma.cuh
 *
 * Blackwell (sm_120) TMA (Tensor Memory Accelerator) and asynchronous barrier primitives.
 * Ported from NInfer (ops/common/mbarrier.cuh, ops/linear/nvfp4/nvfp4_w4a4_tma.cuh).
 */

#pragma once

#include "sm120_mma.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace sm120 {

/* ── Blackwell mbarrier primitives ─────────────────────────────────────── */

__device__ __forceinline__ void cta_mbarrier_init(uint64_t *barrier,
                                                  uint32_t arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(smem_u32(barrier)), "r"(arrivals)
               : "memory");
}

__device__ __forceinline__ void cta_mbarrier_wait(uint64_t *barrier,
                                                  uint32_t phase) {
  constexpr uint32_t kSuspendTicks = 0x989680;
  asm volatile("{\n"
               ".reg .pred done;\n"
               "wait_loop_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
               "@done bra wait_done_%=;\n"
               "bra wait_loop_%=;\n"
               "wait_done_%=:\n"
               "}\n"
               :
               : "r"(smem_u32(barrier)), "r"(phase), "r"(kSuspendTicks)
               : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive(uint64_t *barrier) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :
               : "r"(smem_u32(barrier))
               : "memory");
}

__device__ __forceinline__ void
cta_mbarrier_arrive_expect_tx(uint64_t *barrier, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(smem_u32(barrier)), "r"(bytes)
               : "memory");
}

/* ── TMA Asynchronous 2D Bulk Load ─────────────────────────────────────── */

__device__ __forceinline__ void
tma_load_2d(void *destination, const CUtensorMap *descriptor,
            int32_t coordinate0, int32_t coordinate1, uint64_t *barrier) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(smem_u32(destination)), "l"(descriptor), "r"(coordinate0),
        "r"(coordinate1), "r"(smem_u32(barrier))
      : "memory");
}

/* Host helper to construct 2D CUtensorMap descriptors */
inline CUresult
make_tma_2d_descriptor(CUtensorMap *map, void *global_address,
                       CUtensorMapDataType data_type, uint64_t tensor_cols,
                       uint64_t tensor_rows, uint64_t row_stride_bytes,
                       uint32_t box_cols, uint32_t box_rows,
                       CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  const uint64_t global_dim[] = {tensor_cols, tensor_rows};
  const uint64_t global_stride[] = {row_stride_bytes};
  const uint32_t box_dim[] = {box_cols, box_rows};
  const uint32_t element_stride[] = {1, 1};

  return cuTensorMapEncodeTiled(
      map, data_type, 2, global_address, global_dim, global_stride, box_dim,
      element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

} // namespace sm120
