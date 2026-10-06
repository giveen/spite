/*
 * kernels/qwen/qwen3_5/nvidia/sm_120/rtx_5090/rtx5090_tuning.cuh
 *
 * NVIDIA GeForce RTX 5090 physical hardware constants and scheduling heuristics:
 *   • 192 Streaming Multiprocessors (SMs)
 *   • 96 MB L2 cache
 *   • 1.79 TB/s GDDR7 memory bus
 *   • Wave occupancy padding (pad192)
 *   • K-split crossover heuristic (pick_wpr)
 *   • Shape specialization for Qwen3.5-27B and Qwen3.6-35B-A3B
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rtx5090 {

constexpr int SMS = 192;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK = 8;

/* ── Wave Occupancy Padding ────────────────────────────────────────────── */
/* Rounds CTA count up to exact multiple of 192 to prevent tail-wave latency */
__host__ __device__ __forceinline__ int pad192(int n) {
  return ((n + SMS - 1) / SMS) * SMS;
}

/* ── K-Split Crossover Heuristic ──────────────────────────────────────── */
/*
 * Below 12,288 rows (e.g. QKV projections, MTP stem, MoE intermediate),
 * 1 warp per row leaves the 192 SMs under-saturated. Crossover to 8 warps
 * per row (K-split) to maximize DRAM memory-bus saturation and tensor core
 * occupancy. At or above 12,288 rows, 1 warp per row already saturates 192 SMs.
 */
__host__ __device__ __forceinline__ int pick_wpr(int rows, int cols) {
  if (rows < 12288) return 8;
  return 1;
}

/* ── Shape Specialization Geometries ───────────────────────────────────── */
constexpr int K_QWEN35B_DMODEL = 2048;
constexpr int K_QWEN35B_EXP_INTERMEDIATE = 768;
constexpr int K_QWEN35B_EXP_SHARED = 2048;
constexpr int K_QWEN27B_DMODEL = 5120;
constexpr int K_QWEN27B_DFFN = 17408;
constexpr int K_QWEN_GDN_CHANNELS = 8192;

__host__ __device__ __forceinline__ bool is_shape_specialized_cols(int cols) {
  return cols == K_QWEN35B_DMODEL || cols == K_QWEN27B_DMODEL ||
         cols == K_QWEN_GDN_CHANNELS || cols == K_QWEN27B_DFFN;
}

} // namespace rtx5090
