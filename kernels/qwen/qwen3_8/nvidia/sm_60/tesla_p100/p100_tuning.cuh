/*
 * kernels/qwen/qwen3_8/nvidia/sm_60/tesla_p100/p100_tuning.cuh
 *
 * Tile constants tuned to the Tesla P100 SXM2 (GP100, sm_60) physical limits.
 *
 * P100 SXM2 hardware summary (GP100 die):
 *   SMs:              56
 *   CUDA cores/SM:    64   (3584 total)
 *   Registers/SM:     65536
 *   Shared mem/SM:    64 KB (configurable; default bank is 48 KB)
 *   L2 cache:         4 MB (SXM2)
 *   HBM2 bandwidth:   732 GB/s (SXM2) / 549 GB/s (PCIe)
 *   FP16 throughput:  21.2 TFLOPS (packed __half2)
 *   NVLink:           1.0 (4 links × 40 GB/s = 160 GB/s bidirectional, SXM2)
 *
 * Attention tile (half2 KV smem):
 *   KVFLASH_TILE=32, KVFLASH_WARPS=4  → 128 threads/block
 *   K+V smem: 2 × 32 × (128/2) × 4 = 16 384 bytes = 16 KB
 *   With __launch_bounds__(128, 4): 4 blocks/SM × 16 KB = 64 KB (fills GP100 smem)
 *   56 SMs × 4 blocks = 224 concurrent blocks for n_kv_heads×n_chunks grid.
 *
 *   KVFLASH_TILE is 32 (not 64) so blockDim.x fits in one warp; warp_sum()
 *   over the head_dim dot product is then complete with a single __shfl chain.
 *
 * FFN tile:
 *   d_ffn for Qwen3.8-27B ≈ 13824; gate+up scratch = 2×13824×4 = 110592 B ≈ 108 KB
 *   This lives in global scratchpad (HBM), not shared memory.
 *
 * Multi-GPU (NVLink TP):
 *   P100_TP_MAX_SHARDS=4  — 4-GPU NVLink ring (DGX-1 style)
 *   P100_ALLREDUCE_CHUNK=65536 — 64K floats per NVLink transfer chunk (~256 KB)
 */

#pragma once

/* Flash-decode KV tile dimensions for P100. */
#define P100_KVFLASH_TILE  32
#define P100_KVFLASH_WARPS 4

/*
 * Launch-bounds hint for the flash-decode tile kernel.
 * 128 threads/block (32×4), targeting 4 blocks/SM on GP100 (64 KB smem total).
 * The compiler pins register allocation so 4 blocks actually fit.
 */
#define P100_KVFLASH_LAUNCH_BOUNDS __launch_bounds__(P100_KVFLASH_WARPS * 32, 4)

/* FFN block width (output rows per block). */
#define P100_ROWS_PER_BLOCK 8

/* NVLink tensor-parallelism limits. */
#define P100_TP_MAX_SHARDS    4
#define P100_ALLREDUCE_CHUNK  65536   /* floats per chunk (~256 KB) */

/* Shared memory per block for the KV tile (half2 K+V at head_dim=128).
 * 2 × P100_KVFLASH_TILE × (head_dim/2) × sizeof(__half2) = 16 KB. */
#define P100_ATTN_SMEM_BYTES  16384   /* 16 KB */
