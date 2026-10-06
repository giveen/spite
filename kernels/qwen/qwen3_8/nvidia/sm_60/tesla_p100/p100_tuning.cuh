/*
 * kernels/qwen/qwen3_8/nvidia/sm_60/tesla_p100/p100_tuning.cuh
 *
 * Tile constants tuned to the Tesla P100 SXM2 (GP100, sm_60) physical limits.
 *
 * P100 SXM2 hardware summary (GP100 die):
 *   SMs:              56
 *   CUDA cores/SM:    64   (3584 total)
 *   Registers/SM:     65536
 *   Shared mem/SM:    64 KB (configurable; we request 64 KB for the attn tile)
 *   L2 cache:         4 MB (SXM2)
 *   HBM2 bandwidth:   732 GB/s (SXM2) / 549 GB/s (PCIe)
 *   FP16 throughput:  21.2 TFLOPS (packed __half2)
 *   NVLink:           1.0 (4 links × 40 GB/s = 160 GB/s bidirectional, SXM2)
 *
 * Attention tile:
 *   KVFLASH_TILE=64, KVFLASH_WARPS=4  → 256 threads/block
 *   K+V smem: 2 × 64 × 128 × 4 = 65536 bytes = 64 KB (exactly)
 *   At head_dim=128, 56 SMs × (256 threads / 256 threads/block) = 56 blocks
 *   with n_kv_heads=8 and 4 chunks: 32 blocks — fits comfortably.
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
#define P100_KVFLASH_TILE  64
#define P100_KVFLASH_WARPS 4

/* FFN block width (output rows per block). */
#define P100_ROWS_PER_BLOCK 8

/* NVLink tensor-parallelism limits. */
#define P100_TP_MAX_SHARDS    4
#define P100_ALLREDUCE_CHUNK  65536   /* floats per chunk (~256 KB) */

/* Shared memory per block requested at launch for the KV tile.
 * Must match 2 × P100_KVFLASH_TILE × head_dim × sizeof(float). */
#define P100_ATTN_SMEM_BYTES  65536   /* 64 KB */
