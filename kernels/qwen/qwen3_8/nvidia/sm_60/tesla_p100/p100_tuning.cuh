/*
 * kernels/qwen/qwen3_8/nvidia/sm_60/tesla_p100/p100_tuning.cuh
 *
 * Constants specific to the Tesla P100 card (as opposed to GP100/sm_60 in
 * general, whose tile geometry lives in ../kvflash_pascal.cuh).
 *
 * P100 hardware summary (GP100 die):
 *   SMs:              56
 *   Shared mem/SM:    64 KB  (48 KB max per block)
 *   L2 cache:         4 MB
 *   HBM2 bandwidth:   732 GB/s (SXM2) / 732 or 549 GB/s (PCIe 16 GB / 12 GB)
 *   NVLink 1.0:       SXM2 only, 4 links x 20 GB/s per direction
 *                     (DGX-1: 8 GPUs in a hybrid cube-mesh; GPUs 0-3 and 4-7
 *                     each form a fully connected quad)
 *   PCIe variants have no NVLink: peer copies run over PCIe and tensor
 *   parallelism there is bandwidth-bound — prefer one GPU (Q4_0) or pipeline.
 */

#pragma once

/* NVLink tensor-parallelism limits (kept in sync with
 * crates/spite-parallel/src/p100_multi.rs::P100_TP_MAX_SHARDS). */
#define P100_TP_MAX_SHARDS 4
