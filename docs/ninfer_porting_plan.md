# NInfer to Spite: Comprehensive Porting Plan & Kernel Landscape Survey

This document tracks the architectural port of high-performance CUDA/Blackwell kernels and execution strategies from `/mnt/storage/ninfer` into **Spite** (`/mnt/storage/Projects/spite`).

---

## 1. Executive Summary & Scope-of-Benefit Classification

Following Spite's **Scope-of-Benefit Decision Tree** (`AGENT.md` & `skill/SKILL.md`), every optimization, op, and kernel from NInfer is placed at the exact scope of its benefit rather than dumped into a single target directory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SCOPE-OF-BENEFIT DECISION TREE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Universal Algorithmic / CPU Fallback                                     │
│    └─► kernels/generic/generic/ & crates/spite-compute/                     │
│ 2. General CUDA Level (All NVIDIA GPUs: sm_75 .. sm_120)                    │
│    └─► kernels/qwen/qwen3_5/nvidia/ & kernels/_engine/<op>/nvidia/          │
│ 3. Architecture Level (Blackwell sm_120 ISA & MMA Intrinsics)               │
│    └─► kernels/qwen/qwen3_5/nvidia/sm_120/ & kernels/_engine/<op>/sm_120/   │
│ 4. Card Specialist Level (Physical RTX 5090 Constraints: 192 SMs, 96MB L2)  │
│    └─► kernels/qwen/qwen3_5/nvidia/sm_120/rtx_5090/                         │
│ 5. Host Engine / ABI / Models                                               │
│    └─► crates/spite-abi/, crates/spite-models/, crates/spite-dispatch/      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Feature Implementation Status Matrix

| NInfer Feature / Technique | Target Scope | Status | Details |
|---|:---:|:---:|---|
| **ABI Version Bump & Slots** | Host / ABI | **COMPLETE** | ABI v7: `linear_attn`, `attention_ex`, `mtp_stem` in `spite-abi` |
| **All-GGUF Quant Types (28 types)** | Host & Kernels | **COMPLETE** | `SpiteType` enum 1:1 with ggml; SIMT dequant GEMV in `core/gpu/` |
| **HybridDecoder Host Logic** | `spite-models` | **COMPLETE** | Interleaved GDN + Attention; golden-verified against llama.cpp |
| **RMSNorm + RoPE Fusion** | General CUDA | **COMPLETE** | In-register warp reduction + NEOX RoPE in `attn.cu` (`attn_prep`) |
| **Dynamic Causal Conv1d** | General CUDA | **COMPLETE** | Portable ring-buffer conv1d in `gdn.cu` (`gdn_conv_kernel`) |
| **GDN Recurrent Online Step** | General CUDA | **COMPLETE** | Portable state recurrence in `gdn.cu` (`gdn_recurrent_kernel`) |
| **SIMT Row-Split GEMV (28 quants)**| General CUDA | **COMPLETE** | 256-thread row reduction + multi-job fusion (`gemv.cu`) |
| **Fused MTP Stem** | CUDA / sm_120 | **COMPLETE** | Dual RMSNorm + projection packing across CUDA, sm_120, 5090 |
| **Bitonic Top-8 MoE Router** | General CUDA | **IN PROGRESS** | Porting NInfer `sparse_moe_route.cuh` for `qwen35moe` |
| **MoE Layer Execution (Qwen3.5 MoE)**| Host & Kernels | **IN PROGRESS** | 256 routed experts + 1 shared expert in `hybrid.rs` & CUDA |
| **K-Split MMA Contraction** | sm_120 | **PENDING** | Porting `q4_ksplit_mma.cuh` Tensor Core GEMV |
| **Blackwell NVFP4 W4A4 MMA** | sm_120 | **PENDING** | `mma.sync.aligned.kind::mxf4nvf4` PTX integration |
| **TMA Asynchronous Pipelines** | sm_120 | **PENDING** | `cuTensorMapEncodeTiled` + `cp.async.bulk` |
| **Grid SM Multiples Padding (`pad192`)**| RTX 5090 | **PENDING** | Wave occupancy padding in `rtx_5090/kernel.cu` |
| **K-Split Crossover (`pick_wpr`)** | RTX 5090 | **PENDING** | 8-warp split heuristic below 12,288 rows |
| **Shape Specialization (27B/35B)** | RTX 5090 | **PENDING** | Fixed geometry unrolls for Qwen3.5-27B & 35B-A3B |
| **Walsh-Hadamard D256 Transform** | `_engine` | **PENDING** | In-register 5-stage butterfly shuffle for KV quant |
| **KV Cache Block Codecs** | `_engine` | **PENDING** | INT8_G64, FP8_E4M3, NVFP4_G16 GPU codecs |
| **Speculative Target Verify** | `_engine` | **PENDING** | GPU implementation of `speculative_round.cuh` |

---

## 2. Current Architecture & Implementation Reality

### 2.1 Spite Host Status
- **ABI Contract**: `spite-abi` (ABI v7) provides:
  - `SpiteType`: Exact mapping to GGUF/ggml type IDs across 28 formats (including `Nvfp4 = 40`, `Mxfp4 = 39`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`).
  - `SpiteAttentionExFn`: Gated Q projection, per-head RMSNorm, partial NEOX RoPE, split-KV flash decoding.
  - `SpiteGdnFn`: Gated Delta Net recurrence and dynamic causal conv1d.
  - `SpiteMtpStemFn`: Fused token embedding + hidden norm stem for speculative decoding.
- **Model Execution**:
- **Model Execution**:
  - `crates/spite-models/src/hybrid.rs`: Implements `HybridDecoder` for dense hybrid models (e.g. Qwen3.5-27B) and MoE hybrid models (`qwen35moe`, e.g. Qwen3.6-35B-A3B / Kwaipilot KAT-Coder-V2.5). Golden-verified against llama.cpp logits (`hybrid_golden.rs`).
  - `crates/spite-models/src/qwen/qwen3_5_mtp.rs`: Multi-token prediction decoding integration.
- **MoE Support**:
  - `SpiteMoeParams`, `SpiteMoeFn`, and `moe_ffn` slot in `SpiteKernelInfo` and `DispatchTable`.
  - Supports 256 routed experts (top-8 selection) + 1 shared expert with sigmoid gating.

### 2.2 Kernel Landscape Status
- **Generic CUDA Baseline (`kernels/qwen/qwen3_5/nvidia/`)**:
  - Fully implemented and verified. Passes all tests in `tools/verify/verify.py` (56 matmul quant tests, GDN tests, `attention_ex` tests, `mtp_stem` tests, `moe_ffn` tests).
- **sm_120 Blackwell Architecture (`kernels/qwen/qwen3_5/nvidia/sm_120/`)**:
  - Features float4 vectorization for RMSNorm and MTP stem.
  - Implements `moe.cu` and `sparse_moe_route.cuh` for top-8 MoE dispatch.
  - `attn.cu`, `gdn.cu`, and `gemv.cu` are currently fallbacks to the generic CUDA baseline.
- **RTX 5090 Specialist (`kernels/qwen/qwen3_5/nvidia/sm_120/rtx_5090/`)**:
  - Configures 8-warp blocks (256 threads) for RMSNorm and MTP stem.
  - Integrated `moe.cu` and `sparse_moe_route.cuh`. Live-tested on RTX 5090 with `Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_S.gguf` running at 91.6 tok/s.
  - Awaiting `pad192` grid padding and `pick_wpr` K-split heuristics.

---

## 3. Phased Roadmap & Next Milestones

### Phase 1: ABI & Host Foundation [COMPLETE]
- [x] ABI v7 bump in `crates/spite-abi` and `core/abi.h`.
- [x] `SpiteType` mapping for 28 GGUF types.
- [x] `HybridDecoder` architecture in `spite-models`.
- [x] Golden logits verification vs llama.cpp.

### Phase 2: CUDA Baseline & MoE Support [COMPLETE]
- [x] Fused Q/K RMSNorm + NEOX RoPE + split-KV attention (`attention_ex`).
- [x] Causal conv1d and GDN recurrence step (`linear_attn`).
- [x] SIMT dequantizing GEMV for all 28 quants with multi-job fusion (`q35_gemv_multi`).
- [x] Fused MTP stem op (`mtp_stem`).
- [x] **Qwen3.5 MoE (`qwen35moe`)**:
  - [x] Port NInfer bitonic top-8 warp selection (`sparse_moe_route.cuh`).
  - [x] Implement MoE FFN kernel (256 routed experts + 1 shared expert with sigmoid gate).
  - [x] Extend `HybridDecoder` in `spite-models` to execute MoE blocks for `qwen35moe`.
  - [x] Wire `qwen35moe` GGUF loader tensor mappings in `spite-loader` and `spite-dispatch`.
  - [x] End-to-end live testing on RTX 5090 (`Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_S.gguf`).

### Phase 3: Blackwell sm_120 Acceleration [UPCOMING]
- [ ] Blackwell NVFP4 W4A4 MMA (`mma.sync.aligned.kind::mxf4nvf4.m16n8k64`).
- [ ] TMA Asynchronous Pipelines (`cuTensorMap` + `cp.async.bulk`).
- [ ] Tensor Core K-Split MMA for Q4_K, Q5_K, Q8_0 (`q4_ksplit_mma.cuh`).

### Phase 4: RTX 5090 Physical Tuning [UPCOMING]
- [ ] 192-SM wave occupancy padding (`pad192`).
- [ ] K-Split crossover heuristic (`pick_wpr`).
- [ ] Shape specialization for Qwen3.5-27B and 35B-A3B.

### Phase 5: Cross-Model Engine Subsystems [UPCOMING]
- [ ] In-register Walsh-Hadamard D256 butterfly transform.
- [ ] INT8_G64, FP8_E4M3, and NVFP4_G16 KV cache codecs.
- [ ] Speculative target verification kernel (`speculative_round.cuh`).
