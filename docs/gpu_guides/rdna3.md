# GPU Guide — RDNA3

Cards: RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT, RX 7600

## Capabilities

| Feature              | Value                                    |
|----------------------|------------------------------------------|
| Architecture         | RDNA3 (GFX11xx)                          |
| Matrix cores         | WMMA: FP16, BF16, INT8                   |
| WMMA shape           | m16n16k16 (FP16/BF16), m16n16k32 (INT8) |
| LDS (shared mem)/CU  | 64 KB (128 KB in some configs)           |
| L2 cache (7900 XTX)  | 6 MB                                     |
| L3 cache (Infinity)  | 96 MB (7900 XTX)                         |
| Wavefront size       | 64 threads (wave64) or 32 (wave32)       |
| Memory bandwidth     | 960 GB/s (7900 XTX), 576 GB/s (7800 XT) |

## HIP vs ROCm

Kernels for RDNA3 use HIP. HIP is nearly identical to CUDA — most CUDA
kernels compile with `hipcc` after these substitutions:

```
cuda*      → hip*
__CUDA_*   → __HIP_*
cublasH_t  → hipblasHandle_t
```

`__shfl_xor_sync` exists in HIP. `wmma::` namespace exists in HIP as
`rocwmma::` (include `<rocwmma/rocwmma.hpp>`).

**You do not need a full ROCm stack.** HIP kernels compile with just `hipcc`
from the ROCm runtime package. On most distros: `sudo apt install rocm-hip-sdk`.

## What this means for inference kernels

**L3 (Infinity Cache) is the key advantage.**  
96 MB of L3 on the 7900 XTX is larger than NVIDIA's 72 MB L2 on the 4090,
and it's shared across all CUs. At ctx_len=4096, llama3-8b K/V fits entirely
in L3. Subsequent tokens hit L3 instead of HBM — bandwidth effectively
doubles for the attention KV-read bottleneck.

**Wave64 vs Wave32:**  
RDNA3 defaults to wave64 (64-thread wavefront). WMMA intrinsics require
wave32 for the m16n16k16 shape. Use `__attribute__((reqd_work_group_size(...)))`
or the `__WaveFrontSize32__` attribute to force wave32 when using WMMA.

**WMMA shapes differ from CUDA MMA:**  
CUDA's m16n8k16 becomes m16n16k16 in rocWMMA. Tile sizes need adjustment —
NVIDIA kernels ported 1:1 will be suboptimal.

## Known-fast patterns

- **Tile to fit in L3**: 96 MB is large. Use larger tiles than you would on
  NVIDIA — less reloading from HBM.
- **Avoid LDS bank conflicts**: RDNA3 LDS has 32 banks × 4 bytes. Stride
  access patterns that work on CUDA may need padding here.
- **Use `s_waitcnt`-friendly patterns**: RDNA3's VGPR latency hiding works
  best when memory loads have time to retire before use. Issue loads early.
