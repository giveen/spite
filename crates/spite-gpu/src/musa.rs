//! MUSA backend — Moore Threads MTT GPUs.
//!
//! Moore Threads is a Chinese domestic GPU vendor with a CUDA-compatible API.
//! MUSA (Moore Threads Unified System Architecture) mirrors the CUDA programming
//! model closely: `musa.h` mirrors `cuda.h`, `mublas` mirrors `cublas`, etc.
//!
//! Relevant hardware:
//! - **MTT S80**: consumer/prosumer GPU, 4096 shader processors, 16 GB GDDR6
//! - **MTT S3000**: datacenter card, target for LLM inference workloads
//! - **MTT S4000**: high-end inference (2025)
//!
//! # Build requirements
//!
//! - Moore Threads MUSA SDK (musart, mcc compiler)
//! - `SPITE_MUSA=1` CMake flag
//! - Kernel files: `.mu` extension (CUDA-like C++ dialect)
//!
//! # Porting from CUDA
//!
//! Most CUDA kernel code compiles with minor changes:
//!   `#include <cuda_runtime.h>` → `#include <musa_runtime.h>`
//!   `cudaMalloc`     → `musaMalloc`
//!   `cudaMemcpy`     → `musaMemcpy`
//!   `__global__`     → `__global__`  (same keyword)
//!   `threadIdx/blockIdx/gridDim` → same
//! Warp size is 32 (same as CUDA); shared memory per SM may differ.

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: musaMalloc(&ptr, size)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Musa))
}

pub fn free(_ptr: *mut u8) {
    // TODO: musaFree(ptr)
}

pub fn upload(_ptr: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // TODO: musaMemcpy(ptr, src.as_ptr(), src.len(), musaMemcpyHostToDevice)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Musa))
}

pub fn download(_ptr: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    // TODO: musaMemcpy(dst.as_mut_ptr(), ptr, dst.len(), musaMemcpyDeviceToHost)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Musa))
}
