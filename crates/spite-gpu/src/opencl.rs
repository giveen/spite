//! OpenCL backend — Qualcomm Adreno (Android), older AMD, Intel iGPU.
//!
//! Primary target: **Qualcomm Adreno GPUs** in Snapdragon SoCs running Android.
//! These are the most common mobile GPU for on-device LLM inference.
//!
//! Relevant hardware:
//! - Snapdragon 8 Gen 2 — Adreno 740 (OpenCL 3.0)
//! - Snapdragon 8 Gen 3 — Adreno 750 (OpenCL 3.0)
//! - Snapdragon 8 Elite — Adreno 830 (OpenCL 3.0)
//! - Snapdragon X Elite laptops — Adreno 830 or equivalent
//!
//! Secondary targets:
//! - AMD GCN/RDNA on Windows (when ROCm not available)
//! - Intel iGPU on Linux without SYCL
//!
//! # Build requirements
//!
//! - `SPITE_OPENCL=1` CMake flag
//! - OpenCL ICD loader (libOpenCL.so / OpenCL.lib)
//! - Adreno: Qualcomm Adreno OpenCL SDK for optimised kernels
//!
//! # Unified memory on Adreno
//!
//! Adreno GPUs on Snapdragon share system LPDDR with the CPU (same as Strix Halo).
//! Use `CL_MEM_USE_HOST_PTR` for zero-copy buffers; upload is then a no-op
//! (the kernel accesses the host pointer directly after a cache flush).
//!
//! # Kernel format
//!
//! OpenCL kernels live under `kernels/<family>/<model>/adreno/` as `.cl` files.
//! Compiled at runtime via `clBuildProgram` (no offline SPIR-V required for Adreno).

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::OpenCl))
}

pub fn free(_ptr: *mut u8) {
    // TODO: clReleaseMemObject(cl_mem_handle)
}

pub fn upload(_ptr: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // TODO: clEnqueueWriteBuffer(queue, mem, CL_TRUE, 0, src.len(), src.as_ptr(), ...)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::OpenCl))
}

pub fn download(_ptr: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    // TODO: clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, dst.len(), dst.as_mut_ptr(), ...)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::OpenCl))
}
