//! CANN backend — Huawei Ascend NPU.
//!
//! CANN = Compute Architecture for Neural Networks. Huawei's proprietary
//! framework for Ascend AI processors, analogous to CUDA for NVIDIA.
//!
//! Relevant hardware:
//! - **Ascend 310P**: edge inference accelerator, 8 TOPS INT8, 256 KB L2 SRAM
//! - **Ascend 910B**: 256 TFLOPS FP16 training/inference server chip
//! - **Ascend 910C**: next-gen server NPU (2024+), cluster deployments
//! - Chinese domestic cloud deployments (Huawei Cloud ModelArts) use these
//!
//! # Build requirements
//!
//! - Ascend-cann-toolkit (operator development kit)
//! - `SPITE_CANN=1` CMake flag
//! - `acl.h`, `acl_op_compiler.h` from Ascend SDK
//!
//! # Supported quantization (upstream llama.cpp CANN backend)
//!
//! F16, F32, Q4_0, Q8_0 are supported as of llama.cpp CANN merge (2024).
//! Q4_K_M and larger block quantizations require custom Ascend ops.
//!
//! # Memory model
//!
//! Discrete NPU with its own HBM (Ascend 910B: 32 GB HBM2e).
//! Use `aclrtMalloc` / `aclrtFree`; `aclrtMemcpy` for host↔device copies.

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Cann))
}

pub fn free(_ptr: *mut u8) {
    // TODO: aclrtFree(ptr)
}

pub fn upload(_ptr: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // TODO: aclrtMemcpy(ptr, size, src.as_ptr(), src.len(), ACL_MEMCPY_HOST_TO_DEVICE)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Cann))
}

pub fn download(_ptr: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    // TODO: aclrtMemcpy(dst.as_mut_ptr(), dst.len(), ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Cann))
}
