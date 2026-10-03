//! SYCL backend — Intel Arc, Intel Data Center GPU Flex/Max, Intel iGPU.
//!
//! SYCL is Intel's cross-platform heterogeneous compute framework (ISO C++).
//! It targets:
//! - **Intel Arc discrete GPUs**: A-series (Alchemist), B-series (Battlemage),
//!   including Arc Pro lineup.
//! - **Intel Data Center GPU Flex/Max**: Flex 140/170 (client inference),
//!   Max 1100/1550 (HPC/AI datacenter).
//! - **Intel integrated GPUs**: Gen 12/13 (Tiger Lake, Alder Lake, Raptor Lake,
//!   Meteor Lake, Lunar Lake) via EU compute.
//! - Optionally NVIDIA/AMD GPUs via Intel's oneAPI LLVM backend (non-primary).
//!
//! # Build requirements
//!
//! - Intel oneAPI DPC++ compiler: `icpx` / `dpcpp`
//! - Intel oneAPI Base Toolkit
//! - `SPITE_SYCL=1` CMake flag
//! - Level Zero or OpenCL ICD installed (Level Zero preferred for Arc)
//!
//! # Memory model
//!
//! Arc discrete GPUs have their own VRAM; iGPUs share system RAM (similar to
//! HipUnified). Check `sycl::device::get_info<sycl::info::device::host_unified_memory>()`
//! at runtime to decide whether upload is a real copy.
//!
//! # Known limitations (stubs to fill)
//!
//! - Subgroup size must match the kernel's `reqd_sub_group_size` attribute.
//!   Arc Alchemist: subgroup 8 or 16; Arc Battlemage: subgroup 16 preferred.
//! - Half-precision (f16) kernel dispatch requires `sycl::half` support query.

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: sycl::malloc_device(size, queue) or sycl::malloc_shared for iGPU
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Sycl))
}

pub fn free(_ptr: *mut u8) {
    // TODO: sycl::free(ptr, queue)
}

pub fn upload(_ptr: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // TODO: queue.memcpy(ptr, src.as_ptr(), src.len()).wait()
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Sycl))
}

pub fn download(_ptr: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    // TODO: queue.memcpy(dst.as_mut_ptr(), ptr, dst.len()).wait()
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Sycl))
}
