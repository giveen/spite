//! HIP backend — AMD ROCm, targets RDNA2/3 and CDNA GPUs.
//!
//! HIP mirrors the CUDA API closely; most CUDA code ports with
//! a mechanical sed(1) substitution of cuda → hip prefixes.

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: hipMalloc
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Hip))
}

pub fn free(_ptr: *mut u8) {
    // TODO: hipFree(ptr)
}

pub fn upload(_dst: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // TODO: hipMemcpy(dst, src, len, hipMemcpyHostToDevice)
    Ok(())
}

pub fn download(_src: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    // TODO: hipMemcpy(dst, src, len, hipMemcpyDeviceToHost)
    Ok(())
}
