//! CUDA backend — wraps libcuda / libcudart via raw FFI.
//!
//! Compiled only when SPITE_CUDA=1 (set by CMake when nvcc is found).
//! All functions return BackendUnavailable when the feature is absent.

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: cuMemAlloc / cudaMalloc → return device pointer
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Cuda))
}

pub fn free(_ptr: *mut u8) {
    // TODO: cudaFree(ptr)
}

pub fn upload(_dst: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // TODO: cudaMemcpy(dst, src.ptr, src.len(), cudaMemcpyHostToDevice)
    Ok(())
}

pub fn download(_src: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    // TODO: cudaMemcpy(dst.ptr, src, dst.len(), cudaMemcpyDeviceToHost)
    Ok(())
}

pub fn sync_stream(_stream: *mut std::ffi::c_void) -> Result<(), GpuError> {
    // TODO: cudaStreamSynchronize(stream)
    Ok(())
}
