//! Metal backend — Apple Silicon (M1/M2/M3/M4) unified memory.
//!
//! Apple Silicon has unified memory, so H2D "copies" are zero-cost:
//! the MTLBuffer shares the same physical backing as the CPU allocation.
//! The upload/download stubs here model the logical interface even though
//! the actual implementation skips the copy.

use crate::GpuError;

pub fn alloc(_size: usize) -> Result<*mut u8, GpuError> {
    // TODO: [device newBufferWithLength:size options:MTLResourceStorageModeShared]
    //       return [buffer contents] (unified memory pointer)
    Err(GpuError::BackendUnavailable(crate::GpuBackend::Metal))
}

pub fn free(_ptr: *mut u8) {
    // TODO: [buffer release]  (or rely on ARC if using objc crate)
}

pub fn upload(_dst: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // TODO: unified memory → memcpy (or no-op if buffer wraps the same allocation)
    Ok(())
}

pub fn download(_src: *mut u8, _dst: &mut [u8]) -> Result<(), GpuError> {
    Ok(())
}

pub fn make_command_buffer() -> *mut std::ffi::c_void {
    // TODO: [commandQueue commandBuffer]
    std::ptr::null_mut()
}
