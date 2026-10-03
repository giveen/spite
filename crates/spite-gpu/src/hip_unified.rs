//! HIP unified-memory backend — AMD APUs (Strix Halo, Phoenix, Radeon 780M).
//!
//! On these devices the CPU and iGPU share one physical memory pool (LPDDR5X,
//! up to 128 GB on Strix Halo Max). `hipMallocManaged` / plain `malloc` both
//! land in that pool; there is no discrete VRAM to copy into, so upload and
//! download are zero-cost identity operations.
//!
//! # Build gate
//!
//! Compiled only when `SPITE_HIP=1` is set AND `SPITE_HIP_UNIFIED=1` (or the
//! device query returns `hipMemoryTypeUnified`). On a discrete dGPU the regular
//! `hip.rs` backend is used instead.
//!
//! # Allocation strategy
//!
//! Prefer `hipMallocManaged` for allocations that shaders also read so the
//! runtime can page-migrate transparently. For weight tensors that only the CPU
//! touches (e.g. mmap-backed GGUF), ordinary `malloc` is fine — the iGPU can
//! read host pointers directly on GFX11 APUs (Strix = GFX1150).
//!
//! # Relevant hardware
//!
//! | Product family  | iGPU          | Shared RAM | ROCm target |
//! |-----------------|---------------|------------|-------------|
//! | Strix Halo      | Radeon 890M   | up to 128 GB | gfx1150  |
//! | Strix Point     | Radeon 880M   | up to 96 GB  | gfx1150  |
//! | Phoenix         | Radeon 780M   | up to 64 GB  | gfx1103  |
//! | Hawk Point      | Radeon 890M   | up to 64 GB  | gfx1150  |

use crate::GpuError;

/// Allocate `size` bytes in the unified pool.
///
/// On real hardware: `hipMallocManaged(&ptr, size, hipMemAttachGlobal)`.
/// Here we use a heap Vec and return its stable pointer.
///
/// # TODO
///
/// Replace the Vec fallback with an actual `hipMallocManaged` FFI call when
/// compiled with `SPITE_HIP=1 SPITE_HIP_UNIFIED=1`:
/// ```c
/// hipError_t err = hipMallocManaged(&ptr, size, hipMemAttachGlobal);
/// if (err != hipSuccess) return Err(GpuError::AllocFailed(size));
/// ```
pub fn alloc(size: usize) -> Result<*mut u8, GpuError> {
    // stub: use heap allocation — same address space as iGPU on a real APU
    let mut buf: Vec<u8> = vec![0u8; size];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf); // freed in free()
    Ok(ptr)
}

/// Free a unified allocation.
///
/// On real hardware: `hipFree(ptr)`. The stub reconstructs the Vec to let
/// Rust drop it.
pub fn free(ptr: *mut u8, size: usize) {
    // stub: reconstruct and drop
    unsafe { drop(Vec::from_raw_parts(ptr, size, size)); }
}

/// Upload is a no-op on unified memory — CPU and GPU see the same bytes.
///
/// On discrete HIP the DMA engine copies host→VRAM; on an APU both sides
/// already share the same physical memory, so we only need a fence.
///
/// # TODO
///
/// Insert a `hipStreamSynchronize` / `__builtin_ia32_mfence` here if the
/// kernel compiler does not already guarantee coherence ordering.
#[inline]
pub fn upload(_ptr: *mut u8, _src: &[u8]) -> Result<(), GpuError> {
    // no-op: unified memory is always coherent between CPU and iGPU
    Ok(())
}

/// Download is a no-op on unified memory.
#[inline]
pub fn download(_ptr: *mut u8, dst: &mut [u8]) -> Result<(), GpuError> {
    // no-op: caller already has access to the same bytes via the raw pointer;
    // if they need a slice copy, they can cast ptr themselves
    let _ = dst;
    Ok(())
}

/// Read the unified buffer as a host slice without copying.
///
/// This is the real benefit of unified memory: model weights stay
/// mmap-backed on the host and the iGPU reads them directly.
///
/// # Safety
///
/// `ptr` must have been allocated by `alloc` and not yet freed.
/// Caller must ensure no concurrent GPU write is in flight.
pub unsafe fn as_slice<'a>(ptr: *const u8, size: usize) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(ptr, size) }
}

/// Detect whether this HIP device uses unified memory.
///
/// # TODO
///
/// Call `hipDeviceGetAttribute(&value, hipDeviceAttributeUnifiedAddressing, dev)`
/// and also check `hipDeviceAttributeIntegrated`. Return `true` only when
/// both are non-zero.
pub fn is_unified() -> bool {
    // stub: env var override for testing; real detection via HIP device query
    std::env::var("SPITE_HIP_UNIFIED").map(|v| v == "1").unwrap_or(false)
}
