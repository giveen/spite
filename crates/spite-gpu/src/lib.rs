//! GPU device abstraction.
//!
//! Backend-agnostic `DeviceBuffer` and allocation API. Each backend
//! (CUDA, HIP, Metal, Vulkan, CPU) implements the same interface so
//! spite-executor can swap them without touching model code.
//!
//! At runtime, exactly one backend is active per device. The backend
//! is selected by `detect_backend()` based on what was compiled in
//! and what hardware is present.

pub mod cuda;
pub mod hip;
pub mod metal;
pub mod vulkan;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("backend {0:?} not compiled in (rebuild with the right feature flag)")]
    BackendUnavailable(GpuBackend),
    #[error("allocation failed: {0} bytes")]
    AllocFailed(usize),
    #[error("copy failed: {0}")]
    CopyFailed(String),
    #[error("device error: {0}")]
    DeviceError(String),
    #[error("stream sync failed")]
    SyncFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Cuda,
    Hip,
    Metal,
    Vulkan,
    Cpu,
}

impl GpuBackend {
    /// Detect which backend to use for the primary GPU.
    /// Falls back to `Cpu` when no GPU backend is compiled in.
    pub fn detect() -> Self {
        // TODO: check for libcuda.so, libhip.so, Metal availability at runtime
        GpuBackend::Cpu
    }
}

/// A device-side memory allocation.
pub struct DeviceBuffer {
    pub backend: GpuBackend,
    pub size:    usize,
    ptr:         *mut u8, // backend-managed; null for Cpu-backed vec
    cpu_data:    Vec<u8>, // used only when backend == Cpu
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocate `size` bytes on `backend`.
    pub fn alloc(backend: GpuBackend, size: usize) -> Result<Self, GpuError> {
        match backend {
            GpuBackend::Cuda   => cuda::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Hip    => hip::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Metal  => metal::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Vulkan => vulkan::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Cpu    => Ok(Self {
                backend,
                size,
                ptr:      std::ptr::null_mut(),
                cpu_data: vec![0u8; size],
            }),
        }
    }

    fn from_ptr(backend: GpuBackend, size: usize, ptr: *mut u8) -> Self {
        Self { backend, size, ptr, cpu_data: vec![] }
    }

    /// Copy `src` (host) → this buffer (device).
    pub fn upload(&mut self, src: &[u8]) -> Result<(), GpuError> {
        assert!(src.len() <= self.size);
        match self.backend {
            GpuBackend::Cpu => { self.cpu_data[..src.len()].copy_from_slice(src); Ok(()) }
            GpuBackend::Cuda   => cuda::upload(self.ptr, src),
            GpuBackend::Hip    => hip::upload(self.ptr, src),
            GpuBackend::Metal  => metal::upload(self.ptr, src),
            GpuBackend::Vulkan => vulkan::upload(self.ptr, src),
        }
    }

    /// Copy this buffer (device) → `dst` (host).
    pub fn download(&self, dst: &mut [u8]) -> Result<(), GpuError> {
        assert!(dst.len() <= self.size);
        match self.backend {
            GpuBackend::Cpu => { dst.copy_from_slice(&self.cpu_data[..dst.len()]); Ok(()) }
            GpuBackend::Cuda   => cuda::download(self.ptr, dst),
            GpuBackend::Hip    => hip::download(self.ptr, dst),
            GpuBackend::Metal  => metal::download(self.ptr, dst),
            GpuBackend::Vulkan => vulkan::download(self.ptr, dst),
        }
    }

    /// Raw device pointer (null for Cpu backend — use `as_cpu_slice` instead).
    pub fn as_ptr(&self) -> *mut u8 { self.ptr }

    pub fn as_cpu_slice(&self) -> &[u8] { &self.cpu_data }
    pub fn as_cpu_slice_mut(&mut self) -> &mut [u8] { &mut self.cpu_data }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() { return; }
        match self.backend {
            GpuBackend::Cuda   => cuda::free(self.ptr),
            GpuBackend::Hip    => hip::free(self.ptr),
            GpuBackend::Metal  => metal::free(self.ptr),
            GpuBackend::Vulkan => vulkan::free(self.ptr),
            GpuBackend::Cpu    => {}
        }
    }
}
