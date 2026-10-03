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
pub mod hip_unified;
pub mod metal;
pub mod vulkan;
pub mod sycl;
pub mod opencl;
pub mod cann;
pub mod musa;
pub mod hexagon;

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
    // ── NVIDIA ──────────────────────────────────────────────────────────────
    Cuda,

    // ── AMD ─────────────────────────────────────────────────────────────────
    /// Discrete AMD GPU (dGPU) — separate VRAM, copies required.
    Hip,
    /// AMD APU with unified memory (Strix Halo, Phoenix, Hawk Point).
    /// CPU and iGPU share one LPDDR pool; upload/download are zero-cost.
    HipUnified,

    // ── Apple ────────────────────────────────────────────────────────────────
    /// Metal (Apple Silicon and AMD eGPU on macOS). Unified memory on M-series.
    Metal,

    // ── Cross-vendor ─────────────────────────────────────────────────────────
    /// Vulkan 1.3 compute (GLSL shaders, SPIR-V). Fallback for any Vulkan GPU.
    Vulkan,
    /// OpenCL 2.0+. Primary: Qualcomm Adreno (Android). Also AMD/Intel fallback.
    OpenCl,

    // ── Intel ────────────────────────────────────────────────────────────────
    /// SYCL / oneAPI DPC++. Intel Arc, Intel Data Center GPU Flex/Max, Intel iGPU.
    Sycl,

    // ── Qualcomm ─────────────────────────────────────────────────────────────
    /// Hexagon HTP/HTA (Snapdragon DSP/NPU). Distinct from Adreno (OpenCL).
    Hexagon,

    // ── Huawei ───────────────────────────────────────────────────────────────
    /// CANN — Ascend NPU (Ascend 310P, 910B, 910C).
    Cann,

    // ── Moore Threads ────────────────────────────────────────────────────────
    /// MUSA — Moore Threads MTT GPUs (CUDA-compatible API, domestic China).
    Musa,

    // ── CPU fallback ─────────────────────────────────────────────────────────
    Cpu,
}

impl GpuBackend {
    /// Detect which backend to use for the primary GPU.
    /// Falls back to `Cpu` when no GPU backend is compiled in.
    ///
    /// # TODO (detection order)
    ///
    /// 1. CUDA:    probe `libcuda.so` / `nvcuda.dll`
    /// 2. HIP:     probe `libhip.so`; if device is APU → HipUnified
    /// 3. Metal:   `cfg!(target_os = "macos")`
    /// 4. SYCL:    probe Level Zero ICD or query Intel GPU device
    /// 5. MUSA:    probe `libmusa.so` (Moore Threads)
    /// 6. CANN:    probe `libascendcl.so`
    /// 7. Hexagon: check `/dev/ion` or `libQnnHtp.so` presence (Android/Windows ARM)
    /// 8. OpenCL:  probe ICD loader; prefer for Adreno devices
    /// 9. Vulkan:  last GPU option before falling back to Cpu
    pub fn detect() -> Self {
        if hip_unified::is_unified() {
            return GpuBackend::HipUnified;
        }
        GpuBackend::Cpu
    }

    /// True when CPU and GPU share the same physical memory (no DMA copy needed).
    pub fn is_unified_memory(self) -> bool {
        matches!(self, GpuBackend::HipUnified | GpuBackend::Metal | GpuBackend::Hexagon)
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
            GpuBackend::Cuda        => cuda::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Hip         => hip::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::HipUnified  => hip_unified::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Metal       => metal::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Vulkan      => vulkan::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Sycl        => sycl::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::OpenCl      => opencl::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Cann        => cann::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Musa        => musa::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Hexagon     => hexagon::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Cpu         => Ok(Self {
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
    /// On unified-memory backends (HipUnified, Metal, Hexagon) this is a no-op.
    pub fn upload(&mut self, src: &[u8]) -> Result<(), GpuError> {
        assert!(src.len() <= self.size);
        match self.backend {
            GpuBackend::Cpu         => { self.cpu_data[..src.len()].copy_from_slice(src); Ok(()) }
            GpuBackend::Cuda        => cuda::upload(self.ptr, src),
            GpuBackend::Hip         => hip::upload(self.ptr, src),
            GpuBackend::HipUnified  => hip_unified::upload(self.ptr, src),
            GpuBackend::Metal       => metal::upload(self.ptr, src),
            GpuBackend::Vulkan      => vulkan::upload(self.ptr, src),
            GpuBackend::Sycl        => sycl::upload(self.ptr, src),
            GpuBackend::OpenCl      => opencl::upload(self.ptr, src),
            GpuBackend::Cann        => cann::upload(self.ptr, src),
            GpuBackend::Musa        => musa::upload(self.ptr, src),
            GpuBackend::Hexagon     => hexagon::upload(self.ptr, src),
        }
    }

    /// Copy this buffer (device) → `dst` (host).
    /// On unified-memory backends (HipUnified, Metal, Hexagon) this is a no-op.
    pub fn download(&self, dst: &mut [u8]) -> Result<(), GpuError> {
        assert!(dst.len() <= self.size);
        match self.backend {
            GpuBackend::Cpu         => { dst.copy_from_slice(&self.cpu_data[..dst.len()]); Ok(()) }
            GpuBackend::Cuda        => cuda::download(self.ptr, dst),
            GpuBackend::Hip         => hip::download(self.ptr, dst),
            GpuBackend::HipUnified  => hip_unified::download(self.ptr, dst),
            GpuBackend::Metal       => metal::download(self.ptr, dst),
            GpuBackend::Vulkan      => vulkan::download(self.ptr, dst),
            GpuBackend::Sycl        => sycl::download(self.ptr, dst),
            GpuBackend::OpenCl      => opencl::download(self.ptr, dst),
            GpuBackend::Cann        => cann::download(self.ptr, dst),
            GpuBackend::Musa        => musa::download(self.ptr, dst),
            GpuBackend::Hexagon     => hexagon::download(self.ptr, dst),
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
            GpuBackend::Cuda        => cuda::free(self.ptr),
            GpuBackend::Hip         => hip::free(self.ptr),
            GpuBackend::HipUnified  => hip_unified::free(self.ptr, self.size),
            GpuBackend::Metal       => metal::free(self.ptr),
            GpuBackend::Vulkan      => vulkan::free(self.ptr),
            GpuBackend::Sycl        => sycl::free(self.ptr),
            GpuBackend::OpenCl      => opencl::free(self.ptr),
            GpuBackend::Cann        => cann::free(self.ptr),
            GpuBackend::Musa        => musa::free(self.ptr),
            GpuBackend::Hexagon     => hexagon::free(self.ptr),
            GpuBackend::Cpu         => {}
        }
    }
}
