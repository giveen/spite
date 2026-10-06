//! GPU device abstraction.
//!
//! Backend-agnostic `DeviceBuffer` and allocation API. Each backend
//! (CUDA, HIP, Metal, Vulkan, CPU) implements the same interface so
//! spite-executor can swap them without touching model code.
//!
//! At runtime, exactly one backend is active per device. The backend
//! is selected by `detect_backend()` based on what was compiled in
//! and what hardware is present.

pub mod cann;
pub mod cuda;
pub mod hexagon;
pub mod hip;
pub mod hip_unified;
pub mod metal;
pub mod musa;
pub mod opencl;
pub mod sycl;
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
        if cuda::is_available() {
            return GpuBackend::Cuda;
        }
        if hip_unified::is_unified() {
            return GpuBackend::HipUnified;
        }
        GpuBackend::Cpu
    }

    /// True when CPU and GPU share the same physical memory (no DMA copy needed).
    pub fn is_unified_memory(self) -> bool {
        matches!(
            self,
            GpuBackend::HipUnified | GpuBackend::Metal | GpuBackend::Hexagon
        )
    }
}

/// How two CUDA devices can exchange data directly.
///
/// The CUDA runtime exposes peer *access* (`cudaDeviceCanAccessPeer`) but not
/// whether the link is NVLink or PCIe. [`cuda::p2p_kind`] therefore combines
/// peer access with the card's form factor from its device name (`...SXM...`
/// ⇒ NVLink); a name that does not identify an SXM/NVLink part is reported as
/// [`P2pKind::Pcie`], never NVLink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2pKind {
    /// No direct peer access — transfers stage through host memory.
    None,
    /// Peer copies run over PCIe (add-in cards, PCIe variants of SXM parts).
    Pcie,
    /// An SXM/NVLink form factor *and* peer access are present.
    Nvlink,
}

/// A device-side memory allocation.
pub struct DeviceBuffer {
    pub backend: GpuBackend,
    pub size: usize,
    /// Backend-managed device pointer; for `Cpu` a leaked boxed slice owned by this
    /// buffer (freed in `Drop`), so `as_ptr` is valid for kernels on every backend.
    ptr: *mut u8,
    /// CUDA ordinal the memory lives on; copies and the free switch to it.
    /// Always 0 on other backends.
    device: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocate `size` bytes on `backend` (CUDA: on the current device).
    pub fn alloc(backend: GpuBackend, size: usize) -> Result<Self, GpuError> {
        match backend {
            GpuBackend::Cuda => {
                let device = cuda::current_device()?;
                let ptr = cuda::alloc(size)?;
                // Built directly: struct-update syntax from `from_ptr` would drop
                // (and free) the temporary it copies the pointer out of.
                Ok(Self {
                    backend,
                    size,
                    ptr,
                    device,
                })
            }
            GpuBackend::Hip => hip::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::HipUnified => {
                hip_unified::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr))
            }
            GpuBackend::Metal => metal::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Vulkan => vulkan::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Sycl => sycl::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::OpenCl => opencl::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Cann => cann::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Musa => musa::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr)),
            GpuBackend::Hexagon => {
                hexagon::alloc(size).map(|ptr| Self::from_ptr(backend, size, ptr))
            }
            GpuBackend::Cpu => {
                let ptr = Box::into_raw(vec![0u8; size].into_boxed_slice()).cast::<u8>();
                Ok(Self::from_ptr(backend, size, ptr))
            }
        }
    }

    /// Allocate `size` bytes on CUDA device `device`, leaving the current device unchanged.
    ///
    /// Every other backend ignores `device` and behaves like [`Self::alloc`].
    ///
    /// # Errors
    /// Allocation failure, or an invalid CUDA ordinal.
    pub fn alloc_on(backend: GpuBackend, device: usize, size: usize) -> Result<Self, GpuError> {
        if backend != GpuBackend::Cuda {
            return Self::alloc(backend, size);
        }
        cuda::with_device(device, || Self::alloc(backend, size))
    }

    fn from_ptr(backend: GpuBackend, size: usize, ptr: *mut u8) -> Self {
        Self {
            backend,
            size,
            ptr,
            device: 0,
        }
    }

    /// CUDA ordinal holding this buffer; 0 on every other backend.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Copy `src` (host) → this buffer (device).
    /// On unified-memory backends (HipUnified, Metal, Hexagon) this is a no-op.
    pub fn upload(&mut self, src: &[u8]) -> Result<(), GpuError> {
        assert!(src.len() <= self.size);
        match self.backend {
            GpuBackend::Cpu => {
                self.as_cpu_slice_mut()[..src.len()].copy_from_slice(src);
                Ok(())
            }
            GpuBackend::Cuda => cuda::with_device(self.device, || cuda::upload(self.ptr, src)),
            GpuBackend::Hip => hip::upload(self.ptr, src),
            GpuBackend::HipUnified => hip_unified::upload(self.ptr, src),
            GpuBackend::Metal => metal::upload(self.ptr, src),
            GpuBackend::Vulkan => vulkan::upload(self.ptr, src),
            GpuBackend::Sycl => sycl::upload(self.ptr, src),
            GpuBackend::OpenCl => opencl::upload(self.ptr, src),
            GpuBackend::Cann => cann::upload(self.ptr, src),
            GpuBackend::Musa => musa::upload(self.ptr, src),
            GpuBackend::Hexagon => hexagon::upload(self.ptr, src),
        }
    }

    /// Copy this buffer (device) → `dst` (host).
    /// On unified-memory backends (HipUnified, Metal, Hexagon) this is a no-op.
    pub fn download(&self, dst: &mut [u8]) -> Result<(), GpuError> {
        assert!(dst.len() <= self.size);
        match self.backend {
            GpuBackend::Cpu => {
                dst.copy_from_slice(&self.as_cpu_slice()[..dst.len()]);
                Ok(())
            }
            GpuBackend::Cuda => cuda::with_device(self.device, || cuda::download(self.ptr, dst)),
            GpuBackend::Hip => hip::download(self.ptr, dst),
            GpuBackend::HipUnified => hip_unified::download(self.ptr, dst),
            GpuBackend::Metal => metal::download(self.ptr, dst),
            GpuBackend::Vulkan => vulkan::download(self.ptr, dst),
            GpuBackend::Sycl => sycl::download(self.ptr, dst),
            GpuBackend::OpenCl => opencl::download(self.ptr, dst),
            GpuBackend::Cann => cann::download(self.ptr, dst),
            GpuBackend::Musa => musa::download(self.ptr, dst),
            GpuBackend::Hexagon => hexagon::download(self.ptr, dst),
        }
    }

    /// Raw pointer to the buffer: a device pointer, or host memory for `Cpu`.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Host view of a `Cpu` buffer (empty for device backends).
    pub fn as_cpu_slice(&self) -> &[u8] {
        if self.backend != GpuBackend::Cpu {
            return &[];
        }
        // SAFETY: `ptr` is the leaked boxed slice of `size` bytes owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }
    pub fn as_cpu_slice_mut(&mut self) -> &mut [u8] {
        if self.backend != GpuBackend::Cpu {
            return &mut [];
        }
        // SAFETY: as above, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        match self.backend {
            GpuBackend::Cuda => {
                let ptr = self.ptr;
                // A failed switch still attempts the free on the current device.
                if cuda::with_device(self.device, || {
                    cuda::free(ptr);
                    Ok(())
                })
                .is_err()
                {
                    cuda::free(ptr);
                }
            }
            GpuBackend::Hip => hip::free(self.ptr),
            GpuBackend::HipUnified => unsafe { hip_unified::free(self.ptr, self.size) },
            GpuBackend::Metal => metal::free(self.ptr),
            GpuBackend::Vulkan => vulkan::free(self.ptr),
            GpuBackend::Sycl => sycl::free(self.ptr),
            GpuBackend::OpenCl => opencl::free(self.ptr),
            GpuBackend::Cann => cann::free(self.ptr),
            GpuBackend::Musa => musa::free(self.ptr),
            GpuBackend::Hexagon => hexagon::free(self.ptr),
            // SAFETY: reconstructs the boxed slice leaked in `alloc`.
            GpuBackend::Cpu => unsafe {
                drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    self.ptr, self.size,
                )));
            },
        }
    }
}
