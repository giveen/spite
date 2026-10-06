//! CUDA backend — wraps `libcudart` via runtime `dlopen` (libloading).
//!
//! No link-time CUDA dependency: CPU-only builds and hosts without the CUDA
//! toolkit still compile and run; every call then reports
//! `BackendUnavailable`. The runtime API shares the device's primary
//! context with kernel `.so`s, so pointers allocated here are valid inside
//! kernels launched by those libraries.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

use libloading::Library;

use crate::{GpuBackend, GpuError, P2pKind};

type CudaErr = c_int;
const H2D: c_int = 1;
const D2H: c_int = 2;

struct Api {
    _lib: Library,
    get_device_count: unsafe extern "C" fn(*mut c_int) -> CudaErr,
    set_device: unsafe extern "C" fn(c_int) -> CudaErr,
    get_device: unsafe extern "C" fn(*mut c_int) -> CudaErr,
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> CudaErr,
    free: unsafe extern "C" fn(*mut c_void) -> CudaErr,
    memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> CudaErr,
    memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> CudaErr,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CudaErr,
    stream_sync: unsafe extern "C" fn(*mut c_void) -> CudaErr,
    device_sync: unsafe extern "C" fn() -> CudaErr,
    error_string: unsafe extern "C" fn(CudaErr) -> *const c_char,
    get_device_properties: unsafe extern "C" fn(*mut c_void, c_int) -> CudaErr,
    device_can_access_peer: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CudaErr,
}

/// Library names tried in order. `$SPITE_CUDART` overrides.
fn candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("SPITE_CUDART") {
        v.push(p);
    }
    for n in ["libcudart.so", "libcudart.so.13", "libcudart.so.12"] {
        v.push(n.to_owned());
    }
    for root in [
        std::env::var("CUDA_PATH").ok(),
        Some("/usr/local/cuda".into()),
    ]
    .into_iter()
    .flatten()
    {
        v.push(format!("{root}/lib64/libcudart.so"));
    }
    v
}

fn load() -> Option<Api> {
    let lib = candidates()
        .into_iter()
        .find_map(|n| unsafe { Library::new(n).ok() })?;
    // SAFETY: symbol signatures match the CUDA runtime API (cuda_runtime_api.h).
    unsafe {
        macro_rules! sym {
            ($name:literal) => {
                *lib.get($name).ok()?
            };
        }
        let api = Api {
            get_device_count: sym!(b"cudaGetDeviceCount\0"),
            set_device: sym!(b"cudaSetDevice\0"),
            get_device: sym!(b"cudaGetDevice\0"),
            malloc: sym!(b"cudaMalloc\0"),
            free: sym!(b"cudaFree\0"),
            memcpy: sym!(b"cudaMemcpy\0"),
            memset: sym!(b"cudaMemset\0"),
            mem_get_info: sym!(b"cudaMemGetInfo\0"),
            stream_sync: sym!(b"cudaStreamSynchronize\0"),
            device_sync: sym!(b"cudaDeviceSynchronize\0"),
            error_string: sym!(b"cudaGetErrorString\0"),
            get_device_properties: sym!(b"cudaGetDeviceProperties\0"),
            device_can_access_peer: sym!(b"cudaDeviceCanAccessPeer\0"),
            _lib: lib,
        };
        let mut n = 0;
        if (api.get_device_count)(&mut n) != 0 || n == 0 {
            return None;
        }
        if (api.set_device)(0) != 0 {
            return None;
        }
        Some(api)
    }
}

fn api() -> Result<&'static Api, GpuError> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(load)
        .as_ref()
        .ok_or(GpuError::BackendUnavailable(GpuBackend::Cuda))
}

fn check(a: &Api, code: CudaErr, what: &str) -> Result<(), GpuError> {
    if code == 0 {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr((a.error_string)(code)) }.to_string_lossy();
    Err(GpuError::DeviceError(format!("{what}: {msg} ({code})")))
}

/// True when libcudart loads and at least one CUDA device is present.
pub fn is_available() -> bool {
    api().is_ok()
}

/// Number of visible CUDA devices (honors `CUDA_VISIBLE_DEVICES`).
///
/// # Errors
/// `BackendUnavailable` without libcudart or a device; `DeviceError` if the query fails.
pub fn device_count() -> Result<usize, GpuError> {
    let a = api()?;
    let mut n: c_int = 0;
    check(
        a,
        unsafe { (a.get_device_count)(&mut n) },
        "cudaGetDeviceCount",
    )?;
    Ok(n.max(0) as usize)
}

/// A buffer at least as large as `cudaDeviceProp` on any CUDA 11–13 runtime.
/// Only the leading `char name[256]` is ever read; `cudaDeviceProp` starts with
/// that field in every released layout, so the rest of the struct is opaque.
#[repr(C, align(16))]
struct DevicePropBuf([u8; 2048]);

/// Marketing name of `device` (e.g. `"Tesla P100-PCIE-16GB"`).
///
/// The name carries the form factor that selects a link topology: `...-SXM2-`
/// for an NVLink part, `...-PCIE-` for a PCIe add-in card.
///
/// # Errors
/// `BackendUnavailable` without libcudart or a device; `DeviceError` on a bad ordinal.
pub fn device_name(device: usize) -> Result<String, GpuError> {
    let a = api()?;
    let d = c_int::try_from(device)
        .map_err(|_| GpuError::DeviceError(format!("device ordinal {device} out of range")))?;
    let mut buf = DevicePropBuf([0u8; 2048]);
    check(
        a,
        unsafe { (a.get_device_properties)(buf.0.as_mut_ptr().cast(), d) },
        "cudaGetDeviceProperties",
    )?;
    // SAFETY: `name` is a NUL-terminated `char[256]` at offset 0; CUDA wrote it.
    let name = unsafe { CStr::from_ptr(buf.0.as_ptr().cast::<c_char>()) };
    Ok(name.to_string_lossy().into_owned())
}

/// True when `src` may directly address memory on `dst`
/// (`cudaDeviceCanAccessPeer`). `false` also covers a device pair with no
/// peer path at all.
pub fn can_access_peer(src: usize, dst: usize) -> Result<bool, GpuError> {
    let a = api()?;
    let to_c = |d: usize| {
        c_int::try_from(d)
            .map_err(|_| GpuError::DeviceError(format!("device ordinal {d} out of range")))
    };
    let (s, d) = (to_c(src)?, to_c(dst)?);
    let mut can: c_int = 0;
    check(
        a,
        unsafe { (a.device_can_access_peer)(&mut can, s, d) },
        "cudaDeviceCanAccessPeer",
    )?;
    Ok(can != 0)
}

/// Classify the `src -> dst` device link: [`P2pKind::Nvlink`] only when peer
/// access exists **and** both cards are an SXM/NVLink form factor, else
/// [`P2pKind::Pcie`] (peer access over PCIe) or [`P2pKind::None`].
///
/// This is the runtime "NVLink probe": there is no scalar CUDA attribute for
/// the link medium, so the device name's form factor is the tie-breaker. Names
/// that do not spell out SXM (e.g. `"NVIDIA H100 80GB HBM3"`) are treated as
/// PCIe; callers needing tensor parallelism on such a part must opt in
/// explicitly (`SPITE_P100_NVLINK=1`).
///
/// # Errors
/// Propagates backend/ordinal errors from the underlying queries.
pub fn p2p_kind(src: usize, dst: usize) -> Result<P2pKind, GpuError> {
    if src == dst {
        return Ok(P2pKind::None);
    }
    if !can_access_peer(src, dst)? {
        return Ok(P2pKind::None);
    }
    let nvlink =
        is_nvlink_form_factor(&device_name(src)?) && is_nvlink_form_factor(&device_name(dst)?);
    Ok(if nvlink {
        P2pKind::Nvlink
    } else {
        P2pKind::Pcie
    })
}

/// True when a device name identifies an SXM/NVLink form factor.
pub fn is_nvlink_form_factor(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper.contains("SXM") || upper.contains("NVLINK")
}

/// Ordinal of the calling thread's current CUDA device.
///
/// # Errors
/// `BackendUnavailable` without libcudart or a device; `DeviceError` if the query fails.
pub fn current_device() -> Result<usize, GpuError> {
    let a = api()?;
    let mut d: c_int = 0;
    check(a, unsafe { (a.get_device)(&mut d) }, "cudaGetDevice")?;
    Ok(d.max(0) as usize)
}

/// Make `device` current for the calling thread.
///
/// The current device is per thread: allocations, copies and kernel launches
/// on the legacy default stream all target it.
///
/// # Errors
/// `BackendUnavailable` without libcudart; `DeviceError` for an invalid ordinal.
pub fn set_device(device: usize) -> Result<(), GpuError> {
    let a = api()?;
    let d = c_int::try_from(device)
        .map_err(|_| GpuError::DeviceError(format!("device ordinal {device} out of range")))?;
    check(a, unsafe { (a.set_device)(d) }, "cudaSetDevice")
}

/// Run `f` with `device` current, then restore the previous device.
///
/// Skips both switches when `device` is already current, which is the
/// single-GPU case.
///
/// # Errors
/// Any error from switching devices, or the error `f` returns.
pub fn with_device<T>(
    device: usize,
    f: impl FnOnce() -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    let prev = current_device()?;
    if prev == device {
        return f();
    }
    set_device(device)?;
    let out = f();
    let restored = set_device(prev);
    let out = out?;
    restored?;
    Ok(out)
}

/// `(free, total)` VRAM in bytes on the current device.
pub fn mem_info() -> Result<(usize, usize), GpuError> {
    let a = api()?;
    let (mut free, mut total) = (0usize, 0usize);
    check(
        a,
        unsafe { (a.mem_get_info)(&mut free, &mut total) },
        "cudaMemGetInfo",
    )?;
    Ok((free, total))
}

pub fn alloc(size: usize) -> Result<*mut u8, GpuError> {
    let a = api()?;
    let mut p: *mut c_void = std::ptr::null_mut();
    if unsafe { (a.malloc)(&mut p, size.max(1)) } != 0 || p.is_null() {
        return Err(GpuError::AllocFailed(size));
    }
    Ok(p.cast())
}

pub fn free(ptr: *mut u8) {
    if let Ok(a) = api() {
        unsafe { (a.free)(ptr.cast()) };
    }
}

pub fn upload(dst: *mut u8, src: &[u8]) -> Result<(), GpuError> {
    let a = api()?;
    let rc = unsafe { (a.memcpy)(dst.cast(), src.as_ptr().cast(), src.len(), H2D) };
    check(a, rc, "cudaMemcpy H2D").map_err(|e| GpuError::CopyFailed(e.to_string()))
}

pub fn download(src: *mut u8, dst: &mut [u8]) -> Result<(), GpuError> {
    let a = api()?;
    let rc = unsafe { (a.memcpy)(dst.as_mut_ptr().cast(), src.cast(), dst.len(), D2H) };
    check(a, rc, "cudaMemcpy D2H").map_err(|e| GpuError::CopyFailed(e.to_string()))
}

/// Fill `len` bytes at `dst` with `byte`.
pub fn memset(dst: *mut u8, byte: u8, len: usize) -> Result<(), GpuError> {
    let a = api()?;
    check(
        a,
        unsafe { (a.memset)(dst.cast(), byte as c_int, len) },
        "cudaMemset",
    )
}

/// Synchronize a CUDA stream, or the whole device if `stream` is null.
///
/// # Safety
/// If `stream` is non-null, it must be a valid `cudaStream_t`.
pub unsafe fn sync_stream(stream: *mut c_void) -> Result<(), GpuError> {
    let a = api()?;
    let rc = if stream.is_null() {
        unsafe { (a.device_sync)() }
    } else {
        unsafe { (a.stream_sync)(stream) }
    };
    check(a, rc, "cuda sync").map_err(|_| GpuError::SyncFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs only where a CUDA device exists; otherwise checks the error path.
    #[test]
    fn roundtrip_or_unavailable() {
        if !is_available() {
            assert!(matches!(
                alloc(16),
                Err(GpuError::BackendUnavailable(GpuBackend::Cuda))
            ));
            return;
        }
        let src: Vec<u8> = (0..=255u8).collect();
        let p = alloc(src.len()).unwrap();
        upload(p, &src).unwrap();
        let mut back = vec![0u8; src.len()];
        download(p, &mut back).unwrap();
        assert_eq!(src, back);
        memset(p, 0, src.len()).unwrap();
        download(p, &mut back).unwrap();
        assert!(back.iter().all(|&b| b == 0));
        free(p);
        let (free_b, total) = mem_info().unwrap();
        assert!(free_b <= total && total > 0);
    }

    /// Topology queries are hardware-dependent but must never panic, and a
    /// self-link is always `None`.
    #[test]
    fn topology_probe_is_safe() {
        if !is_available() {
            assert!(matches!(
                device_name(0),
                Err(GpuError::BackendUnavailable(_))
            ));
            assert_eq!(p2p_kind(0, 0).unwrap(), P2pKind::None);
            return;
        }
        let name = device_name(0).unwrap();
        assert!(!name.is_empty());
        assert_eq!(p2p_kind(0, 0).unwrap(), P2pKind::None);
        if device_count().unwrap_or(0) >= 2 {
            let _ = p2p_kind(0, 1).unwrap();
        }
    }

    #[test]
    fn nvlink_form_factor_is_name_based() {
        assert!(is_nvlink_form_factor("Tesla P100-SXM2-16GB"));
        assert!(!is_nvlink_form_factor("Tesla P100-PCIE-16GB"));
        assert!(!is_nvlink_form_factor("NVIDIA GeForce RTX 5090"));
    }
}
