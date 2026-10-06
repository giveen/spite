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

use crate::{GpuBackend, GpuError};

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
}
