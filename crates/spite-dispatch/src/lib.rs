//! Runtime kernel selection.
//!
//! Given a model arch ("llama3") and the detected GPU arch ("sm_89"), walks
//! the kernels/ directory and builds a dispatch table by trying candidates in
//! priority order. Every slot falls back to the generic kernel — the table
//! always succeeds.
//!
//! Fallback chain per op:
//!   kernels/<model>/<exact_gpu>/
//!   kernels/<model>/generic_cuda/   (NVIDIA only)
//!   kernels/<model>/generic/
//!   kernels/generic/<exact_gpu>/
//!   kernels/generic/generic/        ← always present

use std::ffi::CStr;
use std::path::{Path, PathBuf};

use libloading::{Library, Symbol};
use thiserror::Error;

use spite_abi::{
    ABI_VERSION, AttentionFn, FfnFn, KernelInfoFn, LayerFn, RmsNormFn,
    SpecVerifyFn, SpiteKernelInfo, SpiteType, KERNEL_ENTRY_SYMBOL,
};

pub mod fallback;

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error("no generic fallback kernel found in {0}")]
    NoGenericFallback(PathBuf),
    #[error("kernel load failed: {0}")]
    Load(#[from] libloading::Error),
    #[error("ABI version mismatch: kernel={kernel} host={host}")]
    AbiMismatch { kernel: u32, host: u32 },
}

// ── Loaded kernel ──────────────────────────────────────────────────────────

struct LoadedKernel {
    // Library must stay alive; dropping it unloads the .so.
    _lib:  Library,
    info:  &'static SpiteKernelInfo,
}

impl LoadedKernel {
    fn open(path: &Path) -> Result<Self, DispatchError> {
        let lib: Library = unsafe { Library::new(path)? };
        let info_fn: Symbol<KernelInfoFn> = unsafe {
            lib.get(KERNEL_ENTRY_SYMBOL)?
        };
        let info: &'static SpiteKernelInfo = unsafe { &*info_fn() };
        if info.abi_version != ABI_VERSION {
            return Err(DispatchError::AbiMismatch {
                kernel: info.abi_version,
                host:   ABI_VERSION,
            });
        }
        Ok(Self { _lib: lib, info })
    }

    fn gpu_arch(&self) -> &str {
        unsafe { CStr::from_ptr(self.info.gpu_arch) }.to_str().unwrap_or("")
    }

    fn supports_quant(&self, q: SpiteType) -> bool {
        self.info.supported_quants.iter().any(|&v| v != 0 && v == q as u32)
    }
}

// ── Dispatch table ─────────────────────────────────────────────────────────

/// Source label for each resolved op — shown by `spite benchmark --verbose`.
#[derive(Debug, Clone)]
pub struct OpSource {
    pub gpu_arch: String,
    pub path:     PathBuf,
}

pub struct DispatchTable {
    pub rms_norm:          (Option<RmsNormFn>,   OpSource),
    pub attention:         (Option<AttentionFn>,  OpSource),
    pub ffn:               (Option<FfnFn>,        OpSource),
    pub layer:             (Option<LayerFn>,       OpSource),
    pub speculative_verify: (Option<SpecVerifyFn>, OpSource),
    // Keep libraries alive.
    _libs: Vec<LoadedKernel>,
}

impl DispatchTable {
    /// Print which kernel won each slot (for --verbose).
    pub fn print_sources(&self) {
        let rows = [
            ("rms_norm",   &self.rms_norm.1),
            ("attention",  &self.attention.1),
            ("ffn",        &self.ffn.1),
            ("layer",      &self.layer.1),
            ("spec_verify",&self.speculative_verify.1),
        ];
        for (op, src) in rows {
            println!("  {op:<14} → {}/{}", src.gpu_arch, src.path.display());
        }
    }
}

// ── Builder ────────────────────────────────────────────────────────────────

pub struct DispatchBuilder {
    kernels_dir: PathBuf,
    model_arch:  String,
    gpu_arch:    String,
}

impl DispatchBuilder {
    pub fn new(kernels_dir: impl AsRef<Path>, model_arch: &str, gpu_arch: &str) -> Self {
        Self {
            kernels_dir: kernels_dir.as_ref().to_owned(),
            model_arch:  model_arch.to_owned(),
            gpu_arch:    gpu_arch.to_owned(),
        }
    }

    pub fn build(self) -> Result<DispatchTable, DispatchError> {
        // Candidate directories in priority order.
        let vendor_generic = if self.gpu_arch.starts_with("sm_") {
            Some("generic_cuda")
        } else if self.gpu_arch.starts_with("rdna") || self.gpu_arch.starts_with("rx") {
            Some("generic_rocm")
        } else {
            None
        };

        let mut candidates: Vec<PathBuf> = Vec::new();
        candidates.push(self.kernels_dir.join(&self.model_arch).join(&self.gpu_arch));
        if let Some(vg) = vendor_generic {
            candidates.push(self.kernels_dir.join(&self.model_arch).join(vg));
        }
        candidates.push(self.kernels_dir.join(&self.model_arch).join("generic"));
        candidates.push(self.kernels_dir.join("generic").join(&self.gpu_arch));
        candidates.push(self.kernels_dir.join("generic").join("generic"));

        let mut libs: Vec<LoadedKernel> = Vec::new();
        for dir in &candidates {
            if let Some(k) = try_load_dir(dir) {
                libs.push(k);
            }
        }

        let generic_src = OpSource {
            gpu_arch: "generic".into(),
            path:     self.kernels_dir.join("generic").join("generic"),
        };

        let rms_norm           = find_op(&libs, |k| k.info.rms_norm,           &candidates, generic_src.clone());
        let attention          = find_op(&libs, |k| k.info.attention,          &candidates, generic_src.clone());
        let ffn                = find_op(&libs, |k| k.info.ffn,                &candidates, generic_src.clone());
        let layer              = find_op(&libs, |k| k.info.layer,              &candidates, generic_src.clone());
        let speculative_verify = find_op(&libs, |k| k.info.speculative_verify, &candidates, generic_src.clone());

        Ok(DispatchTable {
            rms_norm,
            attention,
            ffn,
            layer,
            speculative_verify,
            _libs: libs,
        })
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn try_load_dir(dir: &Path) -> Option<LoadedKernel> {
    if !dir.is_dir() { return None; }
    // Look for the first .so / .dylib / .dll in the directory.
    let exts = ["so", "dylib", "dll"];
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if exts.iter().any(|e| path.extension().map_or(false, |x| x == *e)) {
            return LoadedKernel::open(&path).ok();
        }
    }
    None
}

fn find_op<T: Copy>(
    libs:       &[LoadedKernel],
    getter:     impl Fn(&LoadedKernel) -> Option<T>,
    candidates: &[PathBuf],
    fallback:   OpSource,
) -> (Option<T>, OpSource) {
    for (lib, dir) in libs.iter().zip(candidates.iter()) {
        if let Some(f) = getter(lib) {
            return (Some(f), OpSource {
                gpu_arch: lib.gpu_arch().to_owned(),
                path:     dir.clone(),
            });
        }
    }
    (None, fallback)
}

// ── GPU detection ──────────────────────────────────────────────────────────

/// Returns the GPU arch string for the primary GPU, e.g. "sm_89", "rdna3".
/// Falls back to "generic" if detection fails.
pub fn detect_gpu_arch() -> String {
    // TODO: use CUDA / HIP / Metal APIs to query compute capability.
    // For now, try reading from environment (useful for testing).
    std::env::var("SPITE_GPU_ARCH").unwrap_or_else(|_| "generic".into())
}
