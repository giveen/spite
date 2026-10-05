//! Runtime kernel selection.
//!
//! `DispatchBuilder` takes a `KernelSpec` (family + model + arch + card + quant)
//! and builds a `DispatchTable` by walking two priority chains:
//!
//! **Model ops** (rms_norm, attention, ffn, layer) use `resolve::model_candidates`:
//!   kernels/<family>/<model>/<company>/<arch>/<card>/<quant>/   ← card + quant specialist
//!   kernels/<family>/<model>/<company>/<arch>/<card>/           ← card specialist
//!   kernels/<family>/<model>/<company>/<arch>/<quant>/          ← quant specialist
//!   kernels/<family>/<model>/<company>/<arch>/                  ← arch baseline
//!   kernels/<family>/<model>/<company>/                         ← vendor generic baseline
//!   kernels/generic/generic_cuda/                               ← vendor generic
//!   kernels/generic/<company>/<arch>/
//!   kernels/generic/generic/                                    ← always present
//!
//! **Engine ops** (speculative, prefill, kv_quant) use `resolve::engine_candidates`:
//!   kernels/_engine/<feature>/<arch>/<card>/          ← card specialist
//!   kernels/_engine/<feature>/<arch>/                 ← arch baseline
//!   kernels/_engine/<feature>/generic/                ← always present

use std::ffi::CStr;
use std::path::{Path, PathBuf};

use libloading::{Library, Symbol};
use thiserror::Error;

use spite_abi::{
    ABI_VERSION, AttentionFn, FfnFn, KERNEL_ENTRY_SYMBOL, KernelInfoFn, LayerFn, MatmulFn,
    RmsNormFn, SpecVerifyFn, SpiteKernelInfo,
};

pub mod cards;
pub mod fallback;
pub mod hot_reload;
pub mod multi;
pub mod resolve;

pub use cards::card_spec;
pub use multi::{CommLink, GpuNode, MultiGpuSpec};
pub use resolve::{
    KernelSpec, arch_to_family_model, company_from_arch, detect_card_id, normalize_card_name,
};

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
    _lib: Library,
    info: &'static SpiteKernelInfo,
    /// Candidate directory this kernel was loaded from.
    dir: PathBuf,
}

impl LoadedKernel {
    fn open(path: &Path) -> Result<Self, DispatchError> {
        let lib: Library = unsafe { Library::new(path)? };
        let info_fn: Symbol<KernelInfoFn> = unsafe { lib.get(KERNEL_ENTRY_SYMBOL)? };
        let info: &'static SpiteKernelInfo = unsafe { &*info_fn() };
        if info.abi_version != ABI_VERSION {
            return Err(DispatchError::AbiMismatch {
                kernel: info.abi_version,
                host: ABI_VERSION,
            });
        }
        let dir = path.parent().map(Path::to_owned).unwrap_or_default();
        Ok(Self {
            _lib: lib,
            info,
            dir,
        })
    }

    fn gpu_arch(&self) -> &str {
        unsafe { CStr::from_ptr(self.info.gpu_arch) }
            .to_str()
            .unwrap_or("")
    }
}

// ── Dispatch table ─────────────────────────────────────────────────────────

/// Source label for each resolved op — shown by `spite dispatch`.
#[derive(Debug, Clone)]
pub struct OpSource {
    pub gpu_arch: String,
    pub path: PathBuf,
}

pub struct DispatchTable {
    // ── Model-specific ops ────────────────────────────────────────────────
    pub rms_norm: (Option<RmsNormFn>, OpSource),
    pub attention: (Option<AttentionFn>, OpSource),
    pub ffn: (Option<FfnFn>, OpSource),
    pub layer: (Option<LayerFn>, OpSource),
    pub matmul: (Option<MatmulFn>, OpSource),
    // ── Engine-level ops (cross-model, card/arch/generic chain) ──────────
    pub speculative_verify: (Option<SpecVerifyFn>, OpSource),
    pub prefill: (Option<LayerFn>, OpSource), // chunked prefill
    // Keep libraries alive.
    _libs: Vec<LoadedKernel>,
}

impl DispatchTable {
    /// Print which kernel won each slot (for --verbose).
    pub fn print_sources(&self) {
        let rows = [
            ("rms_norm", &self.rms_norm.1),
            ("attention", &self.attention.1),
            ("ffn", &self.ffn.1),
            ("layer", &self.layer.1),
            ("matmul", &self.matmul.1),
            ("spec_verify", &self.speculative_verify.1),
            ("prefill", &self.prefill.1),
        ];
        for (op, src) in rows {
            println!("  {op:<14} → {}/{}", src.gpu_arch, src.path.display());
        }
    }
}

// ── Builder ────────────────────────────────────────────────────────────────

pub struct DispatchBuilder {
    kernels_dir: PathBuf,
    spec: KernelSpec,
}

impl DispatchBuilder {
    /// Construct with a fully-populated `KernelSpec`.
    /// Use `resolve::detect_card_id` to fill `spec.card_id` from the GPU name.
    pub fn new(kernels_dir: impl AsRef<Path>, spec: KernelSpec) -> Self {
        Self {
            kernels_dir: kernels_dir.as_ref().to_owned(),
            spec,
        }
    }

    pub fn build(self) -> Result<DispatchTable, DispatchError> {
        let kdir = &self.kernels_dir;

        // ── Model-specific candidates ──────────────────────────────────────
        let model_cands = self.spec.model_candidates(kdir);
        let mut model_libs: Vec<LoadedKernel> = Vec::new();
        for dir in &model_cands {
            if let Some(k) = try_load_dir(dir) {
                model_libs.push(k);
            }
        }

        // ── Engine-level candidates (one set per feature) ──────────────────
        let spec_cands = self.spec.engine_candidates("speculative", kdir);
        let prefill_cands = self.spec.engine_candidates("prefill", kdir);

        let mut spec_libs: Vec<LoadedKernel> = Vec::new();
        let mut prefill_libs: Vec<LoadedKernel> = Vec::new();
        for dir in &spec_cands {
            if let Some(k) = try_load_dir(dir) {
                spec_libs.push(k);
            }
        }
        for dir in &prefill_cands {
            if let Some(k) = try_load_dir(dir) {
                prefill_libs.push(k);
            }
        }

        let generic_src = OpSource {
            gpu_arch: "generic".into(),
            path: kdir.join("generic").join("generic"),
        };

        // Model ops resolved from model candidate chain
        let rms_norm = find_op(&model_libs, |k| k.info.rms_norm, generic_src.clone());
        let attention = find_op(&model_libs, |k| k.info.attention, generic_src.clone());
        let ffn = find_op(&model_libs, |k| k.info.ffn, generic_src.clone());
        let layer = find_op(&model_libs, |k| k.info.layer, generic_src.clone());
        let matmul = find_op(&model_libs, |k| k.info.matmul, generic_src.clone());

        // Engine ops resolved from their own candidate chains
        let speculative_verify = find_op(
            &spec_libs,
            |k| k.info.speculative_verify,
            generic_src.clone(),
        );
        let prefill = find_op(&prefill_libs, |k| k.info.prefill, generic_src.clone());

        let mut all_libs = model_libs;
        all_libs.extend(spec_libs);
        all_libs.extend(prefill_libs);

        Ok(DispatchTable {
            rms_norm,
            attention,
            ffn,
            layer,
            matmul,
            speculative_verify,
            prefill,
            _libs: all_libs,
        })
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn try_load_dir(dir: &Path) -> Option<LoadedKernel> {
    if !dir.is_dir() {
        return None;
    }
    // Look for the first .so / .dylib / .dll in the directory.
    let exts = ["so", "dylib", "dll"];
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if exts
            .iter()
            .any(|e| path.extension().is_some_and(|x| x == *e))
        {
            return LoadedKernel::open(&path).ok();
        }
    }
    None
}

fn find_op<T: Copy>(
    libs: &[LoadedKernel],
    getter: impl Fn(&LoadedKernel) -> Option<T>,
    fallback: OpSource,
) -> (Option<T>, OpSource) {
    // `libs` holds only the candidates that actually loaded, in priority
    // order; each remembers its own dir (zipping with the candidate list
    // would misattribute paths whenever an earlier candidate is missing).
    for lib in libs {
        if let Some(f) = getter(lib) {
            return (
                Some(f),
                OpSource {
                    gpu_arch: lib.gpu_arch().to_owned(),
                    path: lib.dir.clone(),
                },
            );
        }
    }
    (None, fallback)
}

// ── GPU detection ──────────────────────────────────────────────────────────

/// Returns the GPU arch string for the primary GPU, e.g. "sm_89", "rdna3".
///
/// Resolution order:
///   1. `$SPITE_GPU_ARCH` if set (useful for testing / headless hosts).
///   2. `nvidia-smi` compute capability, mapped to `sm_<major><minor>`.
///   3. `"generic"` — the caller falls through to the generic kernel chain.
pub fn detect_gpu_arch() -> String {
    if let Ok(v) = std::env::var("SPITE_GPU_ARCH")
        && !v.is_empty()
    {
        return v;
    }
    if let Some(arch) = detect_nvidia_arch() {
        return arch;
    }
    "generic".into()
}

/// Query `nvidia-smi` for the primary GPU's compute capability and map it to
/// the kernel-tree arch string (`"8.9"` → `"sm_89"`, `"12.0"` → `"sm_120"`).
fn detect_nvidia_arch() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let cap = text.lines().next()?.trim();
    let mut parts = cap.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some(format!("sm_{major}{minor}"))
}
