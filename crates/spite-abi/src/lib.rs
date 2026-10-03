//! Stable C ABI contract between the Rust host and C++23 kernels.
//!
//! These types mirror `core/abi.h` exactly. Both must be updated together.
//! The ABI_VERSION constant is the enforcement mechanism — the dispatcher
//! rejects any kernel whose reported version doesn't match.
//!
//! Rule: never add a field to an existing #[repr(C)] struct. Add a new struct
//! and bump ABI_VERSION instead.

#![cfg_attr(not(feature = "std"), no_std)]

use core::ffi::{c_char, c_int, c_void};

pub const ABI_VERSION: u32 = 2;

// ── Tensor type tag ────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiteType {
    F32  = 0,
    F16  = 1,
    Bf16 = 2,
    Q8_0 = 8,
    Q4_0 = 10,
    Q4K  = 12,
    Q5K  = 13,
    Q6K  = 14,
}

// ── Tensor ─────────────────────────────────────────────────────────────────

/// A view into mmap'd weight data. No allocation, no copy.
/// `data` points directly into the GGUF buffer; valid for the model's lifetime.
#[repr(C)]
pub struct SpiteTensor {
    pub data: *mut c_void,
    /// Dimensions: ne[0]=cols, ne[1]=rows, ne[2..] for higher dims.
    pub ne:   [u32; 4],
    pub kind: SpiteType,
}

// SAFETY: tensor data comes from a read-only mmap. Callers must not write.
unsafe impl Send for SpiteTensor {}
unsafe impl Sync for SpiteTensor {}

impl SpiteTensor {
    pub const fn null() -> Self {
        Self {
            data: core::ptr::null_mut(),
            ne:   [0; 4],
            kind: SpiteType::F32,
        }
    }

    pub fn is_null(&self) -> bool {
        self.data.is_null()
    }
}

// ── Inference context ──────────────────────────────────────────────────────

#[repr(C)]
pub struct SpiteCtx {
    pub n_ctx:            c_int,
    pub n_batch:          c_int,
    pub n_threads:        c_int,
    /// CUDA stream / HIP stream / Metal command buffer. Null for CPU kernels.
    pub gpu_stream:       *mut c_void,
    pub scratchpad:       *mut c_void,
    pub scratchpad_bytes: usize,
}

// ── KV cache ───────────────────────────────────────────────────────────────

#[repr(C)]
pub struct SpiteKvCache {
    pub k:     SpiteTensor,
    pub v:     SpiteTensor,
    pub layer: c_int,
}

// ── Op function pointer types ──────────────────────────────────────────────

pub type RmsNormFn = unsafe extern "C" fn(
    out:    *mut SpiteTensor,
    x:      *const SpiteTensor,
    weight: *const SpiteTensor,
    eps:    f32,
    ctx:    *const SpiteCtx,
) -> c_int;

pub type AttentionFn = unsafe extern "C" fn(
    out:            *mut SpiteTensor,
    x:              *const SpiteTensor,
    wq:             *const SpiteTensor,
    wk:             *const SpiteTensor,
    wv:             *const SpiteTensor,
    wo:             *const SpiteTensor,
    kvcache:        *mut SpiteKvCache,
    pos:            c_int,
    rope_freq_base: f32,
    ctx:            *const SpiteCtx,
) -> c_int;

pub type FfnFn = unsafe extern "C" fn(
    out:    *mut SpiteTensor,
    x:      *const SpiteTensor,
    w_gate: *const SpiteTensor,
    w_up:   *const SpiteTensor,
    w_down: *const SpiteTensor,
    ctx:    *const SpiteCtx,
) -> c_int;

/// Optional: fuse rms_norm + attention + ffn for one layer.
pub type LayerFn = unsafe extern "C" fn(
    out:       *mut SpiteTensor,
    x:         *const SpiteTensor,
    layer_idx: c_int,
    kvcache:   *mut SpiteKvCache,
    pos:       c_int,
    ctx:       *const SpiteCtx,
) -> c_int;

/// Verify N draft tokens against main-model logits.
/// Returns accept mask; first rejection zeroes all subsequent positions.
/// Returning -1 falls back to the generic scalar implementation.
pub type SpecVerifyFn = unsafe extern "C" fn(
    accept_mask:  *mut bool,
    draft_logits: *const SpiteTensor,
    main_logits:  *const SpiteTensor,
    temperature:  f32,
    n_draft:      u32,
    ctx:          *const SpiteCtx,
) -> c_int;

// ── Model capability declaration ───────────────────────────────────────────

/// What a model supports — derived from GGUF metadata by the runtime.
/// Kernel authors do not fill this; it is populated by spite-loader.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SpiteModelCaps {
    /// Model can act as the speculative verifier.
    pub can_verify: bool,
    /// Model can act as the speculative draft.
    pub can_draft: bool,
    /// 0 = speculative decoding not supported for this model.
    pub max_draft_tokens: u32,
    /// Null-terminated array of compatible draft architecture name pointers.
    /// e.g. `["llama3-68m\0", "llama3-1b\0", null]`
    pub draft_archs: *const *const c_char,
}

unsafe impl Send for SpiteModelCaps {}
unsafe impl Sync for SpiteModelCaps {}

impl SpiteModelCaps {
    /// A model that cannot participate in speculative decoding at all.
    pub const fn unsupported() -> Self {
        Self {
            can_verify:       false,
            can_draft:        false,
            max_draft_tokens: 0,
            draft_archs:      core::ptr::null(),
        }
    }

    pub fn supports_speculative(&self) -> bool {
        self.max_draft_tokens > 0
    }
}

// ── Kernel descriptor ──────────────────────────────────────────────────────

/// Returned by `spite_kernel_info()` — the one symbol every kernel exports.
#[repr(C)]
pub struct SpiteKernelInfo {
    pub abi_version: u32,
    pub model_arch:  *const c_char,
    pub gpu_arch:    *const c_char,
    pub author:      *const c_char,
    /// Null-terminated list of SpiteType values this kernel handles.
    pub supported_quants: [u32; 8],
    /// None = not implemented; dispatcher uses fallback.
    pub rms_norm:           Option<RmsNormFn>,
    pub attention:          Option<AttentionFn>,
    pub ffn:                Option<FfnFn>,
    pub layer:              Option<LayerFn>,
    pub speculative_verify: Option<SpecVerifyFn>,
    /// Chunked prefill: process a prompt in fixed-size chunks rather than all
    /// at once, enabling interleaving with decode steps and bounding peak memory.
    /// Reuses `LayerFn` signature; the caller passes `chunk_idx` via `pos`.
    pub prefill:            Option<LayerFn>,
}

unsafe impl Send for SpiteKernelInfo {}
unsafe impl Sync for SpiteKernelInfo {}

/// The one symbol every kernel shared library must export.
pub type KernelInfoFn = unsafe extern "C" fn() -> *const SpiteKernelInfo;
pub const KERNEL_ENTRY_SYMBOL: &[u8] = b"spite_kernel_info\0";
