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

pub const ABI_VERSION: u32 = 7;

// ── Tensor type tag ────────────────────────────────────────────────────────

/// Values are the GGUF / ggml tensor type ids (ABI v6), so a file's type id maps
/// straight onto this enum and block layouts match `core/ggml-common.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiteType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl SpiteType {
    /// Every defined type, ascending by id.
    pub const ALL: [SpiteType; 28] = [
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::Iq1M,
        Self::Bf16,
        Self::Tq1_0,
        Self::Tq2_0,
        Self::Mxfp4,
        Self::Nvfp4,
        Self::Q1_0,
        Self::Q2_0,
    ];

    /// The type for a GGUF/ggml tensor type id, if spite supports it.
    pub fn from_gguf_id(id: u32) -> Option<Self> {
        Self::ALL.iter().copied().find(|t| *t as u32 == id)
    }

    /// Bytes per atomic storage unit (bytes-per-element for full-precision,
    /// bytes-per-block for block-quantised types).
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq4Nl => 18,
            Self::Iq3S => 110,
            Self::Iq2S => 82,
            Self::Iq4Xs => 136,
            Self::Iq1M => 56,
            Self::Bf16 => 2,
            Self::Tq1_0 => 54,
            Self::Tq2_0 => 66,
            Self::Mxfp4 => 17,
            Self::Nvfp4 => 36,
            Self::Q1_0 => 18,
            Self::Q2_0 => 18,
        }
    }

    /// Elements per atomic storage block (1 for full-precision types).
    pub fn block_elements(self) -> u64 {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 => 32,
            Self::Q4_1 => 32,
            Self::Q5_0 => 32,
            Self::Q5_1 => 32,
            Self::Q8_0 => 32,
            Self::Q2K => 256,
            Self::Q3K => 256,
            Self::Q4K => 256,
            Self::Q5K => 256,
            Self::Q6K => 256,
            Self::Iq2Xxs => 256,
            Self::Iq2Xs => 256,
            Self::Iq3Xxs => 256,
            Self::Iq1S => 256,
            Self::Iq4Nl => 32,
            Self::Iq3S => 256,
            Self::Iq2S => 256,
            Self::Iq4Xs => 256,
            Self::Iq1M => 256,
            Self::Bf16 => 1,
            Self::Tq1_0 => 256,
            Self::Tq2_0 => 256,
            Self::Mxfp4 => 32,
            Self::Nvfp4 => 64,
            Self::Q1_0 => 128,
            Self::Q2_0 => 64,
        }
    }
}

// ── Tensor ─────────────────────────────────────────────────────────────────

/// A view into mmap'd weight data or an activation buffer. No allocation, no copy.
///
/// `nb` contains **byte** strides per dimension:
/// - `nb[0]` = `kind.block_bytes()` (bytes per block/element)
/// - `nb[1]` = `nb[0] * (ne[0] / kind.block_elements())` (bytes per row)
/// - `nb[2]` = `nb[1] * ne[1]` (bytes per matrix)
/// - `nb[3]` = `nb[2] * ne[2]` (bytes per batch item)
///
/// The executor guarantees all tensors passed to external kernel `.so` files are
/// contiguous. Use `is_contiguous()` to assert this at kernel entry during
/// development.
#[repr(C)]
pub struct SpiteTensor {
    pub data: *mut c_void,
    /// Dimensions: ne[0]=cols, ne[1]=rows, ne[2]=matrices, ne[3]=batch.
    pub ne: [u32; 4],
    /// Byte strides — see struct comment. nb[0]=0 means zero-sized tensor.
    pub nb: [u64; 4],
    pub kind: SpiteType,
}

// SAFETY: tensor data comes from a read-only mmap. Callers must not write.
unsafe impl Send for SpiteTensor {}
unsafe impl Sync for SpiteTensor {}

impl SpiteTensor {
    pub const fn null() -> Self {
        Self {
            data: core::ptr::null_mut(),
            ne: [0; 4],
            nb: [0; 4],
            kind: SpiteType::F32,
        }
    }

    pub fn is_null(&self) -> bool {
        self.data.is_null()
    }

    /// True iff strides are tightly packed (no padding, no transposition).
    pub fn is_contiguous(&self) -> bool {
        if self.nb[0] == 0 {
            return true;
        }
        let blk = self.kind.block_elements();
        let row = self.nb[0] * (self.ne[0] as u64 / blk);
        if self.ne[1] > 1 && self.nb[1] != row {
            return false;
        }
        if self.ne[2] > 1 && self.nb[2] != self.nb[1] * self.ne[1] as u64 {
            return false;
        }
        if self.ne[3] > 1 && self.nb[3] != self.nb[2] * self.ne[2] as u64 {
            return false;
        }
        true
    }

    /// Compute contiguous byte strides for the given type and shape.
    pub fn contiguous_strides(kind: SpiteType, ne: &[u32; 4]) -> [u64; 4] {
        let blk = kind.block_elements();
        let nb0 = kind.block_bytes();
        let nb1 = nb0 * (ne[0] as u64 / blk).max(1);
        let nb2 = nb1 * ne[1] as u64;
        let nb3 = nb2 * ne[2] as u64;
        [nb0, nb1, nb2, nb3]
    }
}

// ── Inference context ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SpiteCtx {
    pub n_ctx: c_int,
    pub n_batch: c_int,
    pub n_threads: c_int,
    /// Current token position in the sequence (0-based). Used by RoPE and KV cache.
    pub pos: c_int,
    /// Total query heads (n_heads in config).
    pub n_heads: c_int,
    /// KV heads — may be less than n_heads for GQA/MQA.
    pub n_kv_heads: c_int,
    /// CUDA stream / HIP stream / Metal command buffer. Null for CPU kernels.
    pub gpu_stream: *mut c_void,
    pub scratchpad: *mut c_void,
    pub scratchpad_bytes: usize,
}

// ── KV cache ───────────────────────────────────────────────────────────────

#[repr(C)]
pub struct SpiteKvCache {
    pub k: SpiteTensor,
    pub v: SpiteTensor,
    pub layer: c_int,
}

// ── Op function pointer types ──────────────────────────────────────────────

pub type RmsNormFn = unsafe extern "C" fn(
    out: *mut SpiteTensor,
    x: *const SpiteTensor,
    weight: *const SpiteTensor,
    eps: f32,
    ctx: *const SpiteCtx,
) -> c_int;

/// Attention for one token at `ctx.pos`.
///
/// Semantics (ABI v4): the result is **accumulated** into `out`
/// (`out += attn(x)`), so the residual add is fused into the op.
/// `q_norm` / `k_norm` are optional per-head RMSNorm weights `[head_dim]`
/// (Qwen3-style); pass null when the model has none. `norm_eps` applies to
/// them. `head_dim` is `wq.ne[1] / ctx.n_heads`.
pub type AttentionFn = unsafe extern "C" fn(
    out: *mut SpiteTensor,
    x: *const SpiteTensor,
    wq: *const SpiteTensor,
    wk: *const SpiteTensor,
    wv: *const SpiteTensor,
    wo: *const SpiteTensor,
    q_norm: *const SpiteTensor,
    k_norm: *const SpiteTensor,
    norm_eps: f32,
    kvcache: *mut SpiteKvCache,
    rope_freq_base: f32,
    ctx: *const SpiteCtx, // pos, n_heads, n_kv_heads are in ctx
) -> c_int;

/// Dense projection `out[r] = Σ_c w[r, c] · x[c]` (overwrites `out`).
/// Used for the LM head and any standalone projection.
pub type MatmulFn = unsafe extern "C" fn(
    out: *mut SpiteTensor,
    x: *const SpiteTensor,
    w: *const SpiteTensor,
    ctx: *const SpiteCtx,
) -> c_int;

/// FFN activation function selector.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SiluGate = 0, // SwiGLU — LLaMA, Mistral, Qwen
    GeluGate = 1, // GeGLU  — Gemma
    Gelu = 2,     // standard GELU — BERT-family, Phi
    Relu = 3,     // ReLU²  — GPT-NeoX variants
}

/// Gated FFN. Semantics (ABI v4): result is **accumulated** into `out`
/// (`out += ffn(x)`), fusing the residual add.
pub type FfnFn = unsafe extern "C" fn(
    out: *mut SpiteTensor,
    x: *const SpiteTensor,
    w_gate: *const SpiteTensor,
    w_up: *const SpiteTensor,
    w_down: *const SpiteTensor,
    activation: FfnActivation,
    ctx: *const SpiteCtx,
) -> c_int;

/// Multi-head Latent Attention (DeepSeek MLA).
///
/// KV is compressed through low-rank projections before caching.
/// The compressed latent is stored in the KV cache; up-projection
/// happens during the attention score computation.
pub type MlaFn = unsafe extern "C" fn(
    out: *mut SpiteTensor,
    x: *const SpiteTensor,
    w_dq: *const SpiteTensor,  // query down-projection (absorbs W_Q)
    w_uq: *const SpiteTensor,  // query up-projection
    w_dkv: *const SpiteTensor, // KV down-projection (shared compress)
    w_ukv: *const SpiteTensor, // KV up-projection
    wo: *const SpiteTensor,    // output projection
    kvcache: *mut SpiteKvCache,
    rope_freq_base: f32,
    ctx: *const SpiteCtx,
) -> c_int;

/// Optional: fuse rms_norm + attention + ffn for one layer.
pub type LayerFn = unsafe extern "C" fn(
    out: *mut SpiteTensor,
    x: *const SpiteTensor,
    layer_idx: c_int,
    kvcache: *mut SpiteKvCache,
    pos: c_int,
    ctx: *const SpiteCtx,
) -> c_int;

/// Optional: bitmask over [`SpiteType`] values (bit `ty as u32`) of the KV-cache
/// tiers this kernel's attention op can read and write.
///
/// `None` means the kernel predates VBR and is taken to accept F32 KV only -
/// the conservative reading. The host clamps the VBR start tier and degrade
/// ladder to this set so a kernel supporting fewer tiers still runs instead of
/// failing the attention op.
pub type KvCacheKindsFn = unsafe extern "C" fn() -> u64;

/// Verify N draft tokens against main-model logits.
/// Returns accept mask; first rejection zeroes all subsequent positions.
/// Returning -1 falls back to the generic scalar implementation.
pub type SpecVerifyFn = unsafe extern "C" fn(
    accept_mask: *mut bool,
    draft_logits: *const SpiteTensor,
    main_logits: *const SpiteTensor,
    temperature: f32,
    n_draft: u32,
    ctx: *const SpiteCtx,
) -> c_int;

// ── Extended attention (ABI v7) ────────────────────────────────────────────

/// Attention variant for hybrid Qwen3.5-style full-attention layers.
///
/// - `head_dim`: per-head width (not derivable from `wq` when Q is gated).
/// - `rope_dim`: only the first `rope_dim` dims of each q/k head are rotated
///   (NEOX pairing `i <-> i + rope_dim/2`); `rope_dim == head_dim` is plain RoPE.
/// - `gated_q`: 1 when `wq` has `2*n_heads*head_dim` rows laid out per head as
///   `[q | gate]`; each head's attention output is scaled by `sigmoid(gate)`
///   before `wo`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpiteAttnParams {
    pub head_dim: i32,
    pub rope_dim: i32,
    pub gated_q: i32,
}

impl SpiteAttnParams {
    /// Scratch floats the op needs in `ctx.scratchpad` (mirrors `spite_attn_ex_scratch_floats`).
    pub fn scratch_floats(&self, n_heads: usize, n_kv_heads: usize, n_ctx: usize) -> usize {
        let hd = self.head_dim as usize;
        3 * n_heads * hd + 2 * n_kv_heads * hd + n_heads * n_ctx
    }
}

/// `SpiteAttentionFn` plus [`SpiteAttnParams`]; same accumulate-into-`out` semantics.
pub type AttentionExFn = unsafe extern "C" fn(
    out: *mut SpiteTensor,
    x: *const SpiteTensor,
    wq: *const SpiteTensor,
    wk: *const SpiteTensor,
    wv: *const SpiteTensor,
    wo: *const SpiteTensor,
    q_norm: *const SpiteTensor,
    k_norm: *const SpiteTensor,
    norm_eps: f32,
    kvcache: *mut SpiteKvCache,
    rope_freq_base: f32,
    params: *const SpiteAttnParams,
    ctx: *const SpiteCtx,
) -> c_int;

// ── Linear attention: Gated Delta Net layer (ABI v7) ───────────────────────

/// Geometry of one GDN layer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpiteGdnParams {
    /// Key/query heads.
    pub n_kh: i32,
    /// Value heads (`n_vh % n_kh == 0`).
    pub n_vh: i32,
    /// Key, query and value head width `S`; the state is `S x S` per value head.
    pub head_dim: i32,
    /// Depthwise conv taps `K`; the history holds `K - 1` previous inputs.
    pub d_conv: i32,
    /// Epsilon of the q/k L2 normalisation and of the gated RMS norm.
    pub norm_eps: f32,
}

impl SpiteGdnParams {
    /// Conv channels `C = 2*n_kh*S + n_vh*S`.
    pub fn conv_channels(&self) -> usize {
        (2 * self.n_kh as usize + self.n_vh as usize) * self.head_dim as usize
    }

    /// Scratch floats needed in `ctx.scratchpad` (mirrors `spite_gdn_scratch_floats`).
    pub fn scratch_floats(&self) -> usize {
        let s = self.head_dim as usize;
        let v = self.n_vh as usize * s;
        2 * self.n_kh as usize * s + 3 * v + 2 * self.n_vh as usize
    }

    /// Floats of `conv_hist` storage: `(K - 1) * C`.
    pub fn conv_hist_floats(&self) -> usize {
        (self.d_conv as usize).saturating_sub(1) * self.conv_channels()
    }

    /// Floats of recurrent `state` storage: `n_vh * S * S`.
    pub fn state_floats(&self) -> usize {
        self.n_vh as usize * (self.head_dim as usize).pow(2)
    }
}

/// One decode token of a complete GDN layer: `out += W_out . gated_norm(core(x))`.
/// See `SpiteGdnFn` in `core/abi.h` for the full math and layouts. `x` is the
/// RMS-normalised layer input; `conv_hist`/`state` are updated in place.
pub type GdnFn = unsafe extern "C" fn(
    out: *mut SpiteTensor,
    x: *const SpiteTensor,
    w_qkv: *const SpiteTensor,
    w_gate: *const SpiteTensor,
    w_beta: *const SpiteTensor,
    w_alpha: *const SpiteTensor,
    w_out: *const SpiteTensor,
    conv_w: *const SpiteTensor,
    ssm_dt: *const SpiteTensor,
    ssm_a: *const SpiteTensor,
    ssm_norm: *const SpiteTensor,
    conv_hist: *mut SpiteTensor,
    state: *mut SpiteTensor,
    params: *const SpiteGdnParams,
    ctx: *const SpiteCtx,
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
    /// e.g. `["llama4-68m\0", "llama4-1b\0", null]`
    pub draft_archs: *const *const c_char,
}

unsafe impl Send for SpiteModelCaps {}
unsafe impl Sync for SpiteModelCaps {}

impl SpiteModelCaps {
    /// A model that cannot participate in speculative decoding at all.
    pub const fn unsupported() -> Self {
        Self {
            can_verify: false,
            can_draft: false,
            max_draft_tokens: 0,
            draft_archs: core::ptr::null(),
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
    pub model_arch: *const c_char,
    pub gpu_arch: *const c_char,
    pub author: *const c_char,
    /// Null-terminated list of SpiteType values this kernel handles.
    pub supported_quants: [u32; 8],
    /// None = not implemented; dispatcher uses fallback.
    pub rms_norm: Option<RmsNormFn>,
    pub attention: Option<AttentionFn>,
    pub mla: Option<MlaFn>,
    pub ffn: Option<FfnFn>,
    pub layer: Option<LayerFn>,
    pub speculative_verify: Option<SpecVerifyFn>,
    /// Chunked prefill: process a prompt in fixed-size chunks rather than all
    /// at once, enabling interleaving with decode steps and bounding peak memory.
    /// Reuses `LayerFn` signature; the caller passes `chunk_idx` via `pos` in ctx.
    pub prefill: Option<LayerFn>,
    /// Dense projection (LM head). Added in ABI v4.
    pub matmul: Option<MatmulFn>,
    /// KV-cache tiers the attention op accepts. `None` = F32 only.
    /// Added after ABI v4 as a trailing optional slot (no version bump).
    pub kv_cache_kinds: Option<KvCacheKindsFn>,
    /// Gated-delta-net layer (hybrid archs). ABI v7.
    pub linear_attn: Option<GdnFn>,
    /// Extended attention: partial RoPE + gated Q. ABI v7.
    pub attention_ex: Option<AttentionExFn>,
}

unsafe impl Send for SpiteKernelInfo {}
unsafe impl Sync for SpiteKernelInfo {}

/// The one symbol every kernel shared library must export.
pub type KernelInfoFn = unsafe extern "C" fn() -> *const SpiteKernelInfo;
pub const KERNEL_ENTRY_SYMBOL: &[u8] = b"spite_kernel_info\0";

// ── Parallelism strategy ───────────────────────────────────────────────────

/// How to distribute work across GPUs.
///
/// Defined here (in spite-abi) so both spite-executor and spite-parallel
/// can reference it without creating a dependency cycle.
///
/// The `Default` is `None` (single GPU). Use `spite_parallel::gpu_aware_default()`
/// when you want the GPU-count-aware strategy instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShardStrategy {
    /// Single GPU — no parallelism.
    #[default]
    None,
    /// Transformer layers split in sequence across GPUs.
    Pipeline { n_stages: usize },
    /// Weight matrices split column-wise within each layer (Megatron-style).
    Tensor { n_shards: usize },
    /// Tensor parallelism within a node, pipeline across nodes.
    Hybrid { n_shards: usize, n_stages: usize },
}
