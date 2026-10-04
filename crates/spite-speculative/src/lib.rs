//! Speculative decoding.
//!
//! A small draft model proposes N tokens; the main model verifies them all
//! in a single batched forward pass. Accepted tokens are free — you get
//! them without an extra main-model forward pass.
//!
//! ## Per-card, per-model kernel support
//!
//! The verify step (compare draft vs main distributions, accept/reject) is
//! a kernel op in the ABI — `SpiteSpecVerifyFn`. Each GPU architecture can
//! ship an optimized implementation:
//!
//!   kernels/_engine/speculative/sm_89/speculative_verify.cu  ← fused softmax+compare
//!   kernels/_engine/speculative/rdna3/speculative_verify.hip
//!   kernels/_engine/speculative/generic/spec_verify.c        ← scalar fallback
//!
//! If no optimized kernel exists the dispatcher falls back to the generic
//! scalar implementation in `verify::scalar_verify`.
//!
//! ## Model capability check
//!
//! Before starting, `SpiteModelCaps` is checked:
//!   - `can_verify` must be true on the main model
//!   - `can_draft`  must be true on the draft model
//!   - `max_draft_tokens` must be > 0 on the main model
//!   - Draft model's arch must appear in main model's `draft_archs` list
//!
//! If any check fails, `SpectralError::NotSupported` is returned and the
//! caller falls back to regular autoregressive decoding.
//!
//! ## Strategies
//!
//! `Standard`   — one draft model, linear token sequence.
//! `Medusa`     — multiple prediction heads on the main model itself.
//!                No separate draft model needed; adds heads to the ABI.
//! `Eagle`      — draft model trained to match main model's feature space.
//!                More accurate draft; requires paired model weights.

pub mod draft;
pub mod tree;
pub mod verify;

use spite_abi::SpiteModelCaps;
use spite_dispatch::DispatchTable;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpeculativeError {
    #[error("main model does not support speculative decoding")]
    MainNotSupported,
    #[error("draft model cannot act as draft (can_draft=false)")]
    DraftNotSupported,
    #[error("draft arch '{draft}' not listed in main model's compatible drafts")]
    IncompatibleDraft { draft: String },
    #[error("n_draft {requested} exceeds main model max_draft_tokens {max}")]
    TooManyDraftTokens { requested: u32, max: u32 },
    #[error("verify kernel error: {0}")]
    VerifyKernel(i32),
}

// ── Strategy selector ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpecStrategy {
    /// Small draft model + main model verify. The standard approach.
    #[default]
    Standard,
    /// Multiple heads on the main model predict N tokens simultaneously.
    /// No draft model needed — draft_dispatch is ignored.
    Medusa,
    /// Draft model trained to match main model's internal representations.
    Eagle,
}

// ── Session ───────────────────────────────────────────────────────────────

/// A live speculative decoding session.
/// Holds both dispatch tables and the accept/reject state.
pub struct SpecSession {
    pub strategy: SpecStrategy,
    pub n_draft_tokens: u32,
    // Both dispatch tables stay alive for the session duration.
    pub main_dispatch: DispatchTable,
    pub draft_dispatch: Option<DispatchTable>, // None for Medusa
}

impl SpecSession {
    /// Validate that the two models can be paired for speculative decoding.
    ///
    /// Returns `Err(SpeculativeError::MainNotSupported)` if the main model
    /// has `max_draft_tokens == 0` — the caller should fall back to
    /// regular autoregressive decoding rather than hard-failing.
    pub fn new(
        main_caps: &SpiteModelCaps,
        main_dispatch: DispatchTable,
        draft_caps: Option<&SpiteModelCaps>,
        draft_dispatch: Option<DispatchTable>,
        draft_arch: Option<&str>,
        n_draft_tokens: u32,
        strategy: SpecStrategy,
    ) -> Result<Self, SpeculativeError> {
        if !main_caps.supports_speculative() {
            return Err(SpeculativeError::MainNotSupported);
        }
        if n_draft_tokens > main_caps.max_draft_tokens {
            return Err(SpeculativeError::TooManyDraftTokens {
                requested: n_draft_tokens,
                max: main_caps.max_draft_tokens,
            });
        }

        if strategy != SpecStrategy::Medusa {
            let dc = draft_caps.ok_or(SpeculativeError::DraftNotSupported)?;
            if !dc.can_draft {
                return Err(SpeculativeError::DraftNotSupported);
            }
            // Check compatibility
            if let Some(arch) = draft_arch
                && !main_caps_allows_draft(main_caps, arch)
            {
                return Err(SpeculativeError::IncompatibleDraft {
                    draft: arch.to_owned(),
                });
            }
        }

        Ok(Self {
            strategy,
            n_draft_tokens,
            main_dispatch,
            draft_dispatch,
        })
    }

    /// Run one speculative step: draft N tokens, verify, return accepted count.
    ///
    /// Returns the number of tokens accepted (0..=n_draft_tokens).
    /// The caller appends those tokens to the output and continues.
    pub fn step(
        &mut self,
        _context: &[u32],
        _output: &mut Vec<u32>,
    ) -> Result<usize, SpeculativeError> {
        // TODO:
        // 1. draft::run_draft  → draft_logits [n_draft, vocab]
        // 2. main  ::run_main  → main_logits  [n_draft, vocab]  (batched)
        // 3. verify::run_verify → accept_mask [n_draft]
        //    Uses main_dispatch.speculative_verify if available,
        //    else verify::scalar_verify as fallback.
        // 4. Append accepted tokens to output.
        // 5. If any rejection, resample the rejected position from adjusted dist.
        Ok(0)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn main_caps_allows_draft(caps: &SpiteModelCaps, draft_arch: &str) -> bool {
    if caps.draft_archs.is_null() {
        return false;
    }
    // Walk the null-terminated array of C strings
    unsafe {
        let mut ptr = caps.draft_archs;
        while !(*ptr).is_null() {
            let s = std::ffi::CStr::from_ptr(*ptr).to_str().unwrap_or("");
            if s == draft_arch {
                return true;
            }
            ptr = ptr.add(1);
        }
    }
    false
}
