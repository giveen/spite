//! Draft model runner.
//!
//! Runs N forward passes on the draft model to produce candidate tokens
//! and their logit distributions. The main model then verifies them.
//!
//! The draft model has its own DispatchTable — it can have completely
//! different per-card kernels from the main model. A 68M parameter draft
//! running on the same GPU as a 7B main model is the typical setup.

use spite_abi::SpiteCtx;

/// Output of one draft run.
pub struct DraftOutput {
    /// Proposed token ids.
    pub tokens: Vec<u32>,
    /// Logit distributions — one Vec<f32> per draft token.
    pub logits: Vec<Vec<f32>>,
}

/// Run `n_tokens` draft steps starting from `context`.
///
/// Uses `draft_dispatch` for all ops. Returns the proposed tokens
/// and their logit distributions for the verify step.
pub fn run_draft(
    _context:  &[u32],
    _n_tokens: u32,
    _ctx:      &SpiteCtx,
) -> DraftOutput {
    // TODO:
    // for i in 0..n_tokens:
    //   1. embed(context + accepted so far)
    //   2. forward pass through draft model layers
    //   3. project to vocab → logits
    //   4. greedy sample (draft uses greedy; verify adjusts for main dist)
    //   5. append token, append logits
    DraftOutput { tokens: vec![], logits: vec![] }
}
