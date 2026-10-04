//! Continuous batching scheduler.
//!
//! Queues multiple generation requests and packs them into one batched
//! forward pass per iteration. Tokens from different requests share the
//! same GPU forward pass — the scheduler assigns each request a KV cache
//! slot and tracks its state independently.
//!
//! Based on the Orca / vLLM iteration-level scheduling approach:
//!   - Each request gets a slot (dedicated KV cache region + activation row)
//!   - At each step, all `Generating` slots contribute exactly one token
//!   - New requests are inserted into `Free` slots without waiting for
//!     in-progress requests to finish
//!   - Finished requests release their slot immediately
//!
//! The `Executor` runs one batched forward pass per `step()` call.
//! The scheduler is the only component that touches `Executor` directly;
//! `spite-server` talks to the scheduler via `enqueue` / `step`.

use thiserror::Error;

// ── Schedule trait ────────────────────────────────────────────────────────

/// The pluggable scheduling interface.
///
/// Implement this to replace the batching and prioritization strategy
/// for a specific deployment (e.g. priority queues, fair-share, deadline-
/// aware scheduling) without touching the executor or server.
///
/// Register implementations in a `Registry<dyn Schedule>`.
pub trait Schedule: Send + Sync {
    /// Submit a new request. Returns the assigned request id.
    fn submit(&mut self, request: BatchRequest) -> Result<u64, SchedulerError>;

    /// Advance all active requests by one step (one token per decode slot).
    /// Returns outputs for all requests that produced a token this step.
    fn step(&mut self) -> Vec<BatchOutput>;

    /// Cancel an in-flight request. Returns `true` if it was found.
    fn cancel(&mut self, request_id: u64) -> bool;

    /// Number of active (non-free) slots.
    fn active_count(&self) -> usize;
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("no free slot — all {0} slots are occupied")]
    NoSlot(usize),
    #[error("unknown request id {0}")]
    UnknownRequest(u64),
    #[error("executor step failed: {0}")]
    ExecutorError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Free,
    Prefilling,  // processing the prompt
    Generating,  // autoregressive decode
    Done,
}

pub struct Slot {
    pub id:          usize,
    pub state:       SlotState,
    pub request_id:  Option<u64>,
    pub n_generated: usize, // tokens produced so far (not counting prompt)
    pub max_tokens:  usize,
}

pub struct BatchRequest {
    pub id:          u64,
    pub prompt:      Vec<u32>,
    pub max_tokens:  usize,
    pub temperature: f32,
    pub stop_tokens: Vec<u32>,
}

pub struct BatchOutput {
    pub request_id:  u64,
    pub token:       u32,
    pub done:        bool,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    NotDone,
    EosToken,
    StopToken,
    MaxTokens,
}

pub struct Scheduler {
    pub slots:      Vec<Slot>,
    next_req_id:    u64,
    pub executor:   spite_executor::Engine,
}

impl Scheduler {
    pub fn new(n_slots: usize, executor: spite_executor::Engine) -> Self {
        let slots = (0..n_slots).map(|i| Slot {
            id:          i,
            state:       SlotState::Free,
            request_id:  None,
            n_generated: 0,
            max_tokens:  0,
        }).collect();
        Self { slots, next_req_id: 1, executor }
    }

    /// Assign the next free request ID (monotonically increasing).
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_req_id;
        self.next_req_id += 1;
        id
    }

    /// Enqueue a new generation request. Returns its slot index.
    pub fn enqueue(&mut self, req: BatchRequest) -> Result<usize, SchedulerError> {
        let n = self.slots.len();
        let slot = self.slots.iter_mut()
            .find(|s| s.state == SlotState::Free)
            .ok_or(SchedulerError::NoSlot(n))?;
        slot.state      = SlotState::Prefilling;
        slot.request_id = Some(req.id);
        slot.n_generated = 0;
        slot.max_tokens  = req.max_tokens;
        Ok(slot.id)
    }

    /// Advance all active slots by one token.
    ///
    /// Internally: batch all `Generating` slots into one `Executor::decode_step`
    /// call, then distribute outputs back to slots.
    pub fn step(&mut self) -> Result<Vec<BatchOutput>, SchedulerError> {
        // TODO:
        // 1. Collect all Generating slot indices
        // 2. executor.decode_step(batch_tokens, &ctx) → batch_logits
        // 3. For each slot: sample from logits, check stop conditions
        // 4. Update slot state (Done if EOS/max reached)
        // 5. Return BatchOutput per slot
        Ok(vec![])
    }

    pub fn n_free(&self) -> usize {
        self.slots.iter().filter(|s| s.state == SlotState::Free).count()
    }

    pub fn n_active(&self) -> usize {
        self.slots.iter().filter(|s| s.state != SlotState::Free).count()
    }
}
