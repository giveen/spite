//! One sentence describing what this crate does.
//!
//! Expand here: what problem it solves, which crate calls it, what the main
//! entry point is.  Keep it short — 3-5 lines max.
//!
//! # Implementation notes
//!
//! - Replace every occurrence of "Stub" / "stub" / "STUB" with your crate's
//!   actual name before writing any real code.
//! - Remove all `// TODO:` comments once their section is implemented.
//! - Run `cargo check -p spite-<yourname>` before opening a PR.

use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

/// All errors this crate can surface.
///
/// Add a variant for every distinct failure mode; use `#[error("…")]` strings
/// that are complete sentences and mention the concrete value where useful
/// (e.g. `"unsupported dtype: {0}"`).
#[derive(Debug, Error)]
pub enum StubError {
    #[error("not yet implemented")]
    NotImplemented,
    // TODO: add domain-specific variants here, for example:
    // #[error("shape mismatch: expected {expected}, got {actual}")]
    // ShapeMismatch { expected: String, actual: String },
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Runtime parameters for this crate.
///
/// Keep every field `pub` so callers can construct it with struct-literal
/// syntax.  Derive `Default` with sane values so callers only override what
/// they care about.
#[derive(Debug, Clone)]
pub struct StubConfig {
    // TODO: replace with real fields
    pub placeholder: u32,
}

impl Default for StubConfig {
    fn default() -> Self {
        Self {
            placeholder: 0,
        }
    }
}

// ── Primary type ──────────────────────────────────────────────────────────────

/// The main handle callers interact with.
///
/// Constructed once, reused many times.  Hold only the state that must persist
/// across calls; derive nothing you don't use.
pub struct Stub {
    _config: StubConfig,
    // TODO: add fields: buffers, handles, counters, caches …
}

impl Stub {
    /// Create a new instance.
    ///
    /// Returns `Err` if resources cannot be acquired (GPU OOM, bad config,
    /// missing file, etc.).
    ///
    /// # TODO
    ///
    /// 1. Validate `config` fields (ranges, combinations).
    /// 2. Allocate any buffers or handles required.
    /// 3. Return the initialised struct.
    pub fn new(_config: StubConfig) -> Result<Self, StubError> {
        // TODO: implement
        Err(StubError::NotImplemented)
    }

    /// The primary operation this crate exists to perform.
    ///
    /// Describe inputs and outputs clearly; include units (e.g. `[vocab_size,
    /// d_model]`, `[seq_len]`) and ownership semantics (borrowed vs owned).
    ///
    /// # TODO
    ///
    /// 1. Validate input shapes / preconditions.
    /// 2. Implement the algorithm.
    /// 3. Write results into `out` (prefer out-parameters over allocating).
    pub fn run(
        &mut self,
        _input: &[f32],
        _out:   &mut [f32],
    ) -> Result<(), StubError> {
        // TODO: implement
        Err(StubError::NotImplemented)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_compiles() {
        // Replace this smoke-test with real tests once the implementation exists.
        let cfg = StubConfig::default();
        assert_eq!(cfg.placeholder, 0);
    }

    // TODO: add tests for:
    // - happy path (correct output for known input)
    // - boundary conditions (empty slice, max values, zero)
    // - error paths (shape mismatch, OOM, bad config)
}
