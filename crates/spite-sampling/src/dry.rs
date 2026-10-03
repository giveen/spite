//! DRY (Don't Repeat Yourself) repetition penalty.
//!
//! Unlike the standard repetition penalty which uniformly penalises all
//! previously seen tokens, DRY penalises continuations that would extend
//! a repeated n-gram. This avoids punishing common words while strongly
//! discouraging verbatim repeated passages.
//!
//! Algorithm:
//!   For each position j in the recent context (up to allowed_length back),
//!   find the longest suffix of context[..j] that matches the current
//!   generation suffix. If that match is ≥ base_length tokens long, apply
//!   an exponential penalty to the token at context[j]:
//!     penalty_logit -= multiplier ^ (match_length - base_length + 1)
//!
//! Reference: oobabooga/text-generation-webui implementation.

#[derive(Debug, Clone)]
pub struct DryConfig {
    /// Multiplier base for exponential penalty. Typical: 0.8.
    pub multiplier:     f32,
    /// Minimum match length before any penalty is applied. Typical: 2.
    pub base_length:    usize,
    /// How far back to search for matching suffixes. Typical: 512.
    pub allowed_length: usize,
    /// Token ids that break sequence matching (e.g. newline, EOS).
    pub breakers:       Vec<u32>,
}

impl Default for DryConfig {
    fn default() -> Self {
        Self {
            multiplier:     0.8,
            base_length:    2,
            allowed_length: 512,
            breakers:       vec![],
        }
    }
}

/// Apply DRY penalties to `logits` in-place.
///
/// `context`: all tokens generated so far (the full sequence seen by the model).
pub fn apply_dry(logits: &mut [f32], context: &[u32], cfg: &DryConfig) {
    if context.len() < cfg.base_length { return; }

    let search_start = context.len().saturating_sub(cfg.allowed_length);
    let tail = &context[search_start..];
    let end = tail.len();

    // For each position j, compute how long of a suffix match exists between
    // tail[..j] and the current generation context (the end of `tail`).
    for j in (1..end).rev() {
        if cfg.breakers.contains(&tail[j - 1]) { continue; }

        let mut match_len = 0usize;
        // Walk backwards from j-1 and end-1 comparing tokens
        while match_len < j && match_len < end - 1 {
            let a = tail[j - 1 - match_len];
            let b = tail[end - 1 - match_len];
            if a != b || cfg.breakers.contains(&a) { break; }
            match_len += 1;
        }

        if match_len >= cfg.base_length {
            let next_token = tail[j] as usize;
            if next_token < logits.len() {
                let exponent = (match_len - cfg.base_length + 1) as f32;
                logits[next_token] -= cfg.multiplier.powf(exponent);
            }
        }
    }
}
