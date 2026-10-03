//! Logit processors and token samplers.
//!
//! Pipeline: raw logits → [processors] → probabilities → sampler → token id
//!
//! Processors are applied in order and modify the logit vector in-place.
//! The sampler draws one token from the resulting distribution.
//!
//! Common pipeline for interactive use:
//!   Temperature → RepetitionPenalty → TopK → TopP → Greedy/Multinomial

pub mod mirostat;
pub mod dry;
pub mod xtc;

use thiserror::Error;

// ── Sampler trait ─────────────────────────────────────────────────────────

/// The pluggable sampling interface.
///
/// Implement this to replace the entire sampling pipeline for a specific
/// model, task, or user configuration — without touching anything else.
///
/// Register implementations in a `Registry<dyn Sampler>` keyed by
/// `PluginKey` so overrides apply only where they're needed.
///
/// The default implementation runs the standard pipeline:
/// Temperature → RepetitionPenalty → TopK → TopP → Multinomial/Greedy.
pub trait Sampler: Send + Sync {
    /// Apply all logit processors and draw one token id.
    ///
    /// `logits`:  raw pre-softmax logits, length = vocab_size, modified in-place.
    /// `context`: recently generated token ids (for repetition / frequency penalty).
    /// `cfg`:     sampler hyperparameters for this call.
    fn sample(
        &mut self,
        logits:  &mut [f32],
        context: &[u32],
        cfg:     &SamplerConfig,
    ) -> Result<u32, SamplingError>;
}

/// The standard sampler — runs the full pipeline defined by `SamplerConfig`.
/// Registered as the default in the engine's sampler registry.
pub struct DefaultSampler {
    pub rng: u64,
}

impl DefaultSampler {
    pub fn new(seed: u64) -> Self { Self { rng: seed.wrapping_add(1) } }
}

impl Sampler for DefaultSampler {
    fn sample(
        &mut self,
        logits:  &mut [f32],
        context: &[u32],
        cfg:     &SamplerConfig,
    ) -> Result<u32, SamplingError> {
        if logits.is_empty() { return Err(SamplingError::EmptyLogits); }
        let mut v: Vec<f32> = logits.to_vec();
        sample(&mut v, context, cfg, &mut self.rng)
    }
}

#[derive(Debug, Error)]
pub enum SamplingError {
    #[error("empty logit vector")]
    EmptyLogits,
    #[error("all logits are -inf after processing")]
    AllFiltered,
}

// ── Sampler configuration ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature:        f32,
    pub top_k:              usize,   // 0 = disabled
    pub top_p:              f32,     // 1.0 = disabled
    pub min_p:              f32,     // 0.0 = disabled; filters tokens < min_p * max_prob
    pub repetition_penalty: f32,     // 1.0 = disabled; > 1.0 penalises repeats
    pub frequency_penalty:  f32,     // 0.0 = disabled
    pub presence_penalty:   f32,     // 0.0 = disabled
    pub seed:               u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature:        0.7,
            top_k:              0,
            top_p:              0.95,
            min_p:              0.0,
            repetition_penalty: 1.0,
            frequency_penalty:  0.0,
            presence_penalty:   0.0,
            seed:               0,
        }
    }
}

// ── Logit processors ──────────────────────────────────────────────────────

/// Apply temperature: logits[i] /= temperature.
/// temperature = 0 → greedy (argmax), temperature = 1 → unmodified.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 {
        // TODO: greedy — handled downstream by argmax sampler
        return;
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}

/// Zero out all but the top-k logits (set others to -inf).
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() { return; }
    // TODO: partial sort to find kth-largest, then zero below threshold
}

/// Zero out tokens with probability < p of the cumulative distribution.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 { return; }
    // TODO: sort descending, compute cumulative softmax, zero tail
}

/// Zero out tokens with probability < min_p * max_probability.
/// Often better than top-p for maintaining diversity at low temperatures.
pub fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 { return; }
    // TODO: find max logit, compute threshold, zero below it
}

/// Penalise tokens that appear in `context` by dividing their logit by
/// `penalty` (if logit > 0) or multiplying (if logit < 0).
pub fn apply_repetition_penalty(logits: &mut [f32], context: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON { return; }
    for &token_id in context {
        if let Some(l) = logits.get_mut(token_id as usize) {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

// ── Samplers ──────────────────────────────────────────────────────────────

/// Return the token with the highest logit (temperature = 0 path).
pub fn greedy(logits: &[f32]) -> Result<u32, SamplingError> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .ok_or(SamplingError::EmptyLogits)
}

/// Sample from the distribution defined by `logits` using `rng_state`.
/// Logits are converted to probabilities via softmax before sampling.
pub fn multinomial(logits: &[f32], rng_state: &mut u64) -> Result<u32, SamplingError> {
    if logits.is_empty() { return Err(SamplingError::EmptyLogits); }
    // TODO: softmax → cumulative sum → binary search with LCG random draw
    let _ = rng_state;
    greedy(logits) // placeholder until implemented
}

// ── Full pipeline ─────────────────────────────────────────────────────────

/// Apply all processors from `cfg` then sample one token.
/// `context` is the token history used for repetition penalty.
pub fn sample(
    logits:  &mut Vec<f32>,
    context: &[u32],
    cfg:     &SamplerConfig,
    rng:     &mut u64,
) -> Result<u32, SamplingError> {
    apply_repetition_penalty(logits, context, cfg.repetition_penalty);
    apply_temperature(logits, cfg.temperature);
    apply_min_p(logits, cfg.min_p);
    apply_top_k(logits, cfg.top_k);
    apply_top_p(logits, cfg.top_p);

    if cfg.temperature <= 0.0 {
        greedy(logits)
    } else {
        multinomial(logits, rng)
    }
}

/// LCG fast RNG — shared by multinomial, mirostat, and DRY.
pub(crate) fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32)
}
