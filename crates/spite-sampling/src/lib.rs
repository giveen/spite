//! Logit processors and token samplers.
//!
//! Pipeline: raw logits → [processors] → probabilities → sampler → token id
//!
//! Processors are applied in order and modify the logit vector in-place.
//! The sampler draws one token from the resulting distribution.
//!
//! Common pipeline for interactive use:
//!   Temperature → RepetitionPenalty → TopK → TopP → Greedy/Multinomial

pub mod dry;
pub mod mirostat;
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
        logits: &mut [f32],
        context: &[u32],
        cfg: &SamplerConfig,
    ) -> Result<u32, SamplingError>;
}

/// The standard sampler — runs the full pipeline defined by `SamplerConfig`.
/// Registered as the default in the engine's sampler registry.
pub struct DefaultSampler {
    pub rng: u64,
}

impl DefaultSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: seed.wrapping_add(1),
        }
    }
}

impl Sampler for DefaultSampler {
    fn sample(
        &mut self,
        logits: &mut [f32],
        context: &[u32],
        cfg: &SamplerConfig,
    ) -> Result<u32, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyLogits);
        }
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
    pub temperature: f32,
    pub top_k: usize,            // 0 = disabled
    pub top_p: f32,              // 1.0 = disabled
    pub min_p: f32,              // 0.0 = disabled; filters tokens < min_p * max_prob
    pub repetition_penalty: f32, // 1.0 = disabled; > 1.0 penalises repeats
    /// How many trailing context tokens the repetition penalty considers
    /// (`0` = the whole context). llama.cpp's `repeat_last_n` default is 64.
    pub repeat_last_n: usize,
    pub frequency_penalty: f32, // 0.0 = disabled
    pub presence_penalty: f32,  // 0.0 = disabled
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
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
    if k == 0 || k >= logits.len() {
        return;
    }
    // Threshold = kth-largest logit via linear select.
    let mut buf = logits.to_vec();
    buf.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let thresh = buf[k - 1];
    for l in logits.iter_mut() {
        if *l < thresh {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Zero out tokens with probability < p of the cumulative distribution.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 || logits.is_empty() {
        return;
    }
    // Sort indices by logit desc; keep the smallest prefix with mass >= p.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max = logits[idx[0]];
    let total: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let mut cumsum = 0f32;
    let mut kept = 0usize;
    for &i in &idx {
        cumsum += (logits[i] - max).exp();
        kept += 1;
        if cumsum / total >= p {
            break;
        }
    }
    // Always keep at least one token.
    for (n, &i) in idx.iter().enumerate() {
        if n >= kept.max(1) {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// Zero out tokens with probability < min_p * max_probability.
/// Often better than top-p for maintaining diversity at low temperatures.
pub fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 || logits.is_empty() {
        return;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let thresh = max + min_p.ln();
    for l in logits.iter_mut() {
        if *l < thresh {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Penalise tokens that appear in `context` by dividing their logit by
/// `penalty` (if logit > 0) or multiplying (if logit < 0).
///
/// Applies over the whole context and penalises each **distinct** token id
/// once. See [`apply_repetition_penalty_windowed`] for the windowed form.
pub fn apply_repetition_penalty(logits: &mut [f32], context: &[u32], penalty: f32) {
    apply_repetition_penalty_windowed(logits, context, penalty, 0);
}

/// Like [`apply_repetition_penalty`], but only over the last `repeat_last_n`
/// context tokens (`0` = the whole context), and each distinct token id is
/// penalised **once**.
///
/// This matches llama.cpp and HuggingFace. Penalising once per occurrence — the
/// previous behaviour — compounds as `penalty^k` for a token seen `k` times and
/// is far too aggressive (it can suppress a token that is merely common in the
/// prompt, e.g. a leading space).
pub fn apply_repetition_penalty_windowed(
    logits: &mut [f32],
    context: &[u32],
    penalty: f32,
    repeat_last_n: usize,
) {
    if (penalty - 1.0).abs() < f32::EPSILON || logits.is_empty() {
        return;
    }
    let window = if repeat_last_n == 0 || repeat_last_n >= context.len() {
        context
    } else {
        &context[context.len() - repeat_last_n..]
    };
    let mut seen = std::collections::HashSet::with_capacity(window.len());
    for &token_id in window {
        if !seen.insert(token_id) {
            continue;
        }
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
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .ok_or(SamplingError::EmptyLogits)
}

/// Sample from the distribution defined by `logits` using `rng_state`.
/// Logits are converted to probabilities via softmax before sampling.
pub fn multinomial(logits: &[f32], rng_state: &mut u64) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut cumsum = 0f32;
    let u = lcg_f32(rng_state);
    // Draw against unnormalized masses; rescale u by total.
    let total: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let target = u * total;
    for (i, &l) in logits.iter().enumerate() {
        cumsum += (l - max).exp();
        if cumsum >= target {
            return Ok(i as u32);
        }
    }
    // Rounding: fall back to argmax.
    greedy(logits)
}

// ── Full pipeline ─────────────────────────────────────────────────────────

/// Apply all processors from `cfg` then sample one token.
/// `context` is the token history used for repetition penalty.
pub fn sample(
    logits: &mut [f32],
    context: &[u32],
    cfg: &SamplerConfig,
    rng: &mut u64,
) -> Result<u32, SamplingError> {
    apply_repetition_penalty_windowed(logits, context, cfg.repetition_penalty, cfg.repeat_last_n);
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
/// Returns a uniform draw in [0, 1): `state >> 33` fills 31 bits.
pub(crate) fn lcg_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / 2147483648.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_keeps_k_best() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        assert!(logits[1].is_finite() && logits[4].is_finite());
        assert!(logits[0].is_infinite() && logits[2].is_infinite() && logits[3].is_infinite());
    }

    #[test]
    fn top_p_keeps_mass() {
        // Softmax ≈ [0.66, 0.24, 0.09, 0.01]; p=0.8 keeps first two.
        let mut logits = vec![2.0, 1.0, 0.0, -2.0];
        apply_top_p(&mut logits, 0.8);
        assert!(logits[0].is_finite() && logits[1].is_finite());
        assert!(logits[2].is_infinite() && logits[3].is_infinite());
    }

    #[test]
    fn multinomial_deterministic_seed() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut rng_a = 42u64;
        let mut rng_b = 42u64;
        let draws_a: Vec<u32> = (0..10)
            .map(|_| multinomial(&logits, &mut rng_a).unwrap())
            .collect();
        let draws_b: Vec<u32> = (0..10)
            .map(|_| multinomial(&logits, &mut rng_b).unwrap())
            .collect();
        assert_eq!(draws_a, draws_b);
        // Highest-mass token drawn most often over many draws.
        let mut rng = 1u64;
        let n3 = (0..200)
            .filter(|_| multinomial(&logits, &mut rng).unwrap() == 2)
            .count();
        assert!(n3 > 100, "expected token 2 to dominate, got {n3}/200");
    }

    #[test]
    fn repetition_penalty_hits_each_distinct_token_once() {
        // Token 1 appears three times; a per-occurrence penalty would cube it.
        let mut logits = vec![0.0f32, 3.0, 2.0];
        apply_repetition_penalty(&mut logits, &[1, 1, 1], 1.1);
        assert!((logits[1] - 3.0 / 1.1).abs() < 1e-5, "{}", logits[1]);
        assert_eq!(logits[2], 2.0);
    }

    #[test]
    fn repetition_penalty_window_ignores_older_tokens() {
        // Token 1 sits outside the last 2 positions, so it is untouched.
        let mut logits = vec![0.0f32, 3.0];
        apply_repetition_penalty_windowed(&mut logits, &[1, 7, 7], 1.1, 2);
        assert_eq!(logits[1], 3.0);
        // Within the window it is penalised once.
        let mut logits = vec![0.0f32, 3.0];
        apply_repetition_penalty_windowed(&mut logits, &[1, 1], 1.1, 2);
        assert!((logits[1] - 3.0 / 1.1).abs() < 1e-5);
    }
}
