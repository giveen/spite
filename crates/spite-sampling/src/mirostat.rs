//! Mirostat v1 and v2 adaptive sampling.
//!
//! Dynamically adjusts the sampling truncation to target a desired perplexity
//! level τ (in bits), keeping generation quality stable throughout long outputs.
//!
//! Mirostat v2 algorithm (simpler, default):
//!   1. Sort tokens by probability descending.
//!   2. Truncate to the smallest prefix whose sum ≥ a threshold derived from μ.
//!   3. Sample from the truncated, renormalised distribution.
//!   4. Observe the actual surprise of the sampled token: s = -log2(p_token)
//!   5. Update μ ← μ + η × (s − τ)
//!
//! Reference: Basu et al., "Mirostat: A Neural Text Decoding Algorithm
//!            that Directly Controls Perplexity", ICLR 2021.

use crate::lcg_f32;

#[derive(Debug, Clone)]
pub struct MirostatConfig {
    /// 1 (v1, Zipf-based) or 2 (v2, μ-tracking, default).
    pub version: u8,
    /// Target entropy in bits. Typical: 5.0.
    pub tau: f32,
    /// Learning rate for μ updates. Typical: 0.1.
    pub eta: f32,
}

impl Default for MirostatConfig {
    fn default() -> Self {
        Self {
            version: 2,
            tau: 5.0,
            eta: 0.1,
        }
    }
}

/// Per-sequence mirostat state — carry this across tokens.
#[derive(Debug, Clone)]
pub struct MirostatState {
    pub cfg: MirostatConfig,
    /// Running estimate of the cross-entropy; initialised to 2 × τ.
    pub mu: f32,
}

impl MirostatState {
    pub fn new(cfg: MirostatConfig) -> Self {
        let mu = 2.0 * cfg.tau;
        Self { cfg, mu }
    }

    /// Sample one token from `probs` (already softmax'd) and update μ.
    ///
    /// Returns the sampled token index.
    pub fn sample(&mut self, probs: &[f32], rng: &mut u64) -> usize {
        // Sort indices by probability descending
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to tokens whose probability > 2^(-μ)
        let threshold = 2f32.powf(-self.mu);
        let mut cum = 0f32;
        let mut cutoff = indices.len();
        for (k, &i) in indices.iter().enumerate() {
            if probs[i] < threshold && k > 0 {
                cutoff = k;
                break;
            }
            cum += probs[i];
        }
        let candidates = &indices[..cutoff];

        // Sample from renormalised candidates
        let u = lcg_f32(rng) * cum;
        let mut acc = 0f32;
        let mut sampled = candidates[0];
        for &i in candidates {
            acc += probs[i];
            if u <= acc {
                sampled = i;
                break;
            }
        }

        // Update μ
        let surprise = -probs[sampled].max(1e-30).log2();
        self.mu += self.cfg.eta * (surprise - self.cfg.tau);

        sampled
    }
}
