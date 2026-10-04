//! XTC (Exclude Top Choices) sampler.
//!
//! Described in the llama.cpp community: instead of top-p, XTC *removes* the
//! tokens that are too likely (top of the distribution) so the model is forced
//! to pick from the long tail. Produces more varied, creative output.
//!
//! Algorithm:
//!   1. Compute softmax probabilities.
//!   2. Sort tokens by descending probability.
//!   3. Remove all tokens whose cumulative probability exceeds `threshold`,
//!      BUT only if at least `min_keep` tokens remain after removal.
//!   4. Scale `probability` controls the strength: with probability `p`,
//!      apply XTC; otherwise pass through unchanged.
//!
//! Parameters:
//! - `threshold`:    remove tokens from the top of the distribution (0..1).
//!   Typical: 0.1 (remove tokens contributing the top 10% mass).
//! - `probability`:  fraction of tokens that actually undergo XTC (0..1).
//!   1.0 = always apply; 0.0 = never apply.
//! - `min_keep`:     always keep at least this many tokens.

/// Apply XTC to a logit vector.
///
/// `logits` is modified in-place (excluded tokens set to −∞).
/// `rng`:  LCG state, consumed for the probability gate.
pub fn apply_xtc(
    logits: &mut [f32],
    threshold: f32,
    probability: f32,
    min_keep: usize,
    rng: &mut u64,
) {
    if probability <= 0.0 || threshold >= 1.0 {
        return;
    }

    // Probability gate: skip XTC this call if random draw > probability.
    let r = crate::lcg_f32(rng);
    if r > probability {
        return;
    }

    let n = logits.len();
    if n <= min_keep {
        return;
    }

    // Softmax to get probabilities.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv = 1.0 / sum.max(1e-30);
    for p in probs.iter_mut() {
        *p *= inv;
    }

    // Sort indices by descending probability.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    // Walk sorted list; accumulate prob mass, exclude tokens above threshold.
    // The token that pushes cumulative mass over `threshold` is also excluded.
    let mut cum = 0f32;
    let mut cut = 0usize; // how many top tokens to exclude
    for &i in &idx {
        cum += probs[i];
        cut += 1;
        if cum >= threshold {
            break;
        }
    }

    // Don't exclude if it would leave fewer than min_keep.
    if n.saturating_sub(cut) < min_keep {
        cut = n.saturating_sub(min_keep);
    }

    // Set excluded tokens to −∞.
    for &i in idx.iter().take(cut) {
        logits[i] = f32::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xtc_removes_top_tokens() {
        // Strongly biased distribution: token 0 has nearly all the mass.
        let mut logits = vec![10.0f32, 0.0, 0.0, 0.0, 0.0];
        let mut rng = 1u64;
        apply_xtc(&mut logits, 0.5, 1.0, 1, &mut rng);
        // Token 0 should be excluded (it holds > 50% of the mass by itself).
        assert!(
            logits[0] == f32::NEG_INFINITY || logits[0] < 0.0,
            "token 0 should be suppressed, got {}",
            logits[0]
        );
    }

    #[test]
    fn xtc_respects_min_keep() {
        let mut logits = vec![10.0f32, 9.0, 8.0];
        let mut rng = 1u64;
        // Even though top tokens hold most mass, min_keep=3 prevents exclusion.
        apply_xtc(&mut logits, 0.01, 1.0, 3, &mut rng);
        assert!(logits.iter().all(|&l| l > f32::NEG_INFINITY));
    }
}
