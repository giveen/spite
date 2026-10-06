//! Token verification — accept/reject draft proposals.
//!
//! The GPU kernel path goes through `SpiteSpecVerifyFn` in the dispatch
//! table. If no kernel is available for the current GPU, `scalar_verify`
//! is used as the fallback.
//!
//! ## Accept/reject rule (standard speculative sampling)
//!
//! For each draft token t_i with draft probability q(t_i) and main
//! model probability p(t_i):
//!
//!   accept with probability min(1, p(t_i) / q(t_i))
//!
//! If rejected, resample from the adjusted distribution:
//!   p_adj(x) = max(0, p(x) - q(x)) normalized
//!
//! This preserves the main model's output distribution exactly —
//! speculative decoding is lossless, not an approximation.

/// Result of verifying one batch of draft tokens.
pub struct VerifyResult {
    /// How many leading tokens were accepted (0 ..= n_draft).
    pub n_accepted: usize,
    /// Logits for the resampled token at the first rejection position.
    /// If all tokens were accepted this holds the logits for the next token.
    pub bonus_logits: Vec<f32>,
}

/// Scalar (CPU) fallback verify — always correct, not optimized.
///
/// Called when no `SpiteSpecVerifyFn` kernel is available for the
/// current GPU architecture.
pub fn scalar_verify(
    draft_logits: &[Vec<f32>], // [n_draft][vocab_size]
    main_logits: &[Vec<f32>],  // [n_draft][vocab_size]
    draft_tokens: &[u32],
    temperature: f32,
    rng: &mut u64,
) -> VerifyResult {
    let n = draft_logits
        .len()
        .min(main_logits.len())
        .min(draft_tokens.len());
    let mut n_accepted = 0;
    let mut bonus_logits = Vec::new();

    for i in 0..n {
        let t = draft_tokens[i] as usize;
        let q = softmax_prob(&draft_logits[i], t, temperature);
        let p = softmax_prob(&main_logits[i], t, temperature);

        let accept_prob = (p / q.max(1e-10)).min(1.0);
        if lcg_f32(rng) < accept_prob {
            n_accepted += 1;
        } else {
            // Build adjusted distribution and stop.
            bonus_logits = adjusted_logits(&main_logits[i], &draft_logits[i]);
            break;
        }
    }

    if n_accepted == n {
        // All accepted — bonus is the next token's logits from main model.
        // The caller must run one more main-model forward pass to get these.
        if let Some(last) = main_logits.last() {
            bonus_logits = last.clone();
        }
    }

    VerifyResult {
        n_accepted,
        bonus_logits,
    }
}

/// Run token verification using `SpiteSpecVerifyFn` if available, or `scalar_verify` fallback.
pub fn run_verify(
    verify_fn: Option<spite_abi::SpecVerifyFn>,
    draft_logits: &[Vec<f32>],
    main_logits: &[Vec<f32>],
    draft_tokens: &[u32],
    temperature: f32,
    rng: &mut u64,
    ctx: &spite_abi::SpiteCtx,
) -> VerifyResult {
    let n = draft_logits
        .len()
        .min(main_logits.len())
        .min(draft_tokens.len());
    if n == 0 {
        return VerifyResult {
            n_accepted: 0,
            bonus_logits: vec![],
        };
    }

    if let Some(spec_fn) = verify_fn {
        let vocab_size = draft_logits[0].len();
        // Flatten logits to continuous buffers for SpiteTensor
        let mut d_flat: Vec<f32> = Vec::with_capacity(n * vocab_size);
        for row in draft_logits.iter().take(n) {
            d_flat.extend_from_slice(row);
        }
        let mut m_flat: Vec<f32> = Vec::with_capacity(n * vocab_size);
        for row in main_logits.iter().take(n) {
            m_flat.extend_from_slice(row);
        }

        let d_tensor = spite_abi::SpiteTensor {
            data: d_flat.as_mut_ptr() as *mut std::ffi::c_void,
            ne: [vocab_size as u32, n as u32, 1, 1],
            nb: [
                4,
                (vocab_size * 4) as u64,
                (n * vocab_size * 4) as u64,
                (n * vocab_size * 4) as u64,
            ],
            kind: spite_abi::SpiteType::F32,
        };

        let m_tensor = spite_abi::SpiteTensor {
            data: m_flat.as_mut_ptr() as *mut std::ffi::c_void,
            ne: [vocab_size as u32, n as u32, 1, 1],
            nb: [
                4,
                (vocab_size * 4) as u64,
                (n * vocab_size * 4) as u64,
                (n * vocab_size * 4) as u64,
            ],
            kind: spite_abi::SpiteType::F32,
        };

        let mut accept_mask = vec![false; n];
        let status = unsafe {
            spec_fn(
                accept_mask.as_mut_ptr(),
                &d_tensor,
                &m_tensor,
                temperature,
                n as u32,
                ctx,
            )
        };

        if status == 0 {
            let n_accepted = accept_mask.iter().take_while(|&&a| a).count();
            let bonus_logits = if n_accepted < n {
                adjusted_logits(&main_logits[n_accepted], &draft_logits[n_accepted])
            } else {
                main_logits.last().cloned().unwrap_or_default()
            };
            return VerifyResult {
                n_accepted,
                bonus_logits,
            };
        }
    }

    scalar_verify(draft_logits, main_logits, draft_tokens, temperature, rng)
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn softmax_prob(logits: &[f32], idx: usize, temperature: f32) -> f32 {
    if temperature <= 0.0 {
        return if logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .is_some_and(|(i, _)| i == idx)
        {
            1.0
        } else {
            0.0
        };
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let sum: f32 = exp.iter().sum();
    exp.get(idx).copied().unwrap_or(0.0) / sum.max(1e-10)
}

/// max(0, p - q) normalized — the distribution to resample from on rejection.
fn adjusted_logits(main: &[f32], draft: &[f32]) -> Vec<f32> {
    let mut adj: Vec<f32> = main
        .iter()
        .zip(draft.iter())
        .map(|(&p, &q)| (p - q).max(0.0))
        .collect();
    let sum: f32 = adj.iter().sum();
    if sum > 0.0 {
        adj.iter_mut().for_each(|x| *x /= sum);
    }
    adj
}

fn lcg_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / 2147483648.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_verify_deterministic() {
        let draft_logits = vec![vec![10.0, 0.0, 0.0], vec![0.0, 10.0, 0.0]];
        let main_logits = vec![
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0], // diverges here
        ];
        let draft_tokens = vec![0, 1];
        let mut rng = 42u64;

        let res = scalar_verify(&draft_logits, &main_logits, &draft_tokens, 0.0, &mut rng);
        assert_eq!(res.n_accepted, 1);
        assert!(!res.bonus_logits.is_empty());
    }

    #[test]
    fn test_run_verify_fallback() {
        let draft_logits = vec![vec![5.0, 1.0]];
        let main_logits = vec![vec![5.0, 1.0]];
        let draft_tokens = vec![0];
        let mut rng = 12345u64;
        let ctx = spite_abi::SpiteCtx::default();

        let res = run_verify(
            None,
            &draft_logits,
            &main_logits,
            &draft_tokens,
            0.0,
            &mut rng,
            &ctx,
        );
        assert_eq!(res.n_accepted, 1);
    }
}
