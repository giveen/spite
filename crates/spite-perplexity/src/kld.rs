//! KL divergence test — the primary kernel quality gate.
//!
//! Compares the full probability distribution produced by a candidate kernel
//! against the reference (generic fallback) at every token position.
//!
//!   KLD(P‖Q) = Σ_x  P(x) * log( P(x) / Q(x) )
//!
//! where P = reference distribution, Q = candidate kernel distribution.
//!
//! One-sided: we measure how much information is lost by replacing the
//! reference with the candidate. A value near zero means the distributions
//! are effectively identical.
//!
//! ## Thresholds (defaults, configurable)
//!
//!   mean KLD < 0.001  →  PASS  (safe to merge)
//!   mean KLD < 0.01   →  WARN  (small distributional shift; inspect)
//!   mean KLD ≥ 0.01   →  FAIL  (do not merge — kernel is wrong)
//!
//! ## Why not just check PPL?
//!
//! Perplexity averages NLL over the corpus. A kernel can look fine on PPL
//! while systematically shifting probability mass on rare tokens — those
//! tokens have low weight in the average. KLD measures every position in
//! the distribution, catching subtle dequantization bugs that PPL misses.

/// Configuration for a KLD run.
pub struct KldConfig {
    /// PASS threshold. Default: 0.001.
    pub pass_threshold: f64,
    /// WARN threshold. Default: 0.01.
    pub warn_threshold: f64,
    /// Cap the vocabulary when computing KLD (top-k by reference probability).
    /// 0 = full vocabulary. Capping at 1000 is faster and catches real bugs.
    pub top_k: usize,
    /// Maximum number of token positions to evaluate. 0 = all.
    pub max_positions: usize,
}

impl Default for KldConfig {
    fn default() -> Self {
        Self {
            pass_threshold: 0.001,
            warn_threshold: 0.01,
            top_k: 1000,
            max_positions: 0,
        }
    }
}

/// Per-position KLD entry — useful for finding which tokens diverge most.
#[derive(Debug, Clone)]
pub struct PositionKld {
    /// The token that was the input at this position.
    pub token_id: u32,
    /// KLD at this position.
    pub kld: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KldVerdict {
    Pass,
    Warn,
    Fail,
}

impl std::fmt::Display for KldVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Warn => write!(f, "WARN"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

/// Results from a KLD comparison run.
#[derive(Debug, Clone)]
pub struct KldResult {
    pub mean_kld: f64,
    pub max_kld: f64,
    pub p95_kld: f64, // 95th percentile — robust to outliers
    pub n_positions: usize,
    pub verdict: KldVerdict,
    /// Top-10 worst positions for debugging.
    pub worst: Vec<PositionKld>,
}

impl KldResult {
    pub fn passed(&self) -> bool {
        self.verdict == KldVerdict::Pass
    }
}

/// Compute KLD between two forward-pass closures over the same token sequence.
///
/// `reference` and `candidate` each take a context window and return
/// **log-probabilities** over the full vocabulary.
///
/// Log-probs rather than probs avoids numerical underflow on large vocabs.
pub fn compute_kld<Ref, Cand>(
    tokens: &[u32],
    cfg: &KldConfig,
    mut reference: Ref,
    mut candidate: Cand,
) -> KldResult
where
    Ref: FnMut(&[u32]) -> Vec<f32>,
    Cand: FnMut(&[u32]) -> Vec<f32>,
{
    let limit = if cfg.max_positions > 0 {
        tokens.len().min(cfg.max_positions + 1)
    } else {
        tokens.len()
    };

    let mut positions: Vec<PositionKld> = Vec::new();

    for i in 1..limit {
        let ctx = &tokens[..i];

        let ref_log_probs = reference(ctx);
        let cand_log_probs = candidate(ctx);

        let kld = kld_from_logprobs(&ref_log_probs, &cand_log_probs, cfg.top_k);
        positions.push(PositionKld {
            token_id: tokens[i],
            kld,
        });
    }

    summarise(positions, cfg)
}

/// KLD(P‖Q) from log-probability vectors.
/// P = softmax(ref_log_probs), Q = softmax(cand_log_probs)
fn kld_from_logprobs(ref_lp: &[f32], cand_lp: &[f32], top_k: usize) -> f64 {
    let len = ref_lp.len().min(cand_lp.len());
    if len == 0 {
        return 0.0;
    }

    // Numerically stable softmax for P.
    let ref_max = ref_lp[..len]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let ref_exp: Vec<f64> = ref_lp[..len]
        .iter()
        .map(|&x| ((x - ref_max) as f64).exp())
        .collect();
    let ref_sum: f64 = ref_exp.iter().sum();

    // Softmax for Q.
    let cand_max = cand_lp[..len]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let cand_exp: Vec<f64> = cand_lp[..len]
        .iter()
        .map(|&x| ((x - cand_max) as f64).exp())
        .collect();
    let cand_sum: f64 = cand_exp.iter().sum();

    // If top_k, only sum over the top-k tokens by reference probability.
    let indices: Vec<usize> = if top_k > 0 && top_k < len {
        let mut idx: Vec<usize> = (0..len).collect();
        idx.sort_unstable_by(|&a, &b| {
            ref_exp[b]
                .partial_cmp(&ref_exp[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(top_k);
        idx
    } else {
        (0..len).collect()
    };

    let mut kld = 0.0f64;
    let eps = 1e-10;
    for i in indices {
        let p = ref_exp[i] / ref_sum;
        let q = cand_exp[i] / cand_sum;
        if p > eps {
            kld += p * (p / (q + eps)).ln();
        }
    }

    kld
}

fn summarise(mut positions: Vec<PositionKld>, cfg: &KldConfig) -> KldResult {
    if positions.is_empty() {
        return KldResult {
            mean_kld: 0.0,
            max_kld: 0.0,
            p95_kld: 0.0,
            n_positions: 0,
            verdict: KldVerdict::Pass,
            worst: vec![],
        };
    }

    let n = positions.len();
    let mean_kld = positions.iter().map(|p| p.kld).sum::<f64>() / n as f64;
    let max_kld = positions.iter().map(|p| p.kld).fold(0.0f64, f64::max);

    let mut sorted_klds: Vec<f64> = positions.iter().map(|p| p.kld).collect();
    sorted_klds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_idx = ((n as f64) * 0.95) as usize;
    let p95_kld = sorted_klds[p95_idx.min(n - 1)];

    let verdict = if mean_kld < cfg.pass_threshold {
        KldVerdict::Pass
    } else if mean_kld < cfg.warn_threshold {
        KldVerdict::Warn
    } else {
        KldVerdict::Fail
    };

    positions.sort_unstable_by(|a, b| {
        b.kld
            .partial_cmp(&a.kld)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let worst = positions.into_iter().take(10).collect();

    KldResult {
        mean_kld,
        max_kld,
        p95_kld,
        n_positions: n,
        verdict,
        worst,
    }
}
