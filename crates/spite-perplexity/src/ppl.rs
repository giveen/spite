//! Perplexity measurement.
//!
//! Runs the model over a tokenized corpus in sliding windows,
//! accumulates negative log-likelihoods, and reports PPL.
//!
//! Stride is set to half the context window so tokens near the edge
//! always have reasonable context — same methodology as most published
//! LLM perplexity numbers.

/// Configuration for a PPL run.
pub struct PplConfig {
    /// Maximum sequence length. Defaults to model's trained context length.
    pub context_len: usize,
    /// Stride between windows. Defaults to context_len / 2.
    pub stride:      usize,
    /// Maximum number of tokens to evaluate. 0 = all.
    pub max_tokens:  usize,
}

impl Default for PplConfig {
    fn default() -> Self {
        Self { context_len: 2048, stride: 1024, max_tokens: 0 }
    }
}

/// Results from a single PPL run.
#[derive(Debug, Clone)]
pub struct PplResult {
    /// Perplexity. Lower is better.
    pub ppl:     f64,
    /// Mean negative log-likelihood per token.
    pub nll:     f64,
    /// Number of tokens evaluated.
    pub n_tokens: usize,
    /// Per-token NLLs — useful for finding where the model is uncertain.
    pub token_nlls: Vec<f32>,
}

impl PplResult {
    /// Returns true if PPL is within `tolerance` percent of `reference`.
    /// Typical tolerance for a correct kernel: 0.1 (0.1%).
    pub fn within_tolerance(&self, reference: &PplResult, tolerance_pct: f64) -> bool {
        let delta = (self.ppl - reference.ppl).abs() / reference.ppl;
        delta <= tolerance_pct / 100.0
    }
}

/// Compute perplexity over `tokens` using the given forward-pass closure.
///
/// `forward` takes a token slice (the context) and returns log-probabilities
/// for the next token across the vocabulary.
///
/// This is intentionally decoupled from the model internals — the inference
/// loop wires it up once that's implemented.
pub fn compute_ppl<F>(
    tokens:  &[u32],
    cfg:     &PplConfig,
    mut forward: F,
) -> PplResult
where
    F: FnMut(&[u32]) -> Vec<f32>,  // input tokens → log-probs over vocab
{
    let limit = if cfg.max_tokens > 0 {
        tokens.len().min(cfg.max_tokens + cfg.context_len)
    } else {
        tokens.len()
    };

    let mut total_nll  = 0.0f64;
    let mut n_scored   = 0usize;
    let mut token_nlls = Vec::new();

    let mut pos = 0usize;
    while pos + cfg.context_len < limit {
        let window   = &tokens[pos..pos + cfg.context_len];
        let log_probs = forward(window);

        // Score all tokens after the first one — they have context.
        for i in 1..window.len() {
            let target = window[i] as usize;
            if target < log_probs.len() {
                let nll = -log_probs[target] as f64;
                total_nll += nll;
                token_nlls.push(nll as f32);
                n_scored  += 1;
            }
        }

        pos += cfg.stride;
    }

    let nll = if n_scored > 0 { total_nll / n_scored as f64 } else { f64::INFINITY };
    let ppl = nll.exp();

    PplResult { ppl, nll, n_tokens: n_scored, token_nlls }
}
