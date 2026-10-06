//! Benchmarking primitives shared between the binary and integration tests.

/// A single benchmark result — one op, one configuration.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub label: String,
    /// Tokens per second (decode).
    pub tps: f64,
    /// Tokens per second (prefill).
    pub prefill_tps: f64,
    /// Time-to-first-token in milliseconds (prefill).
    pub ttft_ms: f64,
    /// Peak GPU memory in MiB.
    pub peak_mem_mib: u64,
    pub n_runs: usize,
    /// Speculative acceptance rate (0.0 .. 1.0) when speculative decoding is enabled.
    pub acceptance_rate: Option<f64>,
}

impl BenchResult {
    pub fn print(&self) {
        if let Some(acc) = self.acceptance_rate {
            println!(
                "{:<40}  {:>8.1} tok/s  prefill {:>8.1} tok/s  TTFT {:>7.1} ms  mem {:>6} MiB  acceptance {:>5.1}%  (n={})",
                self.label,
                self.tps,
                self.prefill_tps,
                self.ttft_ms,
                self.peak_mem_mib,
                acc * 100.0,
                self.n_runs
            );
        } else {
            println!(
                "{:<40}  {:>8.1} tok/s  prefill {:>8.1} tok/s  TTFT {:>7.1} ms  mem {:>6} MiB  (n={})",
                self.label,
                self.tps,
                self.prefill_tps,
                self.ttft_ms,
                self.peak_mem_mib,
                self.n_runs
            );
        }
    }
}
