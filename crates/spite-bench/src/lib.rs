//! Benchmarking primitives shared between the binary and integration tests.

/// A single benchmark result — one op, one configuration.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub label:        String,
    /// Tokens per second (decode).
    pub tps:          f64,
    /// Time-to-first-token in milliseconds (prefill).
    pub ttft_ms:      f64,
    /// Peak GPU memory in MiB.
    pub peak_mem_mib: u64,
    pub n_runs:       usize,
}

impl BenchResult {
    pub fn print(&self) {
        println!(
            "{:<40}  {:>9.1} tok/s  TTFT {:>7.1} ms  mem {:>6} MiB  (n={})",
            self.label, self.tps, self.ttft_ms, self.peak_mem_mib, self.n_runs
        );
    }
}
