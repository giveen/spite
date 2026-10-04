//! `spite-bench` — kernel throughput and latency harness.
//!
//! Usage:
//!   spite-bench --model path/to/model.gguf [--n-tokens 512] [--n-runs 5]
//!
//! Outputs JSON to stdout when --json is given; human table otherwise.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "spite-bench", about = "Benchmark spite kernel throughput")]
struct Args {
    /// Path to the GGUF model file.
    #[arg(long)]
    model: String,

    /// Number of decode tokens per run.
    #[arg(long, default_value_t = 512)]
    n_tokens: usize,

    /// Number of benchmark runs (results averaged).
    #[arg(long, default_value_t = 5)]
    n_runs: usize,

    /// Emit results as JSON instead of a human table.
    #[arg(long)]
    json: bool,

    /// Print which kernel won each dispatch slot.
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // TODO: load model via spite-loader, build DispatchTable, run warm-up then
    //       timed loops, emit BenchResult per kernel slot.
    println!("spite-bench: model={} n_tokens={} n_runs={}", args.model, args.n_tokens, args.n_runs);
    Ok(())
}
