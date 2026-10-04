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
    use spite_executor::{Executor, ExecutorConfig};
    use spite_models::{ArchRegistry, ModelConfig};
    use spite_tokenizer::Tokenizer;

    let args = Args::parse();
    let gguf = spite_loader::GgufModel::open(&args.model)?;
    let hp = spite_loader::config::ModelHyperparams::from_gguf(&gguf);
    let mut model = ArchRegistry::default().build(ModelConfig::from(hp))?;
    model.load_weights(&gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    if args.verbose {
        let spec = spite_dispatch::KernelSpec::from_arch(model.config().arch.as_str(), "generic");
        let table = spite_dispatch::DispatchBuilder::new("kernels", spec).build()?;
        table.print_sources();
    }

    // Fixed prompt keeps runs comparable; real-model runs use --n-tokens.
    let prompt_ids = tokenizer.encode("Benchmark prompt for throughput measurement.", true)?;
    let exec_cfg = ExecutorConfig::default();
    let mut exec = Executor::new(exec_cfg);
    exec.load_model(model);

    // Warm-up (caches, allocator paths) — not timed.
    let _ = exec.generate(&tokenizer, &prompt_ids, 4, 0.0, 0)?;

    let mut ttft_ms = 0f64;
    let mut decode_s = 0f64;
    let mut n_tok = 0usize;
    for _ in 0..args.n_runs {
        let t0 = std::time::Instant::now();
        exec.reset();
        // ponytail: prefill-all + counted decode steps; chunked prefill if TTFT matters.
        let mut ids = prompt_ids.clone();
        let ctx = spite_abi::SpiteCtx {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: 4,
            pos: 0,
            n_heads: 0,
            n_kv_heads: 0,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: std::ptr::null_mut(),
            scratchpad_bytes: 0,
        };
        let mut logits = exec.prefill(&ids, &ctx)?;
        ttft_ms += t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = std::time::Instant::now();
        let cfg = spite_sampling::SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = 0u64;
        for _ in 0..args.n_tokens {
            let tok = spite_sampling::sample(&mut logits, &ids, &cfg, &mut rng)?;
            ids.push(tok);
            logits = exec.decode_step(tok, &ctx)?;
        }
        decode_s += t1.elapsed().as_secs_f64();
        n_tok += args.n_tokens;
    }
    let runs = args.n_runs.max(1) as f64;
    let result = spite_bench::BenchResult {
        label: args.model.clone(),
        tps: n_tok as f64 / decode_s.max(1e-9),
        ttft_ms: ttft_ms / runs,
        peak_mem_mib: 0,
        n_runs: args.n_runs,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result_json(&result))?);
    } else {
        result.print();
    }
    Ok(())
}

fn result_json(r: &spite_bench::BenchResult) -> serde_json::Value {
    serde_json::json!({
        "label": r.label,
        "tps": r.tps,
        "ttft_ms": r.ttft_ms,
        "peak_mem_mib": r.peak_mem_mib,
        "n_runs": r.n_runs,
    })
}
