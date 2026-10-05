//! `spite-bench` — kernel throughput and latency harness.
//!
//! Usage:
//!   spite-bench --model path/to/model.gguf [--n-tokens 512] [--n-runs 5]
//!
//! Outputs JSON to stdout when --json is given; human table otherwise.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
enum Device {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

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

    /// Execution device.
    #[arg(long, value_enum, default_value_t = Device::Auto)]
    device: Device,

    /// GPU card (e.g. RTX_5090).
    #[arg(long)]
    card: Option<String>,

    /// GPU architecture override (e.g. sm_120).
    #[arg(long)]
    gpu_arch: Option<String>,

    /// Path to kernels directory.
    #[arg(long, default_value = "kernels")]
    kernels_dir: PathBuf,

    /// KV-cache start tier: `TYPE` or `TYPE,TYPE` for K,V
    /// (f32, f16, q8, q5_1, q4). Defaults to the host-wide default (f16).
    #[arg(long = "kv-quant", env = "SPITE_KV_QUANT", value_name = "TYPE[,TYPE]")]
    kv_quant: Option<String>,
}

fn main() -> Result<()> {
    use spite_executor::{Executor, ExecutorConfig};
    use spite_models::{ArchRegistry, ModelConfig};
    use spite_tokenizer::Tokenizer;

    let args = Args::parse();
    let gguf = spite_loader::GgufModel::open(&args.model)?;
    let hp = spite_loader::config::ModelHyperparams::from_gguf(&gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    if let Some(ref card) = args.card {
        let normalized = spite_dispatch::normalize_card_name(card);
        unsafe {
            std::env::set_var("SPITE_CARD_ID", normalized);
        }
    }

    let gpu_arch = args
        .gpu_arch
        .clone()
        .or_else(|| {
            args.card
                .as_deref()
                .map(spite_dispatch::normalize_card_name)
                .map(|c| spite_dispatch::card_spec(&c).gpu_arch.to_string())
        })
        .unwrap_or_else(spite_dispatch::detect_gpu_arch);

    let mut peak_mem_mib = 0u64;

    let kv_cfg = match args.kv_quant.as_deref() {
        Some(s) => s
            .parse::<spite_kvcache::KvQuantConfig>()
            .map_err(|e| anyhow::anyhow!(e))?,
        None => spite_kvcache::KvQuantConfig::default(),
    };
    let kv_label = format!("{}/{}", kv_cfg.key, kv_cfg.val);

    let (model, device_label): (Box<dyn spite_models::ModelArch>, String) = match args.device {
        Device::Cpu => {
            let mut m = ArchRegistry::default().build(ModelConfig::from(hp))?;
            m.load_weights(&gguf)?;
            (m, "cpu".into())
        }
        Device::Cuda | Device::Auto => {
            match spite_models::gpu_dense::GpuDense::resolve_table(
                &hp.arch,
                &gpu_arch,
                &args.kernels_dir,
            ) {
                Some(table) => {
                    let qk_norm = hp.arch == "qwen3";
                    let (m, r) = spite_models::gpu_dense::GpuDense::load(
                        ModelConfig::from(hp),
                        &gguf,
                        table,
                        4096,
                        qk_norm,
                        &kv_cfg,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                    peak_mem_mib =
                        ((r.weights_bytes + r.kv_bytes + r.scratch_bytes) / (1024 * 1024)) as u64;
                    (Box::new(m), format!("cuda ({gpu_arch}, kv {kv_label})"))
                }
                None if args.device == Device::Cuda => {
                    anyhow::bail!(
                        "--device cuda: no CUDA kernel for arch '{}' on '{}' under {}",
                        hp.arch,
                        gpu_arch,
                        args.kernels_dir.display()
                    );
                }
                None => {
                    let mut m = ArchRegistry::default().build(ModelConfig::from(hp))?;
                    m.load_weights(&gguf)?;
                    (m, "cpu".into())
                }
            }
        }
    };

    if args.verbose {
        let mut spec =
            spite_dispatch::KernelSpec::from_arch(model.config().arch.as_str(), &gpu_arch);
        if spec.card_id.is_empty() {
            spec.card_id = spite_dispatch::detect_card_id(args.card.as_deref().unwrap_or(""));
        }
        if let Ok(table) = spite_dispatch::DispatchBuilder::new(&args.kernels_dir, spec).build() {
            table.print_sources();
        }
    }

    // Fixed prompt keeps runs comparable; real-model runs use --n-tokens.
    let prompt_ids = tokenizer.encode("Benchmark prompt for throughput measurement.", true)?;
    let exec_cfg = ExecutorConfig {
        kv_quant: kv_cfg.clone(),
        ..ExecutorConfig::default()
    };
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
        label: format!("{} [{}]", args.model, device_label),
        tps: n_tok as f64 / decode_s.max(1e-9),
        ttft_ms: ttft_ms / runs,
        peak_mem_mib,
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
