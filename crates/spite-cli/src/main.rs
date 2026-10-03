use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use spite_dispatch::{DispatchBuilder, KernelSpec, detect_gpu_arch};
use spite_loader::GgufModel;

#[derive(Parser)]
#[command(name = "spite", about = "Hyper-modular LLM inference engine")]
struct Cli {
    /// Override GPU arch detection (e.g. sm_89, rdna3)
    #[arg(long, env = "SPITE_GPU_ARCH")]
    gpu_arch: Option<String>,

    /// Path to the compiled kernels directory
    #[arg(long, default_value = "kernels")]
    kernels_dir: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run inference on a prompt
    Run {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
    },

    /// Show which kernel is active for each op and its latency
    Benchmark {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        verbose: bool,
    },

    /// Verify a kernel's output against the reference implementation
    Verify {
        /// Path to the kernel .so / .dylib
        kernel: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
    },

    /// Benchmark a specific kernel against the current fallback
    Bench {
        /// Path to the kernel .so / .dylib
        kernel: PathBuf,
        #[arg(long)]
        model: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let gpu_arch = cli.gpu_arch
        .unwrap_or_else(detect_gpu_arch);

    match cli.cmd {
        Cmd::Run { model, prompt, max_tokens } => {
            cmd_run(&model, &prompt, max_tokens, &gpu_arch, &cli.kernels_dir)
        }
        Cmd::Benchmark { model, verbose } => {
            cmd_benchmark(&model, verbose, &gpu_arch, &cli.kernels_dir)
        }
        Cmd::Verify { kernel, model } => {
            cmd_verify(&kernel, model.as_deref())
        }
        Cmd::Bench { kernel, model } => {
            cmd_bench(&kernel, &model)
        }
    }
}

// ── Commands ───────────────────────────────────────────────────────────────

fn cmd_run(
    model_path: &std::path::Path,
    prompt:     &str,
    _max_tokens: usize,
    gpu_arch:   &str,
    kernels_dir: &std::path::Path,
) -> Result<()> {
    let model = GgufModel::open(model_path)?;
    println!("model arch : {}", model.arch());
    println!("gpu arch   : {gpu_arch}");

    let spec  = KernelSpec::from_arch(model.arch(), gpu_arch);
    let table = DispatchBuilder::new(kernels_dir, spec).build()?;
    println!("dispatch   :");
    table.print_sources();

    println!("\nprompt: {prompt}");
    println!("(inference loop not yet implemented)");
    Ok(())
}

fn cmd_benchmark(
    model_path:  &std::path::Path,
    verbose:     bool,
    gpu_arch:    &str,
    kernels_dir: &std::path::Path,
) -> Result<()> {
    let model = GgufModel::open(model_path)?;
    let spec  = KernelSpec::from_arch(model.arch(), gpu_arch);
    let table = DispatchBuilder::new(kernels_dir, spec).build()?;

    println!("[dispatch] model={} gpu={gpu_arch}", model.arch());
    table.print_sources();

    if verbose {
        println!("\n{} tensors loaded", model.n_tensors());
    }

    // TODO: warm up each op and print µs per token
    println!("\n(timing not yet implemented — contribute it!)");
    Ok(())
}

fn cmd_verify(kernel: &std::path::Path, _model: Option<&std::path::Path>) -> Result<()> {
    println!("verifying {}", kernel.display());
    // TODO: load kernel, run ops with random inputs, compare against reference
    println!("(verify not yet implemented)");
    Ok(())
}

fn cmd_bench(kernel: &std::path::Path, model: &std::path::Path) -> Result<()> {
    println!("benching {} against {}", kernel.display(), model.display());
    // TODO: load kernel, run ops, compare throughput against current fallback
    println!("(bench not yet implemented)");
    Ok(())
}
