use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use spite_perplexity::{KldConfig, PplConfig, Report};

#[derive(Parser)]
#[command(
    name = "spite-perplexity",
    about = "PPL and KLD quality tests for spite kernels"
)]
struct Cli {
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "kernels")]
    kernels_dir: PathBuf,

    #[arg(long, env = "SPITE_GPU_ARCH")]
    gpu_arch: Option<String>,

    /// Write JSON report to this path (in addition to stdout).
    #[arg(long)]
    json_out: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Measure perplexity on a corpus.
    Ppl {
        /// Plain text or pre-tokenized file (one token id per line).
        #[arg(long)]
        corpus: PathBuf,

        /// Reference kernel .so to compare against (optional).
        #[arg(long)]
        reference: Option<PathBuf>,

        #[arg(long, default_value_t = 2048)]
        context_len: usize,

        #[arg(long, default_value_t = 1024)]
        stride: usize,

        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
    },

    /// Compare token distributions between two kernels using KL divergence.
    Kld {
        /// Candidate kernel .so (the one you're testing).
        #[arg(long)]
        candidate: PathBuf,

        /// Reference kernel .so (default: generic fallback).
        #[arg(long)]
        reference: Option<PathBuf>,

        #[arg(long)]
        corpus: PathBuf,

        /// KLD threshold to PASS. Default: 0.001.
        #[arg(long, default_value_t = 0.001)]
        pass_threshold: f64,

        /// KLD threshold to WARN (above this is FAIL). Default: 0.01.
        #[arg(long, default_value_t = 0.01)]
        warn_threshold: f64,

        #[arg(long, default_value_t = 1000)]
        top_k: usize,

        #[arg(long, default_value_t = 0)]
        max_positions: usize,
    },

    /// Run both PPL and KLD — the full kernel quality gate.
    All {
        #[arg(long)]
        candidate: PathBuf,

        #[arg(long)]
        reference: Option<PathBuf>,

        #[arg(long)]
        corpus: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let gpu_arch = cli.gpu_arch.unwrap_or_else(spite_dispatch::detect_gpu_arch);

    let model = spite_loader::GgufModel::open(&cli.model)?;

    let report = Report::new(&gpu_arch, model.arch(), "(see subcommand)");

    match &cli.cmd {
        Cmd::Ppl {
            corpus,
            context_len,
            stride,
            max_tokens,
            ..
        } => {
            let cfg = PplConfig {
                context_len: *context_len,
                stride: *stride,
                max_tokens: *max_tokens,
            };
            // TODO: load tokenizer, tokenize corpus, wire up forward pass
            println!("PPL evaluation not yet wired to inference loop.");
            println!("corpus={}, context_len={context_len}", corpus.display());
            let _ = cfg;
        }

        Cmd::Kld {
            candidate,
            pass_threshold,
            warn_threshold,
            top_k,
            max_positions,
            corpus,
            ..
        } => {
            let cfg = KldConfig {
                pass_threshold: *pass_threshold,
                warn_threshold: *warn_threshold,
                top_k: *top_k,
                max_positions: *max_positions,
            };
            println!("KLD evaluation not yet wired to inference loop.");
            println!(
                "candidate={}, corpus={}",
                candidate.display(),
                corpus.display()
            );
            let _ = cfg;
        }

        Cmd::All {
            candidate, corpus, ..
        } => {
            println!("Full eval (PPL + KLD) not yet wired to inference loop.");
            println!(
                "candidate={}, corpus={}",
                candidate.display(),
                corpus.display()
            );
        }
    }

    report.print();

    if let Some(out) = &cli.json_out {
        std::fs::write(out, report.to_json())?;
        println!("JSON report written to {}", out.display());
    }

    // Exit 1 if tests failed — lets CI gate on this.
    if !report.passed() {
        std::process::exit(1);
    }

    Ok(())
}
