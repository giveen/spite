//! spite — One Engine, Your Model, Your Card.
//!
//! Usage:
//!   spite run   -m Qwen/Qwen3-27B --card RTX_5090 -p "Hello"
//!   spite serve -m Qwen/Qwen3-27B --card RTX_5090
//!   spite run   -m Qwen/Qwen3-27B,Qwen/Qwen3-1.7B --card RTX_5090 -p "Hello"
//!   spite dispatch -m Qwen/Qwen3-27B --card RTX_5090
//!   spite pull  Qwen/Qwen3-27B --quant Q4_K_M

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use spite_dispatch::{
    card_spec, normalize_card_name,
    DispatchBuilder, KernelSpec, detect_gpu_arch,
};
use spite_loader::GgufModel;

// ── Shared flag groups ─────────────────────────────────────────────────────

/// Which GPU to target.
#[derive(Args, Clone)]
struct HardwareArgs {
    /// GPU card name: "RTX_5090", "RX_9900_XTX", "MI300X", "M4_Max", …
    /// Automatically determines the GPU arch and VRAM.
    /// Overrides --gpu-arch when both are given.
    #[arg(long = "card", env = "SPITE_CARD", value_name = "CARD")]
    card: Option<String>,

    /// Override GPU arch directly (e.g. sm_89, rdna3, metal, generic).
    /// Use --card when possible — it also supplies VRAM info.
    #[arg(long = "gpu-arch", env = "SPITE_GPU_ARCH", value_name = "ARCH")]
    gpu_arch: Option<String>,

    /// Directory that contains compiled kernel .so files.
    #[arg(
        long = "kernels-dir",
        env  = "SPITE_KERNELS_DIR",
        default_value = "kernels",
        value_name = "DIR",
    )]
    kernels_dir: PathBuf,
}

/// Which model(s) to load.
#[derive(Args, Clone)]
struct ModelArgs {
    /// Model name or path.
    ///
    /// • "Qwen/Qwen3-27B"          — look in $SPITE_MODELS_DIR
    /// • "/abs/path/model.gguf"    — direct path
    /// • "target,draft"             — two models for speculative decoding
    #[arg(
        short = 'm',
        long  = "model",
        value_name = "MODEL[,DRAFT]",
    )]
    model: String,
}

// ── CLI structure ──────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "spite",
    about   = "One Engine, Your Model, Your Card.",
    version,
    propagate_version = true,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate tokens from a single prompt.
    Run {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        hw: HardwareArgs,

        /// Prompt text.
        #[arg(short = 'p', long, value_name = "TEXT")]
        prompt: String,

        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 512, value_name = "N")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy, 1 = default).
        #[arg(long, default_value_t = 1.0, value_name = "T")]
        temperature: f32,
    },

    /// Start an OpenAI-compatible HTTP inference server.
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        hw: HardwareArgs,

        /// Listen address.
        #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
        host: String,

        /// Listen port.
        #[arg(short, long, default_value_t = 8080, value_name = "PORT")]
        port: u16,
    },

    /// Show which kernel wins each op for this model + card.
    Dispatch {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        hw: HardwareArgs,
    },

    /// Download a model from HuggingFace Hub into $SPITE_MODELS_DIR.
    Pull {
        /// Model name: "Qwen/Qwen3-27B"
        model: String,

        /// Quantisation to download.
        #[arg(long, default_value = "Q4_K_M", value_name = "QUANT")]
        quant: String,
    },
}

// ── Entry point ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Run { model, hw, prompt, max_tokens, temperature } => {
            cmd_run(&model, &hw, &prompt, max_tokens, temperature)
        }
        Cmd::Serve { model, hw, host, port } => {
            cmd_serve(&model, &hw, &host, port)
        }
        Cmd::Dispatch { model, hw } => {
            cmd_dispatch(&model, &hw)
        }
        Cmd::Pull { model, quant } => {
            cmd_pull(&model, &quant)
        }
    }
}

// ── Resolution helpers ─────────────────────────────────────────────────────

/// Resolve card name + optional arch override → (card_id, gpu_arch, vram_gib).
fn resolve_hardware(hw: &HardwareArgs) -> (String, String, u32) {
    let card_id = hw.card.as_deref()
        .map(normalize_card_name)
        .unwrap_or_default();

    let spec = card_spec(&card_id);

    let gpu_arch = hw.gpu_arch.clone()
        .unwrap_or_else(|| {
            if !card_id.is_empty() && spec.gpu_arch != "generic" {
                spec.gpu_arch.to_owned()
            } else {
                detect_gpu_arch()
            }
        });

    (card_id, gpu_arch, spec.vram_gib)
}

/// Returns the directory where models are stored.
fn models_dir() -> PathBuf {
    std::env::var("SPITE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".spite").join("models"))
                .unwrap_or_else(|_| PathBuf::from(".spite/models"))
        })
}

/// Resolve a model name to a GGUF path.
///
/// Accepted forms:
///   - Absolute path or relative path that exists → used as-is
///   - "Org/Name" or "Name"                       → searched in models_dir
fn resolve_model_path(name: &str) -> Result<PathBuf> {
    // Direct path?
    let p = Path::new(name);
    if p.is_absolute() || name.starts_with("./") || name.starts_with("../") {
        if p.exists() {
            return Ok(p.to_owned());
        }
        bail!("model path not found: {name}");
    }
    // Bare name or Org/Name — search in models_dir.
    let dir = models_dir();
    if !dir.exists() {
        bail!(
            "model '{name}' not found as a path; \
             $SPITE_MODELS_DIR / ~/.spite/models does not exist.\n\
             Run `spite pull {name}` to download it, or set SPITE_MODELS_DIR."
        );
    }

    // Try a few candidate filenames.
    let candidates = {
        let base = name.replace('/', "--");
        vec![
            dir.join(format!("{base}.gguf")),
            dir.join(name).with_extension("gguf"),  // Org/Name.gguf
            dir.join(name.split('/').last().unwrap_or(name)).with_extension("gguf"),
        ]
    };
    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    // Fuzzy: find first *.gguf whose file stem contains the short name.
    let short = name.split('/').last().unwrap_or(name).to_lowercase();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().map_or(false, |e| e == "gguf") {
                let stem = p.file_stem().unwrap_or_default()
                    .to_string_lossy().to_lowercase();
                if stem.contains(&short) {
                    return Ok(p);
                }
            }
        }
    }

    bail!(
        "model '{name}' not found in {}.\n\
         Run `spite pull {name}` to download it, or point SPITE_MODELS_DIR \
         at the directory containing the .gguf file.",
        dir.display()
    )
}

/// Resolve model names → paths, handling the "target,draft" speculative form.
fn resolve_models(spec: &str) -> Result<(PathBuf, Option<PathBuf>)> {
    let parts: Vec<&str> = spec.splitn(2, ',').collect();
    let target = resolve_model_path(parts[0].trim())?;
    let draft  = parts.get(1)
        .map(|s| resolve_model_path(s.trim()))
        .transpose()?;
    Ok((target, draft))
}

// ── Command implementations ────────────────────────────────────────────────

fn cmd_run(
    model_args:  &ModelArgs,
    hw:          &HardwareArgs,
    prompt:      &str,
    max_tokens:  usize,
    temperature: f32,
) -> Result<()> {
    let (target, draft) = resolve_models(&model_args.model)?;
    let (card_id, gpu_arch, vram_gib) = resolve_hardware(hw);

    let target_gguf = GgufModel::open(&target)?;
    let model_arch  = target_gguf.arch().to_owned();

    print_engine_header(&model_arch, &gpu_arch, &card_id, vram_gib);

    if let Some(ref d) = draft {
        let d_gguf = GgufModel::open(d)?;
        println!("draft model  : {}", d_gguf.arch());
        println!("mode         : speculative decoding");
    }

    let spec  = KernelSpec::from_arch(&model_arch, &gpu_arch);
    let table = DispatchBuilder::new(&hw.kernels_dir, spec).build()?;
    table.print_sources();

    println!("\nprompt       : {prompt}");
    println!("max tokens   : {max_tokens}");
    println!("temperature  : {temperature}");
    println!("\n(inference not yet implemented — contribute it!)");
    Ok(())
}

fn cmd_serve(
    model_args: &ModelArgs,
    hw:         &HardwareArgs,
    host:       &str,
    port:       u16,
) -> Result<()> {
    let (target, draft) = resolve_models(&model_args.model)?;
    let (card_id, gpu_arch, vram_gib) = resolve_hardware(hw);

    let target_gguf = GgufModel::open(&target)?;
    let model_arch  = target_gguf.arch().to_owned();

    print_engine_header(&model_arch, &gpu_arch, &card_id, vram_gib);

    if let Some(ref d) = draft {
        let d_gguf = GgufModel::open(d)?;
        println!("draft model  : {} [speculative]", d_gguf.arch());
    }

    let spec  = KernelSpec::from_arch(&model_arch, &gpu_arch);
    let table = DispatchBuilder::new(&hw.kernels_dir, spec).build()?;
    table.print_sources();

    println!("\nlistening on : http://{host}:{port}");
    println!("(server loop not yet implemented — contribute it!)");
    Ok(())
}

fn cmd_dispatch(model_args: &ModelArgs, hw: &HardwareArgs) -> Result<()> {
    let (target, _) = resolve_models(&model_args.model)?;
    let (card_id, gpu_arch, vram_gib) = resolve_hardware(hw);

    let gguf       = GgufModel::open(&target)?;
    let model_arch = gguf.arch().to_owned();

    print_engine_header(&model_arch, &gpu_arch, &card_id, vram_gib);

    let spec  = KernelSpec::from_arch(&model_arch, &gpu_arch);
    let table = DispatchBuilder::new(&hw.kernels_dir, spec).build()
        .context("kernel dispatch build failed")?;
    table.print_sources();
    Ok(())
}

fn cmd_pull(model: &str, quant: &str) -> Result<()> {
    let dir = models_dir();
    println!("models dir   : {}", dir.display());
    println!("model        : {model}");
    println!("quant        : {quant}");
    println!("\n(download not yet implemented — contribute it!)");
    println!("For now, download the .gguf manually and place it in:");
    println!("  {}/{}.gguf", dir.display(), model.replace('/', "--"));
    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn print_engine_header(model_arch: &str, gpu_arch: &str, card_id: &str, vram_gib: u32) {
    println!("spite — One Engine, Your Model, Your Card.");
    println!("──────────────────────────────────────────");
    println!("model arch   : {model_arch}");
    if card_id.is_empty() {
        println!("gpu arch     : {gpu_arch}");
    } else {
        let vram_str = if vram_gib > 0 { format!(" ({vram_gib} GiB)") } else { String::new() };
        println!("card         : {card_id}{vram_str}");
        println!("gpu arch     : {gpu_arch}");
    }
    println!("kernels      :");
}
