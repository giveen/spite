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
    DispatchBuilder, KernelSpec, MultiGpuSpec, detect_gpu_arch,
};
use spite_loader::GgufModel;
use spite_offload::{OffloadConfig, TieredPlacement};

// ── Shared flag groups ─────────────────────────────────────────────────────

/// Which GPU(s) to target.
///
/// Single GPU:
///   --card RTX_5090
///
/// Multiple GPUs (pipeline parallelism):
///   --card RTX_5070,RTX_3090          same-vendor mixed-arch
///   --card RTX_4090,RX_7900_XTX       cross-vendor (activations via host RAM)
///
/// Layer assignment is automatic (proportional to VRAM).
/// Override with explicit counts: --layer-split 20,12
#[derive(Args, Clone)]
struct HardwareArgs {
    /// GPU card(s). Comma-separated or repeat the flag for multiple GPUs.
    #[arg(
        long = "card",
        env  = "SPITE_CARD",
        value_name = "CARD[,CARD…]",
        value_delimiter = ',',
    )]
    cards: Vec<String>,

    /// Override GPU arch directly (e.g. sm_89, rdna3, metal, generic).
    /// Applies to the first GPU; ignored when --card is given.
    #[arg(long = "gpu-arch", env = "SPITE_GPU_ARCH", value_name = "ARCH")]
    gpu_arch: Option<String>,

    /// Explicit layer counts per GPU (comma-separated integers).
    /// Sum need not equal the model's layer count — counts are re-scaled.
    /// E.g. --layer-split 20,12 gives the first GPU ≈20/32 of the layers.
    #[arg(long = "layer-split", value_name = "N[,N…]", value_delimiter = ',')]
    layer_split: Vec<u32>,

    /// Directory that contains compiled kernel .so files.
    #[arg(
        long = "kernels-dir",
        env  = "SPITE_KERNELS_DIR",
        default_value = "kernels",
        value_name = "DIR",
    )]
    kernels_dir: PathBuf,

    /// Enable weight offload: layers that don't fit in VRAM spill into system
    /// RAM.  Requires sufficient free RAM; add --offload-disk to also allow
    /// spilling to disk (via mmap).
    #[arg(long = "offload-ram", env = "SPITE_OFFLOAD_RAM")]
    offload_ram: bool,

    /// Enable disk offload (implies --offload-ram): layers that don't fit in
    /// VRAM or RAM are streamed from the .gguf file via mmap (~7 GB/s NVMe).
    /// Lets you run any model on any GPU as long as you have the disk space.
    #[arg(long = "offload-disk", env = "SPITE_OFFLOAD_DISK")]
    offload_disk: bool,

    /// Bytes to reserve in VRAM for KV cache and activations (default: 2 GiB).
    #[arg(
        long = "vram-reserve-gib",
        env  = "SPITE_VRAM_RESERVE_GIB",
        default_value_t = 2,
        value_name = "GIB",
    )]
    vram_reserve_gib: u32,
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

/// Resolve hardware args → a `MultiGpuSpec` (works for 1 or N GPUs).
fn resolve_hardware(hw: &HardwareArgs) -> MultiGpuSpec {
    if hw.cards.is_empty() {
        // No --card given: fall back to arch detection or generic.
        let arch = hw.gpu_arch.clone().unwrap_or_else(detect_gpu_arch);
        let raw  = std::env::var("SPITE_CARD").unwrap_or_default();
        let card = if raw.is_empty() { arch.clone() } else { normalize_card_name(&raw) };
        // Synthesise a single node.
        let refs: Vec<&str> = vec![card.as_str()];
        MultiGpuSpec::from_cards(&refs)
    } else {
        let refs: Vec<&str> = hw.cards.iter().map(String::as_str).collect();
        MultiGpuSpec::from_cards(&refs)
    }
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
    let mgpu            = resolve_hardware(hw);

    let target_gguf = GgufModel::open(&target)?;
    let model_arch  = target_gguf.arch().to_owned();

    print_engine_header(&model_arch, &mgpu);

    if let Some(ref d) = draft {
        let d_gguf = GgufModel::open(d)?;
        println!("draft model  : {} [speculative decoding]", d_gguf.arch());
    }

    print_dispatch_tables(&model_arch, &mgpu, &hw.kernels_dir)?;

    let vram_gib    = mgpu.nodes.first().map_or(0, |n| n.vram_gib);
    let model_bytes = std::fs::metadata(&target).map_or(0, |m| m.len());
    maybe_print_offload_plan(hw, vram_gib, model_bytes, 0);

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
    let mgpu            = resolve_hardware(hw);

    let target_gguf = GgufModel::open(&target)?;
    let model_arch  = target_gguf.arch().to_owned();

    print_engine_header(&model_arch, &mgpu);

    if let Some(ref d) = draft {
        let d_gguf = GgufModel::open(d)?;
        println!("draft model  : {} [speculative decoding]", d_gguf.arch());
    }

    print_dispatch_tables(&model_arch, &mgpu, &hw.kernels_dir)?;

    let vram_gib    = mgpu.nodes.first().map_or(0, |n| n.vram_gib);
    let model_bytes = std::fs::metadata(&target).map_or(0, |m| m.len());
    maybe_print_offload_plan(hw, vram_gib, model_bytes, 0);

    println!("\nlistening on : http://{host}:{port}");
    println!("(server loop not yet implemented — contribute it!)");
    Ok(())
}

fn cmd_dispatch(model_args: &ModelArgs, hw: &HardwareArgs) -> Result<()> {
    let (target, _) = resolve_models(&model_args.model)?;
    let mgpu        = resolve_hardware(hw);

    let gguf       = GgufModel::open(&target)?;
    let model_arch = gguf.arch().to_owned();

    print_engine_header(&model_arch, &mgpu);
    print_dispatch_tables(&model_arch, &mgpu, &hw.kernels_dir)?;

    let vram_gib    = mgpu.nodes.first().map_or(0, |n| n.vram_gib);
    let model_bytes = std::fs::metadata(&target).map_or(0, |m| m.len());
    maybe_print_offload_plan(hw, vram_gib, model_bytes, 0);
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

fn print_engine_header(model_arch: &str, mgpu: &MultiGpuSpec) {
    println!("spite — One Engine, Your Model, Your Card.");
    println!("──────────────────────────────────────────");
    println!("model arch   : {model_arch}");
    if mgpu.nodes.len() == 1 {
        let n = &mgpu.nodes[0];
        let vram_s = if n.vram_gib > 0 { format!(" ({} GiB)", n.vram_gib) } else { String::new() };
        println!("card         : {}{vram_s}", n.card_id);
        println!("gpu arch     : {}", n.gpu_arch);
    } else {
        mgpu.print_summary();
    }
    println!("kernels      :");
}

fn print_dispatch_tables(
    model_arch:  &str,
    mgpu:        &MultiGpuSpec,
    kernels_dir: &std::path::Path,
) -> Result<()> {
    let tables = mgpu.build_tables(model_arch, kernels_dir);
    for (i, result) in tables.into_iter().enumerate() {
        let table = result.context("kernel dispatch build failed")?;
        if mgpu.nodes.len() > 1 {
            println!("  [gpu {}] {}:", i, mgpu.nodes[i].card_id);
        }
        table.print_sources();
    }
    Ok(())
}

/// If offload flags are set, plan the tier placement and print a summary.
///
/// `vram_gib` is from the primary GPU node (0 for unified memory).
/// `model_bytes` is the total GGUF file size — a rough weight estimate.
fn maybe_print_offload_plan(hw: &HardwareArgs, vram_gib: u32, model_bytes: u64, n_layers: usize) {
    if !hw.offload_ram && !hw.offload_disk {
        return;
    }
    const GIB: u64 = 1 << 30;

    // Conservative RAM estimate: total installed memory from /proc/meminfo.
    let ram_bytes = read_total_ram_bytes().unwrap_or(16 * GIB);

    // Budget: if disk offload is disabled, cap RAM budget to ram_bytes so
    // the planner won't place anything in the disk tier.
    let ram_budget = if hw.offload_disk { u64::MAX } else { ram_bytes };

    let bytes_per_layer = if n_layers > 0 { model_bytes / n_layers as u64 } else { model_bytes };
    let cfg = OffloadConfig {
        vram_reserved_bytes: hw.vram_reserve_gib as u64 * GIB,
        ram_budget_bytes:    ram_budget,
        ..Default::default()
    };
    match TieredPlacement::plan(vram_gib as u64 * GIB, ram_bytes, bytes_per_layer, n_layers, &cfg) {
        Ok(plan) => {
            println!();
            plan.print_summary(vram_gib, bytes_per_layer);
        }
        Err(e) => eprintln!("offload plan error: {e}"),
    }
}

fn read_total_ram_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}
