//! spite — One Engine, Your Model, Your Card.
//!
//! Quick start (zero tuning flags required):
//!   spite run   -m Qwen/Qwen3-27B  --card RTX_5090  -p "Hello"
//!   spite serve -m Qwen/Qwen3-27B  --card RTX_5090
//!   spite pull  Qwen/Qwen3-27B     --quant Q4_K_M
//!
//! Model features (opt-in):
//!   spite run   -m DeepSeek-V3 --card RTX_5090 --mtp -p "Hello"
//!   spite run   -m Qwen3-27B   --card RTX_5090 --dflash2 -p "Hello"
//!   spite run   -m Gemma3-27B  --card RTX_5090 --vision -p "Describe"
//!
//! Offload (run any model on any GPU):
//!   spite run   -m Llama-3-70B --card RTX_2080 --offload-ram -p "Hello"
//!   spite run   -m Llama-3-70B --card RTX_2080 --offload-disk -p "Hello"
//!
//! Multi-GPU (pipeline parallelism):
//!   spite run   -m Qwen3-27B --card RTX_5070,RTX_3090 -p "Hello"
//!   spite run   -m Qwen3-27B --card RTX_4090,RX_7900_XTX -p "Hello"

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use spite_dispatch::{
    normalize_card_name,
    MultiGpuSpec, detect_gpu_arch,
};
use spite_loader::GgufModel;
use spite_kvcache::KvQuantConfig;
use spite_offload::{OffloadConfig, TieredPlacement};

// ── Hardware args ──────────────────────────────────────────────────────────

/// Which GPU(s) to target.
///
/// Single GPU:
///   --card RTX_5090
///
/// Multiple GPUs (pipeline parallelism):
///   --card RTX_5070,RTX_3090          same-vendor mixed-arch (auto layer split)
///   --card RTX_4090,RX_7900_XTX       cross-vendor (activations via host RAM)
///
/// When no --card is given, spite reads $SPITE_CARD or tries auto-detection.
#[derive(Args, Clone)]
struct HardwareArgs {
    /// GPU card(s). Comma-separated for multi-GPU pipeline parallelism.
    /// Examples: RTX_5090  |  RTX_5070,RTX_3090  |  RTX_4090,RX_7900_XTX
    #[arg(
        long = "card",
        env  = "SPITE_CARD",
        value_name = "CARD[,CARD…]",
        value_delimiter = ',',
    )]
    cards: Vec<String>,

    /// Spill weight layers that don't fit in VRAM into system RAM.
    /// Allows running larger models than your VRAM alone can hold.
    #[arg(long = "offload-ram", env = "SPITE_OFFLOAD_RAM")]
    offload_ram: bool,

    /// Stream weight layers from disk via mmap when neither VRAM nor RAM fits.
    /// Implies --offload-ram.  Works with any NVMe/SSD/HDD (~7 GB/s).
    #[arg(long = "offload-disk", env = "SPITE_OFFLOAD_DISK")]
    offload_disk: bool,

    /// KV cache quantization.  Default: f16 — no change needed unless VRAM is tight.
    ///
    /// Tiers (high → low quality):
    ///   f16    2.0 bpe   full quality (default)
    ///   q8     1.1 bpe   imperceptible quality loss
    ///   q5_1   0.75 bpe  good balance
    ///   q4     0.56 bpe  mild loss at long context
    ///
    /// Asymmetric K,V (K is more attention-sensitive):
    ///   --kv-quant q8,q5_1   K at q8, V at q5_1
    ///   --kv-quant q8,q4     K at q8, V at q4
    ///
    /// The engine degrades automatically when the KV budget fills.
    #[arg(
        long = "kv-quant",
        env  = "SPITE_KV_QUANT",
        value_name = "TYPE[,TYPE]",
    )]
    kv_quant: Option<String>,

    // ── Advanced (hidden from --help; still usable) ────────────────────────

    /// [advanced] Override GPU arch detection (e.g. sm_89, rdna3, metal).
    /// Ignored when --card is given.
    #[arg(long = "gpu-arch", env = "SPITE_GPU_ARCH", value_name = "ARCH", hide = true)]
    gpu_arch: Option<String>,

    /// [advanced] Manual layer counts per GPU (comma-separated).
    /// Default: proportional to each GPU's VRAM.
    /// E.g. --layer-split 20,12 assigns ≈20/32 layers to the first GPU.
    #[arg(long = "layer-split", value_name = "N[,N…]", value_delimiter = ',', hide = true)]
    layer_split: Vec<u32>,

    /// [advanced] Directory containing compiled kernel .so files.
    #[arg(
        long = "kernels-dir",
        env  = "SPITE_KERNELS_DIR",
        default_value = "kernels",
        value_name = "DIR",
        hide = true,
    )]
    kernels_dir: PathBuf,

    /// [advanced] VRAM to reserve for KV cache and activations (default: 2 GiB).
    /// Only relevant when --offload-ram or --offload-disk is set.
    #[arg(
        long = "vram-reserve-gib",
        env  = "SPITE_VRAM_RESERVE_GIB",
        default_value_t = 2,
        value_name = "GIB",
        hide = true,
    )]
    vram_reserve_gib: u32,
}

// ── Model feature args ─────────────────────────────────────────────────────

/// Optional model/kernel features.  All are off by default.
/// Compile spite with the matching feature for these to take effect:
///   cargo xtask compile -m Model --card CARD --mtp --dflash2
#[derive(Args, Clone)]
struct FeatureArgs {
    /// Multi-Token Prediction: generate several tokens per forward pass.
    /// Requires a model with MTP heads (DeepSeek-V3, Medusa variants).
    /// Compile with: cargo xtask compile --mtp
    #[arg(long = "mtp", env = "SPITE_MTP")]
    mtp: bool,

    /// FlashAttention: fused attention kernel, reduces memory bandwidth.
    /// Supported on most cards.  Use --dflash2 for the faster v2 variant.
    /// Compile with: cargo xtask compile --dflash
    #[arg(long = "dflash", env = "SPITE_DFLASH")]
    dflash: bool,

    /// FlashAttention-2: higher-performance fused attention (Dao-AI-Lab).
    /// Requires sm_80+ (A100/H100/RTX 30+) or cdna3 (MI300X).
    /// Compile with: cargo xtask compile --dflash2
    #[arg(long = "dflash2", env = "SPITE_DFLASH2")]
    dflash2: bool,

    /// Vision encoder: enable multimodal image/video input.
    /// Requires a vision-capable model (Gemma 3, LLaVA, Qwen-VL…).
    /// Compile with: cargo xtask compile --vision
    #[arg(long = "vision", env = "SPITE_VISION")]
    vision: bool,
}

// ── Model args ─────────────────────────────────────────────────────────────

/// Which model(s) to load.
#[derive(Args, Clone)]
struct ModelArgs {
    /// Model name or path.
    ///
    /// "Org/Name"            look in $SPITE_MODELS_DIR (default: ~/.spite/models)
    /// "/abs/path/file.gguf" direct path
    /// "target,draft"        two models for speculative decoding
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
    long_about = "One Engine, Your Model, Your Card.\n\n\
                  spite auto-tunes for your GPU and model — no -ngl, no batch-size \
                  knobs, no dozens of flags.\n\
                  Pass --card and -m and you're running at peak performance.",
    version,
    propagate_version = true,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate tokens from a single prompt and print the output.
    ///
    /// Examples:
    ///   spite run -m Qwen/Qwen3-27B --card RTX_5090 -p "Hello"
    ///   spite run -m DeepSeek-V3    --card RTX_5090 --mtp -p "Hello"
    ///   spite run -m Llama-3-70B   --card RTX_2080 --offload-ram -p "Hello"
    Run {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        hw: HardwareArgs,
        #[command(flatten)]
        feat: FeatureArgs,

        /// Prompt text.
        #[arg(short = 'p', long, value_name = "TEXT")]
        prompt: String,

        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 512, value_name = "N")]
        max_tokens: usize,

        /// Sampling temperature.  0 = greedy (deterministic), 1 = default.
        #[arg(long, default_value_t = 1.0, value_name = "T")]
        temperature: f32,

        /// Context window size.  Default: model's native maximum.
        /// Reduce to save KV cache memory (e.g. --ctx 4096).
        #[arg(long, value_name = "N")]
        ctx: Option<usize>,
    },

    /// Start an OpenAI-compatible HTTP inference server.
    ///
    /// Compatible with any client that speaks the OpenAI Chat Completions API
    /// (curl, Python openai SDK, Open WebUI, etc.).
    ///
    /// Examples:
    ///   spite serve -m Qwen/Qwen3-27B --card RTX_5090
    ///   spite serve -m Qwen/Qwen3-27B --card RTX_5090 --port 11434
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        hw: HardwareArgs,
        #[command(flatten)]
        feat: FeatureArgs,

        /// Bind address.
        #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
        host: String,

        /// Bind port.
        #[arg(short, long, default_value_t = 8080, value_name = "PORT")]
        port: u16,

        /// Context window size.  Default: model's native maximum.
        #[arg(long, value_name = "N")]
        ctx: Option<usize>,
    },

    /// Show which kernel wins each op for this model + card.
    ///
    /// Useful for verifying that the right GPU-specific kernels are loaded
    /// and for diagnosing dispatch issues.
    Dispatch {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        hw: HardwareArgs,
        #[command(flatten)]
        feat: FeatureArgs,
    },

    /// Download a model from HuggingFace Hub into $SPITE_MODELS_DIR.
    ///
    /// Models are saved to ~/.spite/models (override with $SPITE_MODELS_DIR).
    ///
    /// Examples:
    ///   spite pull Qwen/Qwen3-27B
    ///   spite pull meta-llama/Llama-3.1-70B --quant Q4_K_M
    Pull {
        /// Model name in HuggingFace "Org/Name" format.
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
        Cmd::Run { model, hw, feat, prompt, max_tokens, temperature, ctx } => {
            cmd_run(&model, &hw, &feat, &prompt, max_tokens, temperature, ctx)
        }
        Cmd::Serve { model, hw, feat, host, port, ctx } => {
            cmd_serve(&model, &hw, &feat, &host, port, ctx)
        }
        Cmd::Dispatch { model, hw, feat } => {
            cmd_dispatch(&model, &hw, &feat)
        }
        Cmd::Pull { model, quant } => {
            cmd_pull(&model, &quant)
        }
    }
}

// ── Hardware resolution ────────────────────────────────────────────────────

fn resolve_hardware(hw: &HardwareArgs) -> MultiGpuSpec {
    if hw.cards.is_empty() {
        let arch = hw.gpu_arch.clone().unwrap_or_else(detect_gpu_arch);
        let raw  = std::env::var("SPITE_CARD").unwrap_or_default();
        let card = if raw.is_empty() { arch.clone() } else { normalize_card_name(&raw) };
        MultiGpuSpec::from_cards(&[card.as_str()])
    } else {
        let refs: Vec<&str> = hw.cards.iter().map(String::as_str).collect();
        MultiGpuSpec::from_cards(&refs)
    }
}

// ── Model path resolution ──────────────────────────────────────────────────

fn models_dir() -> PathBuf {
    std::env::var("SPITE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".spite").join("models"))
                .unwrap_or_else(|_| PathBuf::from(".spite/models"))
        })
}

fn resolve_model_path(name: &str) -> Result<PathBuf> {
    let p = Path::new(name);
    if p.is_absolute() || name.starts_with("./") || name.starts_with("../") {
        if p.exists() { return Ok(p.to_owned()); }
        bail!("model path not found: {name}");
    }
    let dir = models_dir();
    if !dir.exists() {
        bail!(
            "model '{name}' not found as a path; \
             $SPITE_MODELS_DIR / ~/.spite/models does not exist.\n\
             Run `spite pull {name}` to download it, or set SPITE_MODELS_DIR."
        );
    }
    let base = name.replace('/', "--");
    let candidates = vec![
        dir.join(format!("{base}.gguf")),
        dir.join(name).with_extension("gguf"),
        dir.join(name.split('/').last().unwrap_or(name)).with_extension("gguf"),
    ];
    for c in &candidates {
        if c.exists() { return Ok(c.clone()); }
    }
    let short = name.split('/').last().unwrap_or(name).to_lowercase();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().map_or(false, |e| e == "gguf") {
                let stem = p.file_stem().unwrap_or_default()
                    .to_string_lossy().to_lowercase();
                if stem.contains(&short) { return Ok(p); }
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
    feat:        &FeatureArgs,
    prompt:      &str,
    max_tokens:  usize,
    temperature: f32,
    ctx:         Option<usize>,
) -> Result<()> {
    let (target, draft) = resolve_models(&model_args.model)?;
    let mgpu            = resolve_hardware(hw);
    let target_gguf     = GgufModel::open(&target)?;
    let model_arch      = target_gguf.arch().to_owned();

    let kv_cfg = resolve_kv_quant(hw);

    print_engine_header(&model_arch, &mgpu);
    print_feature_summary(feat);
    print_kv_quant_summary(&kv_cfg);

    if let Some(ref d) = draft {
        let d_gguf = GgufModel::open(d)?;
        println!("draft model  : {} [speculative decoding]", d_gguf.arch());
    }

    print_dispatch_tables(&model_arch, &mgpu, &hw.kernels_dir)?;

    let vram_gib    = mgpu.nodes.first().map_or(0, |n| n.vram_gib);
    let model_bytes = std::fs::metadata(&target).map_or(0, |m| m.len());
    maybe_print_offload_plan(hw, vram_gib, model_bytes, 0);

    if let Some(n) = ctx {
        println!("context      : {n} tokens");
    }
    println!("\nprompt       : {prompt}");
    println!("max tokens   : {max_tokens}");
    println!("temperature  : {temperature}");
    println!("\n(inference not yet implemented — contribute it!)");
    Ok(())
}

fn cmd_serve(
    model_args: &ModelArgs,
    hw:         &HardwareArgs,
    feat:       &FeatureArgs,
    host:       &str,
    port:       u16,
    ctx:        Option<usize>,
) -> Result<()> {
    let (target, draft) = resolve_models(&model_args.model)?;
    let mgpu            = resolve_hardware(hw);
    let target_gguf     = GgufModel::open(&target)?;
    let model_arch      = target_gguf.arch().to_owned();
    let kv_cfg          = resolve_kv_quant(hw);

    print_engine_header(&model_arch, &mgpu);
    print_feature_summary(feat);
    print_kv_quant_summary(&kv_cfg);

    if let Some(ref d) = draft {
        let d_gguf = GgufModel::open(d)?;
        println!("draft model  : {} [speculative decoding]", d_gguf.arch());
    }

    print_dispatch_tables(&model_arch, &mgpu, &hw.kernels_dir)?;

    let vram_gib    = mgpu.nodes.first().map_or(0, |n| n.vram_gib);
    let model_bytes = std::fs::metadata(&target).map_or(0, |m| m.len());
    maybe_print_offload_plan(hw, vram_gib, model_bytes, 0);

    if let Some(n) = ctx {
        println!("context      : {n} tokens");
    }
    println!("\nlistening on : http://{host}:{port}");
    println!("(server loop not yet implemented — contribute it!)");
    Ok(())
}

fn cmd_dispatch(model_args: &ModelArgs, hw: &HardwareArgs, feat: &FeatureArgs) -> Result<()> {
    let (target, _) = resolve_models(&model_args.model)?;
    let mgpu        = resolve_hardware(hw);
    let gguf        = GgufModel::open(&target)?;
    let model_arch  = gguf.arch().to_owned();
    let kv_cfg      = resolve_kv_quant(hw);

    print_engine_header(&model_arch, &mgpu);
    print_feature_summary(feat);
    print_kv_quant_summary(&kv_cfg);
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

// ── Print helpers ──────────────────────────────────────────────────────────

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

/// Parse --kv-quant and print it if non-default.
fn resolve_kv_quant(hw: &HardwareArgs) -> KvQuantConfig {
    let Some(ref s) = hw.kv_quant else {
        return KvQuantConfig::default();
    };
    match KvQuantConfig::from_str(s) {
        Ok(cfg) => cfg,
        Err(e)  => { eprintln!("warning: {e} — using f16 default"); KvQuantConfig::default() }
    }
}

fn print_kv_quant_summary(cfg: &KvQuantConfig) {
    if cfg.is_default() { return; }
    if cfg.key == cfg.val {
        println!("kv cache     : {}  (K+V)", cfg.key);
    } else {
        println!("kv cache     : K={}  V={}  (asymmetric)", cfg.key, cfg.val);
    }
}

fn print_feature_summary(feat: &FeatureArgs) {
    let mut active: Vec<&str> = Vec::new();
    if feat.mtp     { active.push("mtp"); }
    if feat.dflash2 { active.push("dflash2"); }
    else if feat.dflash { active.push("dflash"); }
    if feat.vision  { active.push("vision"); }
    if !active.is_empty() {
        println!("features     : {}", active.join(", "));
    }
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

fn maybe_print_offload_plan(hw: &HardwareArgs, vram_gib: u32, model_bytes: u64, n_layers: usize) {
    if !hw.offload_ram && !hw.offload_disk {
        return;
    }
    const GIB: u64 = 1 << 30;
    let ram_bytes  = read_total_ram_bytes().unwrap_or(16 * GIB);
    let ram_budget = if hw.offload_disk { u64::MAX } else { ram_bytes };
    let bytes_per_layer = if n_layers > 0 { model_bytes / n_layers as u64 } else { model_bytes };
    let cfg = OffloadConfig {
        vram_reserved_bytes: hw.vram_reserve_gib as u64 * GIB,
        ram_budget_bytes:    ram_budget,
        ..Default::default()
    };
    match TieredPlacement::plan(vram_gib as u64 * GIB, ram_bytes, bytes_per_layer, n_layers, &cfg) {
        Ok(plan)  => { println!(); plan.print_summary(vram_gib, bytes_per_layer); }
        Err(e)    => eprintln!("offload plan error: {e}"),
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
