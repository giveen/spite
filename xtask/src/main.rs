//! spite build tool.
//!
//!   cargo xtask compile -m Qwen/Qwen3-27B,Google/gemma4 --card RTX_5090
//!
//! Translates model names and a card name into a `cargo build` invocation
//! with the exact set of Cargo features needed — no dead code for models or
//! GPU backends you didn't ask for.

use std::process::{Command, ExitCode};

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use spite_dispatch::{card_spec, normalize_card_name};

// ── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "xtask", about = "spite build tasks")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compile spite for a specific card (or cards) and model set.
    ///
    /// Single GPU:
    ///   cargo xtask compile -m Qwen/Qwen3-27B --card RTX_5090
    ///   cargo xtask compile -m meta-llama/Llama-3-70B --card MI300X
    ///   cargo xtask compile -m meta-llama/Llama-3-8B --card M4_Max
    ///
    /// Multiple GPUs (pipeline parallelism):
    ///   cargo xtask compile -m Qwen/Qwen3-27B --card RTX_5070,RTX_3090
    ///   cargo xtask compile -m Qwen/Qwen3-27B --card RTX_4090,RX_7900_XTX
    Compile {
        /// Model(s) to include. One name or "target,draft" for speculative.
        /// Format: "Org/Name" (HuggingFace style) or just "Name".
        #[arg(short = 'm', long = "model", value_name = "MODEL[,DRAFT]")]
        model: String,

        /// GPU card(s). Comma-separated for multi-GPU pipeline parallelism.
        /// Mixed vendors compile both GPU backends into the same binary.
        #[arg(long = "card", value_name = "CARD[,CARD…]", value_delimiter = ',')]
        cards: Vec<String>,

        /// Build profile: release (default) or dev.
        #[arg(long, default_value = "release")]
        profile: String,

        /// Pass extra Cargo features (comma-separated).
        #[arg(long, value_name = "FEAT,...")]
        extra_features: Option<String>,

        /// Print the cargo command without running it.
        #[arg(long)]
        dry_run: bool,
    },

    /// List every card the engine knows about.
    Cards,

    /// List every model family the engine supports.
    Models,
}

// ── Entry ──────────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.cmd {
        Cmd::Compile { model, cards, profile, extra_features, dry_run } => {
            cmd_compile(&model, &cards, &profile, extra_features.as_deref(), dry_run)
        }
        Cmd::Cards  => cmd_cards(),
        Cmd::Models => cmd_models(),
    }
}

// ── compile ────────────────────────────────────────────────────────────────

fn cmd_compile(
    model_spec:     &str,
    cards_raw:      &[String],
    profile:        &str,
    extra_features: Option<&str>,
    dry_run:        bool,
) -> Result<()> {
    if cards_raw.is_empty() {
        bail!("--card is required");
    }

    // ── Resolve cards ────────────────────────────────────────────────────
    let mut gpu_feats: Vec<&'static str> = Vec::new();
    let mut card_rows: Vec<(String, String, String)> = Vec::new(); // (card_id, arch, label)

    for raw in cards_raw {
        let card_id  = normalize_card_name(raw);
        let spec     = card_spec(&card_id);
        let gpu_feat = gpu_arch_to_feature(spec.gpu_arch);
        let vram_str = if spec.vram_gib > 0 {
            format!("{} GiB VRAM", spec.vram_gib)
        } else {
            "shared VRAM".into()
        };
        let label = format!("{} ({}{})",
            spec.gpu_arch,
            backend_label(spec.gpu_arch),
            if spec.vram_gib > 0 { format!(" · {vram_str}") } else { String::new() }
        );
        if !gpu_feat.is_empty() && !gpu_feats.contains(&gpu_feat) {
            gpu_feats.push(gpu_feat);
        }
        card_rows.push((card_id, spec.gpu_arch.to_owned(), label));
    }

    // Detect cross-vendor pair
    let vendors: Vec<&str> = card_rows.iter().map(|(_, arch, _)| vendor_label(arch)).collect();
    let cross_vendor = vendors.windows(2).any(|w| w[0] != w[1]);

    // ── Resolve models ───────────────────────────────────────────────────
    let model_names: Vec<&str> = model_spec.split(',').map(str::trim).collect();
    let mut model_feats: Vec<&'static str> = Vec::new();
    for name in &model_names {
        let f = model_to_feature(name)?;
        if !model_feats.contains(&f) {
            model_feats.push(f);
        }
    }

    // ── Build feature string ─────────────────────────────────────────────
    let mut features: Vec<&str> = Vec::new();
    features.extend_from_slice(&gpu_feats);
    features.extend_from_slice(&model_feats);
    if let Some(extra) = extra_features {
        for f in extra.split(',') { features.push(f.trim()); }
    }
    let feature_str = features.join(",");

    // ── Print build plan ─────────────────────────────────────────────────
    println!("spite — compile for your card and your models");
    println!("─────────────────────────────────────────────");

    if card_rows.len() == 1 {
        let (id, _, label) = &card_rows[0];
        println!("card     : {id}  →  {label}");
    } else {
        println!("cards    : (pipeline parallelism — {} GPUs)", card_rows.len());
        for (i, (id, _, label)) in card_rows.iter().enumerate() {
            println!("  [{i}] {id:<20} → {label}");
        }
        if cross_vendor {
            println!("  ⚠  cross-vendor: activations transit host RAM between unlike GPUs");
        }
    }

    println!("models   :");
    for (name, feat) in model_names.iter().zip(model_feats.iter()) {
        println!("  {name:<40} → {feat}");
    }
    println!("features : {feature_str}");
    println!();

    // ── Build the cargo command ───────────────────────────────────────────
    let mut args: Vec<String> = vec!["build".into(), "--package".into(), "spite".into()];

    if profile == "release" {
        args.push("--release".into());
    } else if profile != "dev" {
        args.extend(["--profile".into(), profile.into()]);
    }

    if !feature_str.is_empty() {
        args.extend(["--features".into(), feature_str.clone()]);
    }

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    println!("running  : {cargo} {}", args.join(" "));

    if dry_run {
        println!("\n(dry run — not executing)");
        return Ok(());
    }

    println!();
    let status = Command::new(&cargo).args(&args).status()?;
    if !status.success() {
        bail!("cargo build failed");
    }
    Ok(())
}

// ── cards / models info ────────────────────────────────────────────────────

fn cmd_cards() -> Result<()> {
    println!("known cards (use these with --card):\n");
    for (card, arch, vram) in KNOWN_CARDS {
        let vram_s = if *vram > 0 { format!("{vram} GiB") } else { "shared".into() };
        println!("  {card:<20}  {arch:<12}  {vram_s}");
    }
    Ok(())
}

fn cmd_models() -> Result<()> {
    println!("supported model families:\n");
    for (family, feat, examples) in KNOWN_MODELS {
        println!("  {feat:<20}  ({family})");
        for ex in *examples {
            println!("    • {ex}");
        }
        println!();
    }
    Ok(())
}

// ── Mapping tables ─────────────────────────────────────────────────────────

/// Map a normalised `gpu_arch` string to the Cargo feature name.
fn gpu_arch_to_feature(gpu_arch: &str) -> &'static str {
    match gpu_arch {
        "sm_120" => "cuda-sm120",
        "sm_100" => "cuda-sm100",
        "sm_90"  => "cuda-sm90",
        "sm_89"  => "cuda-sm89",
        "sm_86"  => "cuda-sm86",
        "sm_80"  => "cuda-sm80",
        "sm_75"  => "cuda-sm75",
        "sm_70"  => "cuda-sm70",
        "rdna4"  => "rocm-rdna4",
        "rdna3"  => "rocm-rdna3",
        "rdna2"  => "rocm-rdna2",
        "cdna3"  => "rocm-cdna3",
        "cdna2"  => "rocm-cdna2",
        "metal"  => "metal",
        "xe2"    => "vulkan-xe2",
        "xe_hpg" => "vulkan-xe_hpg",
        _        => "",  // generic — no GPU feature, scalar fallback only
    }
}

/// Map a model name (HuggingFace style or plain) to the Cargo feature name.
fn model_to_feature(name: &str) -> Result<&'static str> {
    let lower = name.to_lowercase();
    // Try org prefix first, then substring match.
    if lower.starts_with("qwen") || lower.contains("/qwen")        { return Ok("model-qwen"); }
    if lower.contains("gemma")                                      { return Ok("model-gemma"); }
    if lower.contains("llama")                                      { return Ok("model-llama"); }
    if lower.starts_with("mistral") || lower.contains("/mistral")   { return Ok("model-mistral"); }
    if lower.contains("deepseek")                                   { return Ok("model-deepseek"); }
    if lower.contains("phi")                                        { return Ok("model-phi"); }
    if lower.contains("falcon")                                     { return Ok("model-falcon"); }
    if lower.contains("gpt2") || lower.starts_with("openai/gpt")   { return Ok("model-gpt2"); }
    if lower.contains("gemma") || lower.starts_with("google/")     { return Ok("model-gemma"); }
    if lower.starts_with("meta") || lower.starts_with("meta-llama") { return Ok("model-llama"); }
    bail!(
        "unknown model family for '{name}'.\n\
         Run `cargo xtask models` to see supported families, \
         or add a new entry to xtask/src/main.rs."
    )
}

fn backend_label(gpu_arch: &str) -> &'static str {
    if gpu_arch.starts_with("sm_")  { return "NVIDIA CUDA"; }
    if gpu_arch.starts_with("rdna") || gpu_arch.starts_with("cdna") { return "AMD ROCm"; }
    if gpu_arch == "metal"           { return "Apple Metal"; }
    if gpu_arch.starts_with("xe")   { return "Intel oneAPI"; }
    "generic / CPU scalar"
}

fn vendor_label(gpu_arch: &str) -> &'static str {
    if gpu_arch.starts_with("sm_")                                   { return "nvidia"; }
    if gpu_arch.starts_with("rdna") || gpu_arch.starts_with("cdna") { return "amd"; }
    if gpu_arch == "metal"                                           { return "apple"; }
    if gpu_arch.starts_with("xe")                                    { return "intel"; }
    "generic"
}

// ── Static info tables ─────────────────────────────────────────────────────

static KNOWN_CARDS: &[(&str, &str, u32)] = &[
    ("rtx_5090",  "sm_120",  32), ("rtx_5080",  "sm_120",  16),
    ("rtx_4090",  "sm_89",   24), ("rtx_4080",  "sm_89",   16),
    ("rtx_3090",  "sm_86",   24), ("rtx_3080",  "sm_86",   10),
    ("h200",      "sm_90",  141), ("h100",      "sm_90",   80),
    ("a100",      "sm_80",   80), ("a6000",     "sm_86",   48),
    ("rx_9900_xtx","rdna4",  32), ("rx_9900_xt","rdna4",   32),
    ("rx_7900_xtx","rdna3",  24), ("rx_7900_xt","rdna3",   20),
    ("rx_6900_xt","rdna2",   16), ("rx_6800_xt","rdna2",   16),
    ("mi350x",    "cdna3",  288), ("mi300x",    "cdna3",  192),
    ("mi250x",    "cdna2",  128),
    ("m4_max",    "metal",    0), ("m4_pro",    "metal",   0),
    ("m4",        "metal",    0), ("m3_max",    "metal",   0),
    ("m3",        "metal",    0), ("m2_max",    "metal",   0),
    ("m2",        "metal",    0), ("m1_max",    "metal",   0),
    ("m1",        "metal",    0),
    ("b770",      "xe2",     16), ("b580",      "xe2",     12),
    ("a770",      "xe_hpg",  16), ("a750",      "xe_hpg",   8),
];

static KNOWN_MODELS: &[(&str, &str, &[&str])] = &[
    ("LLaMA / LLaMA 2 / LLaMA 3",  "model-llama",
     &["meta-llama/Llama-3.1-70B", "meta-llama/Llama-3-8B"]),
    ("Qwen 2 / Qwen 3",            "model-qwen",
     &["Qwen/Qwen3-27B", "Qwen/Qwen3-1.7B", "Qwen/Qwen2.5-72B"]),
    ("Gemma / Gemma 2 / Gemma 3",  "model-gemma",
     &["Google/gemma-3-27b", "Google/gemma-2-9b"]),
    ("Mistral / Mixtral",          "model-mistral",
     &["mistralai/Mistral-7B-v0.3", "mistralai/Mixtral-8x7B"]),
    ("DeepSeek V2 / V3 / R1",      "model-deepseek",
     &["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"]),
    ("Phi-3 / Phi-4",              "model-phi",
     &["microsoft/Phi-4", "microsoft/Phi-3.5-mini-instruct"]),
    ("Falcon / Falcon H1",         "model-falcon",
     &["tiiuae/Falcon3-10B-Instruct", "tiiuae/falcon-40b"]),
    ("GPT-2 (legacy)",             "model-gpt2",
     &["openai/gpt2", "openai/gpt2-xl"]),
];
