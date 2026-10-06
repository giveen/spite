//! spite — One Engine, Your Model, Your Card.
//!
//! Quick start (zero tuning flags required):
//!   spite run   -m Qwen/Qwen3.5-27B  --card RTX_5090  -p "Hello"
//!   spite serve -m Qwen/Qwen3.5-27B  --card RTX_5090
//!   spite pull  Qwen/Qwen3.5-27B     --quant Q4_K_M
//!
//! Model features (opt-in):
//!   spite run   -m DeepSeek-V4 --card RTX_5090 --mtp -p "Hello"
//!   spite run   -m Qwen3.5-27B   --card RTX_5090 --dflash2 -p "Hello"
//!   spite run   -m Gemma4-27B  --card RTX_5090 --vision -p "Describe"
//!
//! Offload (run any model on any GPU):
//!   spite run   -m Llama-4-Maverick --card RTX_2080 --offload-ram -p "Hello"
//!   spite run   -m Llama-4-Maverick --card RTX_2080 --offload-disk -p "Hello"
//!
//! Multi-GPU (pipeline parallelism):
//!   spite run   -m Qwen3.5-27B --card RTX_5070,RTX_3090 -p "Hello"
//!   spite run   -m Qwen3.5-27B --card RTX_4090,RX_7900_XTX -p "Hello"

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use spite_dispatch::{MultiGpuSpec, detect_gpu_arch, normalize_card_name};
use spite_kvcache::KvQuantConfig;
use spite_loader::GgufModel;
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
        env = "SPITE_CARD",
        value_name = "CARD[,CARD…]",
        value_delimiter = ','
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

    /// Override the KV cache quantization starting tier.
    ///
    /// Default: the engine starts at f16 and degrades automatically as VRAM fills
    /// (f16 → q8 → q5_1 → q4).  You only need this flag to pin a lower starting
    /// tier — for example to leave more VRAM headroom at very long context.
    ///
    ///   f16    2.0 bpe   full precision (default start)
    ///   q8     1.1 bpe   imperceptible quality loss
    ///   q5_1   0.75 bpe  good balance
    ///   q4     0.56 bpe  mild loss at long context
    ///
    /// Asymmetric (K is more sensitive than V):
    ///   --kv-quant q8,q5_1   K at q8, V at q5_1
    #[arg(long = "kv-quant", env = "SPITE_KV_QUANT", value_name = "TYPE[,TYPE]")]
    kv_quant: Option<String>,

    // ── Advanced (hidden from --help; still usable) ────────────────────────
    /// [advanced] Override GPU arch detection (e.g. sm_89, rdna3, metal).
    /// Ignored when --card is given.
    #[arg(
        long = "gpu-arch",
        env = "SPITE_GPU_ARCH",
        value_name = "ARCH",
        hide = true
    )]
    gpu_arch: Option<String>,

    /// [advanced] Manual layer shares per GPU (comma-separated), one per GPU
    /// in --gpus order. Default: stay on the first GPU when the model fits,
    /// else spread by free VRAM.
    /// E.g. --layer-split 20,12 assigns ≈20/32 layers to the first GPU.
    #[arg(
        long = "layer-split",
        value_name = "N[,N…]",
        value_delimiter = ',',
        hide = true
    )]
    layer_split: Vec<u32>,

    /// [advanced] CUDA device ordinals for the pipeline split, in layer order
    /// (default: every visible GPU). A repeated ordinal puts several stages on
    /// one GPU, which exercises the split on a single card.
    #[arg(
        long = "gpus",
        env = "SPITE_GPUS",
        value_name = "ID[,ID…]",
        value_delimiter = ',',
        hide = true
    )]
    gpus: Vec<usize>,

    /// [advanced] Directory containing compiled kernel .so files.
    #[arg(
        long = "kernels-dir",
        env = "SPITE_KERNELS_DIR",
        default_value = "kernels",
        value_name = "DIR",
        hide = true
    )]
    kernels_dir: PathBuf,

    /// [advanced] VRAM to reserve for KV cache and activations (default: 2 GiB).
    /// Only relevant when --offload-ram or --offload-disk is set.
    #[arg(
        long = "vram-reserve-gib",
        env = "SPITE_VRAM_RESERVE_GIB",
        default_value_t = 2,
        value_name = "GIB",
        hide = true
    )]
    vram_reserve_gib: u32,

    /// [advanced] Compute device: auto (CUDA when a GPU kernel resolves,
    /// else CPU), cpu, or cuda (fail if the CUDA path is unavailable).
    #[arg(
        long = "device",
        env = "SPITE_DEVICE",
        value_enum,
        default_value_t = Device::Auto,
        hide = true
    )]
    device: Device,
}

/// Where the forward pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
enum Device {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

/// Device selection resolved from CLI args, passed to `generate`.
pub struct Placement<'a> {
    device: Device,
    kernels_dir: &'a Path,
    gpu_arch: &'a str,
    /// `--gpus`: CUDA ordinals for a pipeline split (empty: all visible).
    gpus: &'a [usize],
    /// `--layer-split`: layer shares per GPU (empty: automatic).
    layer_split: &'a [u32],
}

impl Placement<'_> {
    /// CPU-only placement (tests, hosts without a GPU).
    pub fn cpu() -> Placement<'static> {
        Placement {
            device: Device::Cpu,
            kernels_dir: Path::new("kernels"),
            gpu_arch: "generic",
            gpus: &[],
            layer_split: &[],
        }
    }

    /// Pipeline split for the hybrid decoder on CUDA.
    ///
    /// Without `--gpus`, every visible GPU is a candidate; the decoder only
    /// spreads when the model does not fit on the first one, or when
    /// `--layer-split` asks for it.
    fn hybrid_split(&self) -> spite_models::hybrid::LayerSplit {
        let devices = if self.gpus.is_empty() {
            match spite_gpu::cuda::device_count() {
                Ok(n) if self.layer_split.is_empty() => (0..n).collect(),
                // Explicit shares without --gpus: the first N visible GPUs.
                Ok(n) => (0..n.min(self.layer_split.len())).collect(),
                Err(_) => Vec::new(),
            }
        } else {
            self.gpus.to_vec()
        };
        spite_models::hybrid::LayerSplit {
            devices,
            shares: self.layer_split.to_vec(),
        }
    }
}

// ── Model feature args ─────────────────────────────────────────────────────

/// Optional model/kernel features.  All are off by default.
/// Compile spite with the matching feature for these to take effect:
///   cargo xtask compile -m Model --card CARD --mtp --dflash2
#[derive(Args, Clone)]
struct FeatureArgs {
    /// Multi-Token Prediction: generate several tokens per forward pass.
    /// Requires a model with MTP heads (DeepSeek-V4, Medusa variants).
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
    /// Requires a vision-capable model (Gemma 4, LLaVA, Qwen-VL…).
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
    #[arg(short = 'm', long = "model", value_name = "MODEL[,DRAFT]")]
    model: String,
}

// ── CLI structure ──────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "spite",
    about = "One Engine, Your Model, Your Card.",
    long_about = "One Engine, Your Model, Your Card.\n\n\
                  spite auto-tunes for your GPU and model — no -ngl, no batch-size \
                  knobs, no dozens of flags.\n\
                  Pass --card and -m and you're running at peak performance.",
    version,
    propagate_version = true
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
    ///   spite run -m Qwen/Qwen3.5-27B --card RTX_5090 -p "Hello"
    ///   spite run -m DeepSeek-V4    --card RTX_5090 --mtp -p "Hello"
    ///   spite run -m Llama-4-Maverick   --card RTX_2080 --offload-ram -p "Hello"
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
    ///   spite serve -m Qwen/Qwen3.5-27B --card RTX_5090
    ///   spite serve -m Qwen/Qwen3.5-27B --card RTX_5090 --port 11434
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
    ///   spite pull Qwen/Qwen3.5-27B
    ///   spite pull meta-llama/Llama-4-Maverick --quant Q4_K_M
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
        Cmd::Run {
            model,
            hw,
            feat,
            prompt,
            max_tokens,
            temperature,
            ctx,
        } => cmd_run(&model, &hw, &feat, &prompt, max_tokens, temperature, ctx),
        Cmd::Serve {
            model,
            hw,
            feat,
            host,
            port,
            ctx,
        } => cmd_serve(&model, &hw, &feat, &host, port, ctx),
        Cmd::Dispatch { model, hw, feat } => cmd_dispatch(&model, &hw, &feat),
        Cmd::Pull { model, quant } => cmd_pull(&model, &quant),
    }
}

// ── Hardware resolution ────────────────────────────────────────────────────

fn resolve_hardware(hw: &HardwareArgs) -> MultiGpuSpec {
    let mut spec = if hw.cards.is_empty() {
        let arch = hw.gpu_arch.clone().unwrap_or_else(detect_gpu_arch);
        let raw = std::env::var("SPITE_CARD").unwrap_or_default();
        let card = if raw.is_empty() {
            arch.clone()
        } else {
            normalize_card_name(&raw)
        };
        MultiGpuSpec::from_cards(&[card.as_str()])
    } else {
        let refs: Vec<&str> = hw.cards.iter().map(String::as_str).collect();
        MultiGpuSpec::from_cards(&refs)
    };
    // Card names drop the form factor, so same-vendor defaults to PCIe; promote
    // to NVLink/XGMI only where the runtime probe confirms it.
    spec.probe_links();
    spec
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
        if p.exists() {
            return Ok(p.to_owned());
        }
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
        dir.join(name.split('/').next_back().unwrap_or(name))
            .with_extension("gguf"),
    ];
    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }
    let short = name.split('/').next_back().unwrap_or(name).to_lowercase();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "gguf") {
                let stem = p
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase();
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

fn resolve_models(spec: &str) -> Result<(PathBuf, Option<PathBuf>)> {
    let parts: Vec<&str> = spec.splitn(2, ',').collect();
    let target = resolve_model_path(parts[0].trim())?;
    let draft = parts
        .get(1)
        .map(|s| resolve_model_path(s.trim()))
        .transpose()?;
    Ok((target, draft))
}

// ── Command implementations ────────────────────────────────────────────────

fn cmd_run(
    model_args: &ModelArgs,
    hw: &HardwareArgs,
    feat: &FeatureArgs,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    ctx: Option<usize>,
) -> Result<()> {
    let (target, draft) = resolve_models(&model_args.model)?;
    let mgpu = resolve_hardware(hw);
    let target_gguf = GgufModel::open(&target)?;
    let model_arch = target_gguf.arch().to_owned();

    let kv_cfg = resolve_kv_quant(hw);

    print_engine_header(&model_arch, &mgpu);
    print_feature_summary(feat);
    print_kv_quant_summary(&kv_cfg);

    if let Some(ref d) = draft {
        let d_gguf = GgufModel::open(d)?;
        println!("draft model  : {} [speculative decoding]", d_gguf.arch());
    }

    print_dispatch_tables(&model_arch, &mgpu, &hw.kernels_dir)?;

    let vram_gib = mgpu.nodes.first().map_or(0, |n| n.vram_gib);
    let model_bytes = std::fs::metadata(&target).map_or(0, |m| m.len());
    maybe_print_offload_plan(hw, vram_gib, model_bytes, 0);

    if let Some(n) = ctx {
        println!("context      : {n} tokens");
    }
    println!("\nprompt       : {prompt}");
    println!("max tokens   : {max_tokens}");
    println!("temperature  : {temperature}");

    let gpu_arch = mgpu
        .nodes
        .first()
        .map(|n| n.gpu_arch.clone())
        .unwrap_or_default();
    let place = Placement {
        device: hw.device,
        kernels_dir: &hw.kernels_dir,
        gpu_arch: &gpu_arch,
        gpus: &hw.gpus,
        layer_split: &hw.layer_split,
    };
    let text = generate(
        &target_gguf,
        prompt,
        max_tokens,
        temperature,
        ctx,
        &place,
        &kv_cfg,
    )?;
    println!("\n{text}");
    Ok(())
}

/// Shared generate path for `run` (server reuses the same crates).
/// Tokenize → prefill → decode loop → detokenize.
pub fn generate(
    gguf: &GgufModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    ctx_len: Option<usize>,
    place: &Placement,
    kv_cfg: &KvQuantConfig,
) -> Result<String> {
    use spite_executor::{Executor, ExecutorConfig};
    use spite_tokenizer::Tokenizer;

    let mut exec_cfg = ExecutorConfig::default();
    if let Some(n) = ctx_len {
        exec_cfg.ctx_len = n;
    }
    // Carried to the model by `Executor::load_model`, which enables VBR.
    exec_cfg.kv_quant = kv_cfg.clone();

    let t_load = std::time::Instant::now();
    let model = build_model(gguf, exec_cfg.ctx_len, place, kv_cfg)?;
    eprintln!("load time    : {:.2} s", t_load.elapsed().as_secs_f64());

    let tokenizer = Tokenizer::from_gguf(gguf)?;
    let ids = tokenizer.encode(prompt, tokenizer.add_bos())?;

    let mut exec = Executor::new(exec_cfg);
    exec.load_model(model);

    let t_gen = std::time::Instant::now();
    let pieces = exec.generate(
        &tokenizer,
        &ids,
        max_tokens,
        temperature,
        0x1234_5678_9abc_def0,
    )?;
    let secs = t_gen.elapsed().as_secs_f64();
    eprintln!(
        "generate     : {} prompt + {} new tokens in {secs:.2} s ({:.2} tok/s)",
        ids.len(),
        pieces.len(),
        (ids.len() + pieces.len()) as f64 / secs.max(1e-9)
    );
    Ok(pieces.into_iter().map(|(_, s)| s).collect())
}

/// Build the model on the requested device.
///
/// CUDA: weights are uploaded to VRAM as stored (no host dequant) and every
/// op runs through the resolved `sm_*` kernel. CPU: the existing dequantized
/// F32 path.
fn build_model(
    gguf: &GgufModel,
    ctx_len: usize,
    place: &Placement,
    kv_cfg: &KvQuantConfig,
) -> Result<Box<dyn spite_models::ModelArch>> {
    use spite_models::gpu_dense::GpuDense;
    use spite_models::{ArchRegistry, ModelConfig};

    let hp = spite_loader::config::ModelHyperparams::from_gguf(gguf);
    let mut cfg = ModelConfig::from(hp);
    // The effective context is what VBR sizes its degradation thresholds to;
    // never exceed the context the executor will actually allow.
    cfg.max_seq_len = cfg.max_seq_len.min(ctx_len).max(1);

    if spite_models::hybrid::is_hybrid(&cfg) {
        use spite_models::hybrid::HybridDecoder;
        // Device kernels first (unless --device cpu), then the generic CPU kernels;
        // both run the same op-driven forward.
        let want_gpu = place.device != Device::Cpu;
        let resolved = if want_gpu {
            HybridDecoder::resolve_table(&cfg.arch, place.gpu_arch, place.kernels_dir, true)
        } else {
            None
        };
        if resolved.is_none() && place.device == Device::Cuda {
            bail!(
                "--device cuda: no kernel provides every hybrid op (rms_norm, attention_ex, linear_attn, ffn, matmul) \
                 for arch '{}' on '{}' under {}",
                cfg.arch,
                place.gpu_arch,
                place.kernels_dir.display()
            );
        }
        let resolved = resolved.or_else(|| {
            HybridDecoder::resolve_table(&cfg.arch, place.gpu_arch, place.kernels_dir, false)
        });
        if let Some((table, backend)) = resolved {
            let split = if backend == spite_gpu::GpuBackend::Cuda {
                place.hybrid_split()
            } else {
                spite_models::hybrid::LayerSplit::default()
            };
            let (model, r) = HybridDecoder::load_split(cfg, gguf, table, backend, ctx_len, &split)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
            println!(
                "device       : {:?} (hybrid) — weights {:.2} GiB + KV {:.2} GiB ({:?}) + recurrent state {:.2} GiB + scratch {:.2} MiB{}",
                r.backend,
                gib(r.weights_bytes),
                gib(r.kv_bytes),
                r.kv_kind,
                gib(r.state_bytes),
                r.scratch_bytes as f64 / (1u64 << 20) as f64,
                r.mem.map_or(String::new(), |(free, total)| format!(
                    "; {:.2}/{:.2} GiB free",
                    gib(free),
                    gib(total)
                )),
            );
            if r.stages.len() > 1 {
                for s in &r.stages {
                    println!(
                        "  stage      : GPU {} — layers {}..{} ({:.2} GiB)",
                        s.device,
                        s.layers.start,
                        s.layers.end,
                        gib(s.bytes)
                    );
                }
            }
            return Ok(Box::new(model));
        }
        // No op-complete kernel set: fall through to the Rust CPU implementation.
    }

    if place.device != Device::Cpu {
        match GpuDense::resolve_table(&cfg.arch, place.gpu_arch, place.kernels_dir) {
            Some(table) => {
                let qk_norm = cfg.arch == "qwen3";
                let (model, r) = GpuDense::load(cfg, gguf, table, ctx_len, qk_norm, kv_cfg)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
                println!(
                    "device       : cuda ({}) — weights {:.2} GiB + KV {:.2} GiB + scratch {:.2} MiB in VRAM; {:.2}/{:.2} GiB free",
                    place.gpu_arch,
                    gib(r.weights_bytes),
                    gib(r.kv_bytes),
                    r.scratch_bytes as f64 / (1u64 << 20) as f64,
                    gib(r.free_after),
                    gib(r.total)
                );
                // The kernel may not be able to read the requested tier; say so
                // rather than letting the summary above contradict what runs.
                if r.kv_quant_effective.key != kv_cfg.key || r.kv_quant_effective.val != kv_cfg.val
                {
                    println!(
                        "kv cache     : kernel accepts fewer tiers; using K={} V={} (requested K={} V={})",
                        r.kv_quant_effective.key, r.kv_quant_effective.val, kv_cfg.key, kv_cfg.val
                    );
                }
                return Ok(Box::new(model));
            }
            None if place.device == Device::Cuda => bail!(
                "--device cuda: no CUDA kernel for arch '{}' on '{}' under {} \
                 (build with cmake -DSPITE_MODELS=<family>/<model> -DSPITE_GPU_ARCHS=<card> \
                 and `cmake --install build --prefix .`)",
                cfg.arch,
                place.gpu_arch,
                place.kernels_dir.display()
            ),
            None => {}
        }
    }

    println!("device       : cpu");
    let mut model = ArchRegistry::default()
        .build(cfg)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    model
        .load_weights(gguf)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(model)
}

fn cmd_serve(
    model_args: &ModelArgs,
    hw: &HardwareArgs,
    feat: &FeatureArgs,
    host: &str,
    port: u16,
    ctx: Option<usize>,
) -> Result<()> {
    let (target, draft) = resolve_models(&model_args.model)?;
    let mgpu = resolve_hardware(hw);
    let target_gguf = GgufModel::open(&target)?;
    let model_arch = target_gguf.arch().to_owned();
    let kv_cfg = resolve_kv_quant(hw);

    print_engine_header(&model_arch, &mgpu);
    print_feature_summary(feat);
    print_kv_quant_summary(&kv_cfg);

    if let Some(ref d) = draft {
        let d_gguf = GgufModel::open(d)?;
        println!("draft model  : {} [speculative decoding]", d_gguf.arch());
    }

    print_dispatch_tables(&model_arch, &mgpu, &hw.kernels_dir)?;

    let vram_gib = mgpu.nodes.first().map_or(0, |n| n.vram_gib);
    let model_bytes = std::fs::metadata(&target).map_or(0, |m| m.len());
    maybe_print_offload_plan(hw, vram_gib, model_bytes, 0);

    if let Some(n) = ctx {
        println!("context      : {n} tokens");
    }
    println!("\nlistening on : http://{host}:{port}");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let state = std::sync::Arc::new(spite_server::AppState::load(
            &target,
            &hw.kernels_dir,
            &mgpu
                .nodes
                .first()
                .map(|n| n.gpu_arch.clone())
                .unwrap_or_default(),
            4,
        )?);
        let app = spite_server::api::router(state);
        let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

fn cmd_dispatch(model_args: &ModelArgs, hw: &HardwareArgs, feat: &FeatureArgs) -> Result<()> {
    let (target, _) = resolve_models(&model_args.model)?;
    let mgpu = resolve_hardware(hw);
    let gguf = GgufModel::open(&target)?;
    let model_arch = gguf.arch().to_owned();
    let kv_cfg = resolve_kv_quant(hw);

    print_engine_header(&model_arch, &mgpu);
    print_feature_summary(feat);
    print_kv_quant_summary(&kv_cfg);
    print_dispatch_tables(&model_arch, &mgpu, &hw.kernels_dir)?;

    let vram_gib = mgpu.nodes.first().map_or(0, |n| n.vram_gib);
    let model_bytes = std::fs::metadata(&target).map_or(0, |m| m.len());
    maybe_print_offload_plan(hw, vram_gib, model_bytes, 0);
    Ok(())
}

fn cmd_pull(model: &str, quant: &str) -> Result<()> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;
    println!("models dir   : {}", dir.display());
    println!("model        : {model}");
    println!("quant        : {quant}");

    // Download the matching .gguf via the Hugging Face CLI. The quant name is
    // matched as a substring of the file name (e.g. Q4_K_M -> *Q4_K_M*.gguf).
    let include = if quant.is_empty() {
        "*.gguf".to_string()
    } else {
        format!("*{quant}*.gguf")
    };
    println!("\nrunning: hf download {model} --include {include}");
    let status = std::process::Command::new("hf")
        .args(["download", model, "--include", &include, "--local-dir"])
        .arg(&dir)
        .status()?;

    if !status.success() {
        bail!(
            "`hf download` failed. Is the hf CLI installed and authenticated? \
             (https://huggingface.co/docs/hub/cli)"
        );
    }
    println!("\nDownloaded to {}", dir.display());
    Ok(())
}

// ── Print helpers ──────────────────────────────────────────────────────────

fn print_engine_header(model_arch: &str, mgpu: &MultiGpuSpec) {
    println!("spite — One Engine, Your Model, Your Card.");
    println!("──────────────────────────────────────────");
    println!("model arch   : {model_arch}");
    if mgpu.nodes.len() == 1 {
        let n = &mgpu.nodes[0];
        let vram_s = if n.vram_gib > 0 {
            format!(" ({} GiB)", n.vram_gib)
        } else {
            String::new()
        };
        println!("card         : {}{vram_s}", n.card_id);
        println!("gpu arch     : {}", n.gpu_arch);
    } else {
        mgpu.print_summary();
    }
    println!("kernels      :");
}

fn resolve_kv_quant(hw: &HardwareArgs) -> KvQuantConfig {
    let Some(ref s) = hw.kv_quant else {
        return KvQuantConfig::default();
    };
    match KvQuantConfig::from_str(s) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("warning: {e} — using f16 default");
            KvQuantConfig::default()
        }
    }
}

fn print_kv_quant_summary(cfg: &KvQuantConfig) {
    // VBR is always active: the tier shown is the starting point before the
    // cache degrades with depth (f16 → q8 → q5_1 → q4).
    if cfg.key == cfg.val {
        println!("kv cache     : {} start  (auto-VBR)", cfg.key);
    } else {
        println!(
            "kv cache     : K={} V={} start  (auto-VBR)",
            cfg.key, cfg.val
        );
    }
}

fn print_feature_summary(feat: &FeatureArgs) {
    let mut active: Vec<&str> = Vec::new();
    if feat.mtp {
        active.push("mtp");
    }
    if feat.dflash2 {
        active.push("dflash2");
    } else if feat.dflash {
        active.push("dflash");
    }
    if feat.vision {
        active.push("vision");
    }
    if !active.is_empty() {
        println!("features     : {}", active.join(", "));
    }
}

fn print_dispatch_tables(
    model_arch: &str,
    mgpu: &MultiGpuSpec,
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
    let ram_bytes = read_total_ram_bytes().unwrap_or(16 * GIB);
    let ram_budget = if hw.offload_disk { u64::MAX } else { ram_bytes };
    let bytes_per_layer = if n_layers > 0 {
        model_bytes / n_layers as u64
    } else {
        model_bytes
    };
    let cfg = OffloadConfig {
        vram_reserved_bytes: hw.vram_reserve_gib as u64 * GIB,
        ram_budget_bytes: ram_budget,
        ..Default::default()
    };
    match TieredPlacement::plan(
        vram_gib as u64 * GIB,
        ram_bytes,
        bytes_per_layer,
        n_layers,
        &cfg,
        None,
    ) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_fake_mistral4_produces_text() {
        let fake = spite_testkit::FakeGguf {
            arch: "mistral4".to_string(),
            ..Default::default()
        };
        let tmp = fake.write_to_tempfile().unwrap();
        let gguf = GgufModel::open(tmp.path()).unwrap();
        let text = generate(
            &gguf,
            "AB",
            4,
            0.0,
            None,
            &Placement::cpu(),
            &KvQuantConfig::default(),
        )
        .unwrap();
        assert!(!text.is_empty(), "expected generated text, got empty");
    }
}
