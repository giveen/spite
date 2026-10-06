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

    /// Speculative decoding strategy: "none", "mtp".
    #[arg(long, default_value = "none")]
    spec: String,

    /// Number of draft tokens for speculative decoding (e.g. 1..=4).
    #[arg(long, default_value_t = 1)]
    draft_tokens: usize,

    /// Convenience flag to enable MTP speculative decoding (equivalent to --spec mtp).
    #[arg(long)]
    mtp: bool,

    /// Prompt text to benchmark with.
    #[arg(long)]
    prompt: Option<String>,

    /// Number of prompt tokens for prefill throughput test (pads/repeats prompt if larger).
    #[arg(long)]
    n_prompt: Option<usize>,
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

    let cfg = ModelConfig::from(hp);
    let (model, device_label): (Box<dyn spite_models::ModelArch>, String) =
        if spite_models::hybrid::is_hybrid(&cfg) {
            use spite_models::hybrid::HybridDecoder;
            let want_gpu = args.device != Device::Cpu;
            let resolved = if want_gpu {
                HybridDecoder::resolve_table(&cfg.arch, &gpu_arch, &args.kernels_dir, true)
            } else {
                None
            };
            let resolved = resolved.or_else(|| {
                HybridDecoder::resolve_table(&cfg.arch, &gpu_arch, &args.kernels_dir, false)
            });
            match resolved {
                Some((table, backend)) => {
                    // Every visible GPU is a candidate; the decoder stays on
                    // the first one unless the model does not fit there.
                    let split = spite_models::hybrid::LayerSplit {
                        devices: if backend == spite_gpu::GpuBackend::Cuda {
                            (0..spite_gpu::cuda::device_count().unwrap_or(0)).collect()
                        } else {
                            Vec::new()
                        },
                        shares: Vec::new(),
                    };
                    let (m, r) =
                        HybridDecoder::load_split(cfg, &gguf, table, backend, 4096, &split)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    for s in r.stages.iter().filter(|_| r.stages.len() > 1) {
                        eprintln!(
                            "stage: GPU {} layers {}..{} ({} MiB)",
                            s.device,
                            s.layers.start,
                            s.layers.end,
                            s.bytes >> 20
                        );
                    }
                    peak_mem_mib = ((r.weights_bytes + r.state_bytes + r.scratch_bytes)
                        / (1024 * 1024)) as u64;
                    let dev = if backend == spite_gpu::GpuBackend::Cpu {
                        "cpu".into()
                    } else {
                        format!("cuda ({gpu_arch})")
                    };
                    (Box::new(m), dev)
                }
                None => {
                    anyhow::bail!("no hybrid kernel for arch '{}' on '{}'", cfg.arch, gpu_arch);
                }
            }
        } else {
            match args.device {
                Device::Cpu => {
                    let mut m = ArchRegistry::default().build(cfg)?;
                    m.load_weights(&gguf)?;
                    (m, "cpu".into())
                }
                Device::Cuda | Device::Auto => {
                    match spite_models::gpu_dense::GpuDense::resolve_table(
                        &cfg.arch,
                        &gpu_arch,
                        &args.kernels_dir,
                    ) {
                        Some(table) => {
                            let qk_norm = cfg.arch == "qwen3";
                            let (m, r) = spite_models::gpu_dense::GpuDense::load(
                                cfg, &gguf, table, 4096, qk_norm, &kv_cfg,
                            )
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                            peak_mem_mib = ((r.weights_bytes + r.kv_bytes + r.scratch_bytes)
                                / (1024 * 1024)) as u64;
                            (Box::new(m), format!("cuda ({gpu_arch}, kv {kv_label})"))
                        }
                        None if args.device == Device::Cuda => {
                            anyhow::bail!(
                                "--device cuda: no CUDA kernel for arch '{}' on '{}' under {}",
                                cfg.arch,
                                gpu_arch,
                                args.kernels_dir.display()
                            );
                        }
                        None => {
                            let mut m = ArchRegistry::default().build(cfg)?;
                            m.load_weights(&gguf)?;
                            (m, "cpu".into())
                        }
                    }
                }
            }
        };

    let use_mtp = args.mtp || args.spec.eq_ignore_ascii_case("mtp");
    let draft_tokens = if use_mtp { args.draft_tokens.max(1) } else { 0 };

    let mut spec = spite_dispatch::KernelSpec::from_arch(model.config().arch.as_str(), &gpu_arch);
    if spec.card_id.is_empty() {
        spec.card_id = spite_dispatch::detect_card_id(args.card.as_deref().unwrap_or(""));
    }
    let dispatch_table = spite_dispatch::DispatchBuilder::new(&args.kernels_dir, spec)
        .build()
        .ok();
    if args.verbose
        && let Some(ref table) = dispatch_table
    {
        table.print_sources();
    }

    let backend = if args.device == Device::Cpu {
        spite_gpu::GpuBackend::Cpu
    } else {
        spite_gpu::GpuBackend::detect()
    };

    let mut mtp_runner = if use_mtp {
        let table = dispatch_table.unwrap_or_else(spite_dispatch::DispatchTable::fallback);
        Some(MtpBenchRunner::new(&gguf, model.config(), table, backend)?)
    } else {
        None
    };

    // Fixed prompt keeps runs comparable; real-model runs use --n-tokens.
    let prompt_text = args
        .prompt
        .as_deref()
        .unwrap_or("Benchmark prompt for throughput measurement.");
    let mut prompt_ids = tokenizer.encode(prompt_text, true)?;
    if let Some(target_len) = args.n_prompt {
        if target_len > prompt_ids.len() && !prompt_ids.is_empty() {
            let base = prompt_ids.clone();
            while prompt_ids.len() < target_len {
                let take = (target_len - prompt_ids.len()).min(base.len());
                prompt_ids.extend_from_slice(&base[..take]);
            }
        } else if target_len > 0 && target_len < prompt_ids.len() {
            prompt_ids.truncate(target_len);
        }
    }

    let exec_cfg = ExecutorConfig {
        kv_quant: kv_cfg.clone(),
        ..ExecutorConfig::default()
    };
    let mut exec = Executor::new(exec_cfg);
    exec.load_model(model);

    // Warm-up (caches, allocator paths) — not timed.
    let _ = exec.generate(&tokenizer, &prompt_ids, 4, 0.0, 0)?;

    let mut ttft_ms = 0f64;
    let mut prefill_s = 0f64;
    let mut n_prefill_tok = 0usize;
    let mut decode_s = 0f64;
    let mut n_tok = 0usize;
    let mut total_drafts = 0usize;
    let mut accepted_drafts = 0usize;

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
        let prefill_elapsed = t0.elapsed().as_secs_f64();
        ttft_ms += prefill_elapsed * 1000.0;
        prefill_s += prefill_elapsed;
        n_prefill_tok += ids.len();

        let t1 = std::time::Instant::now();
        let cfg = spite_sampling::SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = 0u64;

        let mut run_tokens = 0usize;
        if !use_mtp {
            for _ in 0..args.n_tokens {
                let tok = spite_sampling::sample(&mut logits, &ids, &cfg, &mut rng)?;
                ids.push(tok);
                logits = exec.decode_step(tok, &ctx)?;
            }
            run_tokens = args.n_tokens;
        } else {
            let mtp = mtp_runner.as_mut().expect("mtp runner initialized");
            while run_tokens < args.n_tokens {
                // 1. Sample verified token from trunk logits
                let tok = spite_sampling::sample(&mut logits, &ids, &cfg, &mut rng)?;
                ids.push(tok);
                run_tokens += 1;
                if run_tokens >= args.n_tokens {
                    break;
                }

                // 2. Draft up to K tokens using MTP stem
                let mut current_tok = tok;
                for _ in 0..draft_tokens {
                    mtp.draft_step(current_tok, &ctx)?;

                    let draft_tok = spite_sampling::sample(&mut logits, &ids, &cfg, &mut rng)?;
                    total_drafts += 1;

                    // 3. Verify candidate token with main model forward
                    let next_logits = exec.decode_step(draft_tok, &ctx)?;
                    let expected_tok =
                        spite_sampling::sample(&mut next_logits.clone(), &ids, &cfg, &mut rng)?;

                    if draft_tok == expected_tok {
                        ids.push(draft_tok);
                        run_tokens += 1;
                        accepted_drafts += 1;
                        logits = next_logits;
                        current_tok = draft_tok;
                        if run_tokens >= args.n_tokens {
                            break;
                        }
                    } else {
                        exec.rollback(1);
                        ids.push(expected_tok);
                        run_tokens += 1;
                        logits = exec.decode_step(expected_tok, &ctx)?;
                        break;
                    }
                }
            }
        }
        decode_s += t1.elapsed().as_secs_f64();
        n_tok += run_tokens;
    }
    let runs = args.n_runs.max(1) as f64;
    let prefill_tps = n_prefill_tok as f64 / prefill_s.max(1e-9);
    let decode_tps = n_tok as f64 / decode_s.max(1e-9);
    let full_label = if use_mtp {
        format!(
            "{} [{}, spec=mtp K={}]",
            args.model, device_label, draft_tokens
        )
    } else {
        format!("{} [{}]", args.model, device_label)
    };
    let acceptance_rate = if use_mtp && total_drafts > 0 {
        Some(accepted_drafts as f64 / total_drafts as f64)
    } else {
        None
    };
    let result = spite_bench::BenchResult {
        label: full_label,
        tps: decode_tps,
        prefill_tps,
        ttft_ms: ttft_ms / runs,
        peak_mem_mib,
        n_runs: args.n_runs,
        acceptance_rate,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result_json(&result))?);
    } else {
        result.print();
    }
    Ok(())
}

struct MtpBenchRunner {
    table: spite_dispatch::DispatchTable,
    buf_out: spite_gpu::DeviceBuffer,
    buf_embed: spite_gpu::DeviceBuffer,
    buf_hidden: spite_gpu::DeviceBuffer,
    buf_enorm: Option<spite_gpu::DeviceBuffer>,
    buf_hnorm: Option<spite_gpu::DeviceBuffer>,
    d_model: usize,
    vocab_size: usize,
    norm_eps: f32,
    embd_tensor: spite_abi::SpiteTensor,
}

impl MtpBenchRunner {
    fn new(
        gguf: &spite_loader::GgufModel,
        cfg: &spite_models::ModelConfig,
        table: spite_dispatch::DispatchTable,
        backend: spite_gpu::GpuBackend,
    ) -> Result<Self> {
        let d = cfg.d_model;
        let buf_out = spite_gpu::DeviceBuffer::alloc(backend, 2 * d * 4)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut buf_embed =
            spite_gpu::DeviceBuffer::alloc(backend, d * 4).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut buf_hidden =
            spite_gpu::DeviceBuffer::alloc(backend, d * 4).map_err(|e| anyhow::anyhow!("{e}"))?;

        let init_zeros = vec![0.0f32; d];
        let bytes_zeros =
            unsafe { std::slice::from_raw_parts(init_zeros.as_ptr() as *const u8, d * 4) };
        buf_embed
            .upload(bytes_zeros)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        buf_hidden
            .upload(bytes_zeros)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mtp_layer = cfg.n_layers;
        let t_en = gguf.tensor(&format!("blk.{mtp_layer}.nextn.enorm.weight"));
        let t_hn = gguf.tensor(&format!("blk.{mtp_layer}.nextn.hnorm.weight"));

        let mut buf_enorm = None;
        let mut buf_hnorm = None;

        if !t_en.data.is_null() && !t_hn.data.is_null() {
            let mut b_en = spite_gpu::DeviceBuffer::alloc(backend, d * 4)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut b_hn = spite_gpu::DeviceBuffer::alloc(backend, d * 4)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let mut en_f32 = vec![0.0f32; d];
            let mut hn_f32 = vec![0.0f32; d];

            let en_bytes = unsafe {
                std::slice::from_raw_parts(t_en.data as *const u8, t_en.nb[0] as usize * d)
            };
            let hn_bytes = unsafe {
                std::slice::from_raw_parts(t_hn.data as *const u8, t_hn.nb[0] as usize * d)
            };
            let _ = spite_compute::dequant::dequant_to_f32(en_bytes, t_en.kind, d, &mut en_f32);
            let _ = spite_compute::dequant::dequant_to_f32(hn_bytes, t_hn.kind, d, &mut hn_f32);

            let en_u8 = unsafe { std::slice::from_raw_parts(en_f32.as_ptr() as *const u8, d * 4) };
            let hn_u8 = unsafe { std::slice::from_raw_parts(hn_f32.as_ptr() as *const u8, d * 4) };
            b_en.upload(en_u8).map_err(|e| anyhow::anyhow!("{e}"))?;
            b_hn.upload(hn_u8).map_err(|e| anyhow::anyhow!("{e}"))?;

            buf_enorm = Some(b_en);
            buf_hnorm = Some(b_hn);
        } else if !cfg.arch.contains("gemma") {
            let mut b_en = spite_gpu::DeviceBuffer::alloc(backend, d * 4)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut b_hn = spite_gpu::DeviceBuffer::alloc(backend, d * 4)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let ones = vec![1.0f32; d];
            let ones_u8 = unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, d * 4) };
            b_en.upload(ones_u8).map_err(|e| anyhow::anyhow!("{e}"))?;
            b_hn.upload(ones_u8).map_err(|e| anyhow::anyhow!("{e}"))?;
            buf_enorm = Some(b_en);
            buf_hnorm = Some(b_hn);
        }

        let embd_tensor = gguf.tensor("token_embd.weight");

        Ok(Self {
            table,
            buf_out,
            buf_embed,
            buf_hidden,
            buf_enorm,
            buf_hnorm,
            d_model: d,
            vocab_size: cfg.vocab_size,
            norm_eps: cfg.norm_eps,
            embd_tensor,
        })
    }

    fn draft_step(&mut self, token: u32, ctx: &spite_abi::SpiteCtx) -> Result<()> {
        let d = self.d_model;
        if !self.embd_tensor.data.is_null() {
            let tok_idx = (token as usize).min(self.vocab_size.saturating_sub(1));
            let row_offset = tok_idx.saturating_mul(self.embd_tensor.nb[1] as usize);
            let row_bytes = self.embd_tensor.nb[1] as usize;
            let src = unsafe {
                std::slice::from_raw_parts(
                    (self.embd_tensor.data as *const u8).add(row_offset),
                    row_bytes,
                )
            };
            let mut embed_f32 = vec![0.0f32; d];
            let _ = spite_compute::dequant::dequant_to_f32(
                src,
                self.embd_tensor.kind,
                d,
                &mut embed_f32,
            );
            let embed_u8 =
                unsafe { std::slice::from_raw_parts(embed_f32.as_ptr() as *const u8, d * 4) };
            self.buf_embed
                .upload(embed_u8)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        let mut t_out = spite_abi::SpiteTensor {
            data: self.buf_out.as_ptr().cast(),
            ne: [(2 * d) as u32, 1, 1, 1],
            nb: [
                4,
                (2 * d * 4) as u64,
                (2 * d * 4) as u64,
                (2 * d * 4) as u64,
            ],
            kind: spite_abi::SpiteType::F32,
        };
        let t_embed = spite_abi::SpiteTensor {
            data: self.buf_embed.as_ptr().cast(),
            ne: [d as u32, 1, 1, 1],
            nb: [4, (d * 4) as u64, (d * 4) as u64, (d * 4) as u64],
            kind: spite_abi::SpiteType::F32,
        };
        let t_hidden = spite_abi::SpiteTensor {
            data: self.buf_hidden.as_ptr().cast(),
            ne: [d as u32, 1, 1, 1],
            nb: [4, (d * 4) as u64, (d * 4) as u64, (d * 4) as u64],
            kind: spite_abi::SpiteType::F32,
        };
        let t_en = self.buf_enorm.as_ref().map(|b| spite_abi::SpiteTensor {
            data: b.as_ptr().cast(),
            ne: [d as u32, 1, 1, 1],
            nb: [4, (d * 4) as u64, (d * 4) as u64, (d * 4) as u64],
            kind: spite_abi::SpiteType::F32,
        });
        let t_hn = self.buf_hnorm.as_ref().map(|b| spite_abi::SpiteTensor {
            data: b.as_ptr().cast(),
            ne: [d as u32, 1, 1, 1],
            nb: [4, (d * 4) as u64, (d * 4) as u64, (d * 4) as u64],
            kind: spite_abi::SpiteType::F32,
        });

        let (func_opt, _) = &self.table.mtp_stem;
        let status = if let Some(func) = func_opt {
            unsafe {
                func(
                    &mut t_out,
                    &t_embed,
                    &t_hidden,
                    t_en.as_ref().map_or(std::ptr::null(), |t| t as *const _),
                    t_hn.as_ref().map_or(std::ptr::null(), |t| t as *const _),
                    self.norm_eps,
                    ctx,
                )
            }
        } else {
            unsafe {
                spite_dispatch::fallback::mtp_stem(
                    &mut t_out,
                    &t_embed,
                    &t_hidden,
                    t_en.as_ref().map_or(std::ptr::null(), |t| t as *const _),
                    t_hn.as_ref().map_or(std::ptr::null(), |t| t as *const _),
                    self.norm_eps,
                    ctx,
                )
            }
        };

        if status != 0 {
            anyhow::bail!("mtp_stem failed with error code {status}");
        }

        Ok(())
    }
}

fn result_json(r: &spite_bench::BenchResult) -> serde_json::Value {
    let mut v = serde_json::json!({
        "label": r.label,
        "tps": r.tps,
        "decode_tps": r.tps,
        "prefill_tps": r.prefill_tps,
        "ttft_ms": r.ttft_ms,
        "peak_mem_mib": r.peak_mem_mib,
        "n_runs": r.n_runs,
    });
    if let Some(acc) = r.acceptance_rate {
        v.as_object_mut()
            .unwrap()
            .insert("acceptance_rate".into(), serde_json::json!(acc));
        v.as_object_mut().unwrap().insert(
            "acceptance_percentage".into(),
            serde_json::json!(acc * 100.0),
        );
        v.as_object_mut().unwrap().insert(
            "acceptance_pct".into(),
            serde_json::json!(format!("{:.1}%", acc * 100.0)),
        );
    }
    v
}
