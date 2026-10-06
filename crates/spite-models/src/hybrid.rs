//! Hybrid linear/full-attention decoder (Qwen3.5-style: Gated Delta Net layers
//! interleaved with gated-Q full-attention layers), driven entirely by the ABI
//! dispatch table.
//!
//! Nothing here is vendor specific: every layer step is an ABI op
//! (`rms_norm`, `attention_ex`, `linear_attn`, `ffn`, `matmul`) resolved from
//! whatever kernel wins for the card, and all buffers are [`DeviceBuffer`]s of
//! the chosen [`GpuBackend`]. With `GpuBackend::Cpu` the same forward runs on
//! the generic C reference kernels, so a new vendor only has to supply ops.
//!
//! Residuals are fused into the ops (`attention_ex`, `linear_attn` and `ffn`
//! accumulate into the hidden state), exactly as in [`crate::gpu_dense`].
//! Weights stay in their stored (possibly quantized) format on the device.

use std::collections::HashMap;
use std::ffi::c_int;
use std::path::Path;
use std::sync::Mutex;

use spite_abi::{
    FfnActivation, SpiteAttnParams, SpiteCtx, SpiteGdnParams, SpiteKvCache, SpiteTensor, SpiteType,
};
use spite_compute::dequant::dequant_to_f32;
use spite_dispatch::{DispatchBuilder, DispatchTable, KernelSpec};
use spite_gpu::{DeviceBuffer, GpuBackend, cuda};
use spite_loader::GgufModel;

use crate::dense::packed_bytes;
use crate::gpu_dense::{
    DeviceWeight, VRAM_HEADROOM, err, f32_bytes, f32_bytes_mut, f32_tensor, gib, rc,
};
use crate::{ModelArch, ModelConfig, ModelError};

/// Source label of the portable CPU reference kernel.
const GENERIC: &str = "generic";

/// Per-layer tensors, resolved once at load.
struct Ffn {
    norm: SpiteTensor,
    gate: SpiteTensor,
    up: SpiteTensor,
    down: SpiteTensor,
}

struct AttnLayer {
    norm: SpiteTensor,
    wq: SpiteTensor,
    wk: SpiteTensor,
    wv: SpiteTensor,
    wo: SpiteTensor,
    q_norm: SpiteTensor,
    k_norm: SpiteTensor,
    /// Index into `HState::kv`.
    kv: usize,
    ffn: Ffn,
}

struct GdnLayer {
    norm: SpiteTensor,
    w_qkv: SpiteTensor,
    w_gate: SpiteTensor,
    w_beta: SpiteTensor,
    w_alpha: SpiteTensor,
    w_out: SpiteTensor,
    conv_w: SpiteTensor,
    ssm_dt: SpiteTensor,
    ssm_a: SpiteTensor,
    ssm_norm: SpiteTensor,
    /// Index into `HState::gdn`.
    st: usize,
    ffn: Ffn,
}

enum Layer {
    Attn(AttnLayer),
    Gdn(GdnLayer),
}

struct KvPair {
    k: DeviceBuffer,
    v: DeviceBuffer,
}

struct GdnState {
    conv_hist: DeviceBuffer,
    state: DeviceBuffer,
}

struct HState {
    /// Owns the device memory every `SpiteTensor` in `layers` points into.
    _weights: HashMap<String, DeviceWeight>,
    kv: Vec<KvPair>,
    gdn: Vec<GdnState>,
    scratch: DeviceBuffer,
    h: DeviceBuffer,
    n: DeviceBuffer,
    logits: DeviceBuffer,
}

/// Memory accounting reported after a successful load.
pub struct HybridReport {
    pub backend: GpuBackend,
    pub weights_bytes: usize,
    pub kv_bytes: usize,
    pub state_bytes: usize,
    pub scratch_bytes: usize,
    pub kv_kind: SpiteType,
    /// `(free, total)` device memory after load; `None` off CUDA.
    pub mem: Option<(usize, usize)>,
}

/// Hybrid decoder bound to one backend and one dispatch table.
pub struct HybridDecoder {
    config: ModelConfig,
    table: DispatchTable,
    backend: GpuBackend,
    n_ctx: usize,
    kv_kind: SpiteType,
    attn: SpiteAttnParams,
    gdn: SpiteGdnParams,
    layers: Vec<Layer>,
    out_norm: SpiteTensor,
    out_w: SpiteTensor,
    embd_host: Vec<u8>,
    embd_kind: SpiteType,
    state: Mutex<HState>,
}

/// True for archs this decoder implements: recurrent GDN layers with a dense FFN.
/// (The MoE variant `qwen35moe` needs expert routing ops and is not covered.)
pub fn is_hybrid(cfg: &ModelConfig) -> bool {
    cfg.ssm_d_state > 0 && cfg.n_expert == 0 && cfg.recurrent_layers.iter().any(|&r| r)
}

impl HybridDecoder {
    /// Resolve a dispatch table for `arch`. With `use_gpu` every needed op must
    /// come from a non-generic (device) kernel and the detected GPU backend is
    /// returned; without it, from the generic CPU kernel on `GpuBackend::Cpu`.
    /// `None` when the required kernels are absent: the caller decides the fallback.
    pub fn resolve_table(
        arch: &str,
        gpu_arch: &str,
        kernels_dir: &Path,
        use_gpu: bool,
    ) -> Option<(DispatchTable, GpuBackend)> {
        let (gpu_arch, backend) = if use_gpu {
            let b = GpuBackend::detect();
            if b == GpuBackend::Cpu {
                return None;
            }
            (gpu_arch, b)
        } else {
            (GENERIC, GpuBackend::Cpu)
        };
        let mut spec = KernelSpec::from_arch(arch, gpu_arch);
        if use_gpu && spec.card_id.is_empty() {
            spec.card_id = spite_dispatch::detect_card_id("");
        }
        let table = DispatchBuilder::new(kernels_dir, spec).build().ok()?;
        let want_generic = !use_gpu;
        let src_ok = |s: &spite_dispatch::OpSource| (s.gpu_arch == GENERIC) == want_generic;
        let ok = table.rms_norm.0.is_some()
            && table.attention_ex.0.is_some()
            && table.linear_attn.0.is_some()
            && table.ffn.0.is_some()
            && table.matmul.0.is_some()
            && src_ok(&table.rms_norm.1)
            && src_ok(&table.attention_ex.1)
            && src_ok(&table.linear_attn.1)
            && src_ok(&table.ffn.1)
            && src_ok(&table.matmul.1);
        ok.then_some((table, backend))
    }

    /// Upload every trunk tensor of `gguf`, allocate KV, GDN state and scratch.
    pub fn load(
        config: ModelConfig,
        gguf: &GgufModel,
        table: DispatchTable,
        backend: GpuBackend,
        n_ctx: usize,
    ) -> Result<(Self, HybridReport), ModelError> {
        if !is_hybrid(&config) {
            return Err(err("not a hybrid (gated delta net) architecture"));
        }
        let d = config.d_model;
        let n_layers = config.n_layers;
        let n_heads = config.n_heads.max(1);
        let n_kv = config.n_kv_heads.max(1);
        let head_dim = if config.key_length > 0 {
            config.key_length
        } else {
            d / n_heads
        };
        let rope_dim = if config.rope_dim_count > 0 {
            config.rope_dim_count.min(head_dim)
        } else {
            head_dim
        };
        let n_ctx = n_ctx.clamp(1, config.max_seq_len.max(1));
        let (n_kh, n_vh, s, k_conv) = (
            config.ssm_n_group,
            config.ssm_dt_rank,
            config.ssm_d_state,
            config.ssm_d_conv,
        );
        if n_kh == 0 || n_vh == 0 || n_vh % n_kh != 0 || config.ssm_d_inner != n_vh * s {
            return Err(err(format!(
                "inconsistent GDN geometry: groups {n_kh}, heads {n_vh}, state {s}, inner {}",
                config.ssm_d_inner
            )));
        }
        if config.recurrent_layers.len() < n_layers {
            return Err(err("recurrent layer flags shorter than layer count"));
        }
        let attn = SpiteAttnParams {
            head_dim: head_dim as i32,
            rope_dim: rope_dim as i32,
            gated_q: 1,
        };
        let gdn = SpiteGdnParams {
            n_kh: n_kh as i32,
            n_vh: n_vh as i32,
            head_dim: s as i32,
            d_conv: k_conv as i32,
            norm_eps: config.norm_eps,
        };

        // Trunk tensors only: NextN/MTP blocks (index >= n_layers) are not run.
        let is_trunk = |name: &str| {
            name.strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
                .is_none_or(|i| i < n_layers)
        };
        let mut sizes = Vec::new();
        let mut weights_bytes = 0usize;
        for name in gguf.tensor_names().filter(|n| is_trunk(n)) {
            let t = gguf.tensor(name);
            let n_elem: usize = t.ne.iter().map(|&x| x.max(1) as usize).product();
            let bytes = packed_bytes(t.kind, n_elem)
                .ok_or_else(|| err(format!("unsupported dtype {:?} for {name}", t.kind)))?;
            weights_bytes += bytes;
            sizes.push((name.to_owned(), t, bytes));
        }

        // KV tier: the best the winning attention_ex kernel declares.
        let kv_kind = [SpiteType::F16, SpiteType::F32]
            .into_iter()
            .find(|k| table.kv_cache_kinds_ex & (1u64 << *k as u32) != 0)
            .unwrap_or(SpiteType::F32);
        let kv_row = n_kv * head_dim;
        let kv_elem = kv_kind.block_bytes() as usize;
        let n_attn = (0..n_layers)
            .filter(|&i| !config.recurrent_layers[i])
            .count();
        let n_gdn = n_layers - n_attn;
        let kv_side = n_ctx * kv_row * kv_elem;
        let kv_bytes = n_attn * 2 * kv_side;
        let state_bytes = n_gdn * (gdn.conv_hist_floats() + gdn.state_floats()) * 4;
        let scratch_floats = [
            2 * config.d_ffn,
            attn.scratch_floats(n_heads, n_kv, n_ctx),
            gdn.scratch_floats(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let scratch_bytes = scratch_floats * 4;
        let act_bytes = (2 * d + config.vocab_size) * 4;
        let need = weights_bytes + kv_bytes + state_bytes + scratch_bytes + act_bytes;
        if backend == GpuBackend::Cuda {
            let (free, _) = cuda::mem_info().map_err(|e| err(e.to_string()))?;
            if need + VRAM_HEADROOM > free {
                return Err(err(format!(
                    "model needs {:.2} GiB VRAM (weights {:.2} + KV {:.2} @ {n_ctx} ctx + state {:.2}), \
                     only {:.2} GiB free — reduce --ctx",
                    gib(need),
                    gib(weights_bytes),
                    gib(kv_bytes),
                    gib(state_bytes),
                    gib(free)
                )));
            }
        }

        let alloc = |n: usize| {
            DeviceBuffer::alloc(backend, n.max(1)).map_err(|e| err(format!("alloc: {e}")))
        };
        let mut weights = HashMap::with_capacity(sizes.len());
        for (name, t, bytes) in sizes {
            let mut buf = alloc(bytes)?;
            // SAFETY: `t.data` points at `bytes` bytes of the GGUF mmap.
            let src = unsafe { std::slice::from_raw_parts(t.data as *const u8, bytes) };
            buf.upload(src)
                .map_err(|e| err(format!("upload {name}: {e}")))?;
            weights.insert(
                name,
                DeviceWeight {
                    buf,
                    ne: t.ne,
                    kind: t.kind,
                },
            );
        }

        let get = |name: &str| -> Result<SpiteTensor, ModelError> {
            weights
                .get(name)
                .map(DeviceWeight::tensor)
                .ok_or_else(|| ModelError::MissingWeight(name.into()))
        };
        let mut layers = Vec::with_capacity(n_layers);
        let (mut kv_i, mut gdn_i) = (0usize, 0usize);
        for l in 0..n_layers {
            let b = format!("blk.{l}");
            let ffn = Ffn {
                norm: get(&format!("{b}.post_attention_norm.weight"))?,
                gate: get(&format!("{b}.ffn_gate.weight"))?,
                up: get(&format!("{b}.ffn_up.weight"))?,
                down: get(&format!("{b}.ffn_down.weight"))?,
            };
            let norm = get(&format!("{b}.attn_norm.weight"))?;
            if config.recurrent_layers[l] {
                layers.push(Layer::Gdn(GdnLayer {
                    norm,
                    w_qkv: get(&format!("{b}.attn_qkv.weight"))?,
                    w_gate: get(&format!("{b}.attn_gate.weight"))?,
                    w_beta: get(&format!("{b}.ssm_beta.weight"))?,
                    w_alpha: get(&format!("{b}.ssm_alpha.weight"))?,
                    w_out: get(&format!("{b}.ssm_out.weight"))?,
                    conv_w: get(&format!("{b}.ssm_conv1d.weight"))?,
                    ssm_dt: get(&format!("{b}.ssm_dt.bias"))?,
                    ssm_a: get(&format!("{b}.ssm_a"))?,
                    ssm_norm: get(&format!("{b}.ssm_norm.weight"))?,
                    st: gdn_i,
                    ffn,
                }));
                gdn_i += 1;
            } else {
                let wq = get(&format!("{b}.attn_q.weight"))?;
                if wq.ne[1] as usize != 2 * n_heads * head_dim {
                    return Err(ModelError::ShapeMismatch {
                        name: format!("{b}.attn_q.weight"),
                        expected: vec![wq.ne[0], (2 * n_heads * head_dim) as u32],
                        actual: wq.ne.to_vec(),
                    });
                }
                layers.push(Layer::Attn(AttnLayer {
                    norm,
                    wq,
                    wk: get(&format!("{b}.attn_k.weight"))?,
                    wv: get(&format!("{b}.attn_v.weight"))?,
                    wo: get(&format!("{b}.attn_output.weight"))?,
                    q_norm: get(&format!("{b}.attn_q_norm.weight"))?,
                    k_norm: get(&format!("{b}.attn_k_norm.weight"))?,
                    kv: kv_i,
                    ffn,
                }));
                kv_i += 1;
            }
        }
        let out_norm = get("output_norm.weight")?;
        // Tied embeddings: token_embd doubles as the LM head.
        let out_w = get("output.weight").or_else(|_| get("token_embd.weight"))?;

        let zero = |buf: &mut DeviceBuffer| {
            buf.upload(&vec![0u8; buf.size])
                .map_err(|e| err(format!("zero: {e}")))
        };
        let mut kv = Vec::with_capacity(n_attn);
        for _ in 0..n_attn {
            kv.push(KvPair {
                k: alloc(kv_side)?,
                v: alloc(kv_side)?,
            });
        }
        let mut gdn_st = Vec::with_capacity(n_gdn);
        for _ in 0..n_gdn {
            let mut g = GdnState {
                conv_hist: alloc(gdn.conv_hist_floats() * 4)?,
                state: alloc(gdn.state_floats() * 4)?,
            };
            zero(&mut g.conv_hist)?;
            zero(&mut g.state)?;
            gdn_st.push(g);
        }
        let st = HState {
            _weights: weights,
            kv,
            gdn: gdn_st,
            scratch: alloc(scratch_bytes)?,
            h: alloc(d * 4)?,
            n: alloc(d * 4)?,
            logits: alloc(config.vocab_size * 4)?,
        };

        let embd = gguf.tensor("token_embd.weight");
        if embd.is_null() {
            return Err(ModelError::MissingWeight("token_embd.weight".into()));
        }
        let embd_elems: usize = embd.ne.iter().map(|&x| x.max(1) as usize).product();
        let embd_bytes = packed_bytes(embd.kind, embd_elems)
            .ok_or_else(|| err("unsupported token_embd dtype"))?;
        // SAFETY: a view into the GGUF mmap, copied out.
        let embd_host =
            unsafe { std::slice::from_raw_parts(embd.data as *const u8, embd_bytes) }.to_vec();

        let report = HybridReport {
            backend,
            weights_bytes,
            kv_bytes,
            state_bytes,
            scratch_bytes,
            kv_kind,
            mem: (backend == GpuBackend::Cuda)
                .then(|| cuda::mem_info().ok())
                .flatten(),
        };
        Ok((
            Self {
                config,
                table,
                backend,
                n_ctx,
                kv_kind,
                attn,
                gdn,
                layers,
                out_norm,
                out_w,
                embd_host,
                embd_kind: embd.kind,
                state: Mutex::new(st),
            },
            report,
        ))
    }

    fn embed(&self, tok: u32, out: &mut [f32]) -> Result<(), ModelError> {
        let d = self.config.d_model;
        if tok as usize >= self.config.vocab_size {
            return Err(err(format!("token id {tok} out of range")));
        }
        let row_bytes = packed_bytes(self.embd_kind, d).ok_or_else(|| err("embd dtype"))?;
        let off = tok as usize * row_bytes;
        let src = self
            .embd_host
            .get(off..off + row_bytes)
            .ok_or_else(|| err("embd row out of bounds"))?;
        dequant_to_f32(src, self.embd_kind, d, out).map_err(|e| err(format!("embed: {e}")))
    }
}

fn kv_tensor(buf: &DeviceBuffer, kind: SpiteType, row: usize, n_ctx: usize) -> SpiteTensor {
    let ne = [row as u32, n_ctx as u32, 1, 1];
    SpiteTensor {
        data: buf.as_ptr().cast(),
        ne,
        nb: SpiteTensor::contiguous_strides(kind, &ne),
        kind,
    }
}

impl ModelArch for HybridDecoder {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Recurrent state must restart from zero; KV rows are overwritten by position.
    fn reset_cache(&self) {
        if let Ok(mut st) = self.state.lock() {
            for g in &mut st.gdn {
                for buf in [&mut g.conv_hist, &mut g.state] {
                    let zeros = vec![0u8; buf.size];
                    // A failed upload leaves stale state; surface it on the next forward.
                    let _ = buf.upload(&zeros);
                }
            }
        }
    }

    fn forward(
        &self,
        tokens: &[u32],
        logits_out: &mut [f32],
        ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let cfg = &self.config;
        let (d, vocab) = (cfg.d_model, cfg.vocab_size);
        if logits_out.len() != tokens.len() * vocab {
            return Err(err("logits_out shape mismatch"));
        }
        let (Some(rms_norm), Some(attention_ex), Some(linear_attn), Some(ffn), Some(matmul)) = (
            self.table.rms_norm.0,
            self.table.attention_ex.0,
            self.table.linear_attn.0,
            self.table.ffn.0,
            self.table.matmul.0,
        ) else {
            return Err(err("dispatch table incomplete for hybrid decoder"));
        };
        let mut guard = self.state.lock().map_err(|_| err("state lock"))?;
        let st = &mut *guard;

        let mut h_t = f32_tensor(&st.h, d);
        let mut n_t = f32_tensor(&st.n, d);
        let mut logits_t = f32_tensor(&st.logits, vocab);
        let kv_row = cfg.n_kv_heads * self.attn.head_dim as usize;

        let mut emb = vec![0f32; d];
        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            if pos >= self.n_ctx {
                return Err(err(format!(
                    "position {pos} exceeds allocated context {}",
                    self.n_ctx
                )));
            }
            let kctx = SpiteCtx {
                n_ctx: self.n_ctx as c_int,
                n_batch: 1,
                n_threads: ctx.n_threads,
                pos: pos as c_int,
                n_heads: cfg.n_heads as c_int,
                n_kv_heads: cfg.n_kv_heads as c_int,
                gpu_stream: std::ptr::null_mut(),
                scratchpad: st.scratch.as_ptr().cast(),
                scratchpad_bytes: st.scratch.size,
            };
            self.embed(tok, &mut emb)?;
            st.h.upload(f32_bytes(&emb))
                .map_err(|e| err(e.to_string()))?;

            for (li, layer) in self.layers.iter().enumerate() {
                let ffn_w = match layer {
                    Layer::Attn(a) => {
                        let kvp = &st.kv[a.kv];
                        let mut kv = SpiteKvCache {
                            k: kv_tensor(&kvp.k, self.kv_kind, kv_row, self.n_ctx),
                            v: kv_tensor(&kvp.v, self.kv_kind, kv_row, self.n_ctx),
                            layer: li as c_int,
                        };
                        // SAFETY: every tensor points at live memory owned by `st` / `self`;
                        // the kernel ABI version is checked at load.
                        unsafe {
                            rc(
                                rms_norm(&mut n_t, &h_t, &a.norm, cfg.norm_eps, &kctx),
                                "rms_norm",
                                li,
                            )?;
                            rc(
                                attention_ex(
                                    &mut h_t,
                                    &n_t,
                                    &a.wq,
                                    &a.wk,
                                    &a.wv,
                                    &a.wo,
                                    &a.q_norm,
                                    &a.k_norm,
                                    cfg.norm_eps,
                                    &mut kv,
                                    cfg.rope_theta,
                                    &self.attn,
                                    &kctx,
                                ),
                                "attention_ex",
                                li,
                            )?;
                        }
                        &a.ffn
                    }
                    Layer::Gdn(g) => {
                        let gs = &st.gdn[g.st];
                        let mut conv_hist = f32_tensor(&gs.conv_hist, self.gdn.conv_hist_floats());
                        let mut state = f32_tensor(&gs.state, self.gdn.state_floats());
                        // SAFETY: as above.
                        unsafe {
                            rc(
                                rms_norm(&mut n_t, &h_t, &g.norm, cfg.norm_eps, &kctx),
                                "rms_norm",
                                li,
                            )?;
                            rc(
                                linear_attn(
                                    &mut h_t,
                                    &n_t,
                                    &g.w_qkv,
                                    &g.w_gate,
                                    &g.w_beta,
                                    &g.w_alpha,
                                    &g.w_out,
                                    &g.conv_w,
                                    &g.ssm_dt,
                                    &g.ssm_a,
                                    &g.ssm_norm,
                                    &mut conv_hist,
                                    &mut state,
                                    &self.gdn,
                                    &kctx,
                                ),
                                "linear_attn",
                                li,
                            )?;
                        }
                        &g.ffn
                    }
                };
                // SAFETY: as above.
                unsafe {
                    rc(
                        rms_norm(&mut n_t, &h_t, &ffn_w.norm, cfg.norm_eps, &kctx),
                        "rms_norm",
                        li,
                    )?;
                    rc(
                        ffn(
                            &mut h_t,
                            &n_t,
                            &ffn_w.gate,
                            &ffn_w.up,
                            &ffn_w.down,
                            FfnActivation::SiluGate,
                            &kctx,
                        ),
                        "ffn",
                        li,
                    )?;
                }
            }

            // SAFETY: as above.
            unsafe {
                rc(
                    rms_norm(&mut n_t, &h_t, &self.out_norm, cfg.norm_eps, &kctx),
                    "rms_norm",
                    cfg.n_layers,
                )?;
                rc(
                    matmul(&mut logits_t, &n_t, &self.out_w, &kctx),
                    "matmul",
                    cfg.n_layers,
                )?;
            }
            // Blocking copy on the default stream also syncs the kernels.
            st.logits
                .download(f32_bytes_mut(&mut logits_out[ti * vocab..(ti + 1) * vocab]))
                .map_err(|e| err(e.to_string()))?;
        }
        Ok(())
    }
}

impl HybridDecoder {
    /// Backend this decoder's buffers live on.
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }
}
