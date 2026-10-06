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
    FfnActivation, SpiteAttnParams, SpiteCtx, SpiteGdnParams, SpiteKvCache, SpiteMoeParams,
    SpiteTensor, SpiteType,
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

/// Per-layer dense FFN tensors.
struct DenseFfn {
    norm: SpiteTensor,
    gate: SpiteTensor,
    up: SpiteTensor,
    down: SpiteTensor,
}

/// Per-layer MoE FFN tensors (e.g. Qwen3.5 MoE).
struct MoeFfn {
    norm: SpiteTensor,
    w_gate_inp: SpiteTensor,
    w_up_exps: SpiteTensor,
    w_gate_exps: SpiteTensor,
    w_down_exps: SpiteTensor,
    w_up_shexp: Option<SpiteTensor>,
    w_gate_shexp: Option<SpiteTensor>,
    w_down_shexp: Option<SpiteTensor>,
    params: SpiteMoeParams,
}

enum LayerFfn {
    Dense(Box<DenseFfn>),
    Moe(Box<MoeFfn>),
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
    ffn: LayerFfn,
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
    ffn: LayerFfn,
}

enum Layer {
    Attn(AttnLayer),
    Gdn(GdnLayer),
}

/// NextN / MTP draft head: one gated full-attention block stored after the
/// trunk as `blk.{n_layers}`, fed `[enorm(embed(x_{t+1})) || hnorm(h_t)]`.
struct MtpHead {
    enorm: SpiteTensor,
    hnorm: SpiteTensor,
    eh_proj: SpiteTensor,
    layer: AttnLayer,
    head_norm: SpiteTensor,
    head: SpiteTensor,
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
    /// Final-normed hidden state of the last trunk token or MTP step: the
    /// `h_t` the next MTP step consumes.
    hid: DeviceBuffer,
    /// MTP only: the draft token's embedding and the packed `[2*d]` stem.
    emb: DeviceBuffer,
    pack: DeviceBuffer,
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
    /// The NextN (MTP) draft head was loaded and `mtp_step` is available.
    pub mtp: bool,
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
    mtp: Option<MtpHead>,
    out_norm: SpiteTensor,
    out_w: SpiteTensor,
    embd_host: Vec<u8>,
    embd_kind: SpiteType,
    state: Mutex<HState>,
}

/// True for archs this decoder implements: recurrent GDN layers with dense or MoE FFN.
pub fn is_hybrid(cfg: &ModelConfig) -> bool {
    cfg.ssm_d_state > 0 && cfg.recurrent_layers.iter().any(|&r| r)
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
        let ffn_ok = (table.ffn.0.is_some() && src_ok(&table.ffn.1))
            || (table.moe_ffn.0.is_some() && src_ok(&table.moe_ffn.1));
        let ok = table.rms_norm.0.is_some()
            && table.attention_ex.0.is_some()
            && table.linear_attn.0.is_some()
            && ffn_ok
            && table.matmul.0.is_some()
            && src_ok(&table.rms_norm.1)
            && src_ok(&table.attention_ex.1)
            && src_ok(&table.linear_attn.1)
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

        // NextN/MTP: only the first block (`blk.{n_layers}`) is used, and only
        // when it is a full-attention block and an mtp_stem op runs on this
        // backend; otherwise its tensors are not uploaded at all.
        let mtp_b = format!("blk.{n_layers}");
        let mtp_stem_ok = table.mtp_stem.0.is_some()
            && (table.mtp_stem.1.gpu_arch == GENERIC) == (backend == GpuBackend::Cpu);
        let has_tensor = |name: String| gguf.tensor_names().any(|n| n == name);
        let has_mtp = config.n_nextn_predict_layers > 0
            && mtp_stem_ok
            && has_tensor(format!("{mtp_b}.nextn.eh_proj.weight"))
            && has_tensor(format!("{mtp_b}.attn_q.weight"));
        let n_blocks = n_layers + usize::from(has_mtp);
        let is_loaded = |name: &str| {
            name.strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
                .is_none_or(|i| i < n_blocks)
        };
        let mut sizes = Vec::new();
        let mut weights_bytes = 0usize;
        for name in gguf.tensor_names().filter(|n| is_loaded(n)) {
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
        let n_kv_layers = n_attn + usize::from(has_mtp);
        let kv_side = n_ctx * kv_row * kv_elem;
        let kv_bytes = n_kv_layers * 2 * kv_side;
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
        // h, n, hid, logits (+ emb and the [2*d] stem pack for MTP).
        let act_bytes = (3 * d + config.vocab_size + if has_mtp { 3 * d } else { 0 }) * 4;
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
        let load_ffn = |b: &str| -> Result<LayerFfn, ModelError> {
            let post_norm = get(&format!("{b}.post_attention_norm.weight"))?;
            Ok(if config.n_expert > 0 {
                let w_gate_inp = get(&format!("{b}.ffn_gate_inp.weight"))?;
                let w_up_exps = get(&format!("{b}.ffn_up_exps.weight"))?;
                let w_gate_exps = get(&format!("{b}.ffn_gate_exps.weight"))?;
                let w_down_exps = get(&format!("{b}.ffn_down_exps.weight"))?;
                let w_up_shexp = get(&format!("{b}.ffn_up_shexp.weight")).ok();
                let w_gate_shexp = get(&format!("{b}.ffn_gate_shexp.weight")).ok();
                let w_down_shexp = get(&format!("{b}.ffn_down_shexp.weight")).ok();
                let shared_intermediate_size =
                    w_up_shexp.as_ref().map(|t| t.ne[1] as i32).unwrap_or(0);
                let params = SpiteMoeParams {
                    num_experts: config.n_expert as i32,
                    num_experts_per_tok: config.n_expert_used as i32,
                    intermediate_size: w_up_exps.ne[1] as i32,
                    shared_intermediate_size,
                    weights_scale: config.expert_weights_scale,
                };
                LayerFfn::Moe(Box::new(MoeFfn {
                    norm: post_norm,
                    w_gate_inp,
                    w_up_exps,
                    w_gate_exps,
                    w_down_exps,
                    w_up_shexp,
                    w_gate_shexp,
                    w_down_shexp,
                    params,
                }))
            } else {
                LayerFfn::Dense(Box::new(DenseFfn {
                    norm: post_norm,
                    gate: get(&format!("{b}.ffn_gate.weight"))?,
                    up: get(&format!("{b}.ffn_up.weight"))?,
                    down: get(&format!("{b}.ffn_down.weight"))?,
                }))
            })
        };
        let load_attn = |b: &str, kv: usize| -> Result<AttnLayer, ModelError> {
            let wq = get(&format!("{b}.attn_q.weight"))?;
            if wq.ne[1] as usize != 2 * n_heads * head_dim {
                return Err(ModelError::ShapeMismatch {
                    name: format!("{b}.attn_q.weight"),
                    expected: vec![wq.ne[0], (2 * n_heads * head_dim) as u32],
                    actual: wq.ne.to_vec(),
                });
            }
            Ok(AttnLayer {
                norm: get(&format!("{b}.attn_norm.weight"))?,
                wq,
                wk: get(&format!("{b}.attn_k.weight"))?,
                wv: get(&format!("{b}.attn_v.weight"))?,
                wo: get(&format!("{b}.attn_output.weight"))?,
                q_norm: get(&format!("{b}.attn_q_norm.weight"))?,
                k_norm: get(&format!("{b}.attn_k_norm.weight"))?,
                kv,
                ffn: load_ffn(b)?,
            })
        };
        let mut layers = Vec::with_capacity(n_layers);
        let (mut kv_i, mut gdn_i) = (0usize, 0usize);
        for l in 0..n_layers {
            let b = format!("blk.{l}");
            if config.recurrent_layers[l] {
                layers.push(Layer::Gdn(GdnLayer {
                    norm: get(&format!("{b}.attn_norm.weight"))?,
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
                    ffn: load_ffn(&b)?,
                }));
                gdn_i += 1;
            } else {
                layers.push(Layer::Attn(load_attn(&b, kv_i)?));
                kv_i += 1;
            }
        }
        let out_norm = get("output_norm.weight")?;
        // Tied embeddings: token_embd doubles as the LM head.
        let out_w = get("output.weight").or_else(|_| get("token_embd.weight"))?;
        let mtp = if has_mtp {
            let b = &mtp_b;
            Some(MtpHead {
                enorm: get(&format!("{b}.nextn.enorm.weight"))?,
                hnorm: get(&format!("{b}.nextn.hnorm.weight"))?,
                eh_proj: get(&format!("{b}.nextn.eh_proj.weight"))?,
                layer: load_attn(b, n_attn)?,
                head_norm: get(&format!("{b}.nextn.shared_head_norm.weight"))
                    .or_else(|_| get("output_norm.weight"))?,
                head: get(&format!("{b}.nextn.shared_head_head.weight"))
                    .or_else(|_| get("output.weight"))
                    .or_else(|_| get("token_embd.weight"))?,
            })
        } else {
            None
        };

        let zero = |buf: &mut DeviceBuffer| {
            buf.upload(&vec![0u8; buf.size])
                .map_err(|e| err(format!("zero: {e}")))
        };
        let mut kv = Vec::with_capacity(n_kv_layers);
        for _ in 0..n_kv_layers {
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
            hid: alloc(d * 4)?,
            emb: alloc(if has_mtp { d * 4 } else { 0 })?,
            pack: alloc(if has_mtp { 2 * d * 4 } else { 0 })?,
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
            mtp: has_mtp,
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
                mtp,
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
        let (Some(rms_norm), Some(_), Some(linear_attn), Some(matmul)) = (
            self.table.rms_norm.0,
            self.table.attention_ex.0,
            self.table.linear_attn.0,
            self.table.matmul.0,
        ) else {
            return Err(err("dispatch table incomplete for hybrid decoder"));
        };
        let mut guard = self.state.lock().map_err(|_| err("state lock"))?;
        let st = &mut *guard;

        let mut h_t = f32_tensor(&st.h, d);
        let mut n_t = f32_tensor(&st.n, d);
        let mut hid_t = f32_tensor(&st.hid, d);
        let mut logits_t = f32_tensor(&st.logits, vocab);

        let mut emb = vec![0f32; d];
        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            if pos >= self.n_ctx {
                return Err(err(format!(
                    "position {pos} exceeds allocated context {}",
                    self.n_ctx
                )));
            }
            let kctx = self.kctx(st, pos, ctx.n_threads);
            self.embed(tok, &mut emb)?;
            st.h.upload(f32_bytes(&emb))
                .map_err(|e| err(e.to_string()))?;

            for (li, layer) in self.layers.iter().enumerate() {
                let ffn_w = match layer {
                    Layer::Attn(a) => {
                        self.attn_block(st, a, li, &mut h_t, &mut n_t, &kctx)?;
                        &a.ffn
                    }
                    Layer::Gdn(g) => {
                        let gs = &st.gdn[g.st];
                        let mut conv_hist = f32_tensor(&gs.conv_hist, self.gdn.conv_hist_floats());
                        let mut state = f32_tensor(&gs.state, self.gdn.state_floats());
                        // SAFETY: every tensor points at live memory owned by `st` / `self`;
                        // the kernel ABI version is checked at load.
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
                self.ffn_block(ffn_w, li, &mut h_t, &mut n_t, &kctx)?;
            }

            // SAFETY: as above.
            unsafe {
                rc(
                    rms_norm(&mut hid_t, &h_t, &self.out_norm, cfg.norm_eps, &kctx),
                    "rms_norm",
                    cfg.n_layers,
                )?;
                rc(
                    matmul(&mut logits_t, &hid_t, &self.out_w, &kctx),
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

    /// True when the NextN (MTP) draft head is loaded; see [`Self::mtp_step`].
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// One NextN (MTP) draft step, on the same backend and ops as the trunk.
    ///
    /// Consumes the final-normed hidden state `h_t` left by the latest
    /// [`ModelArch::forward`] token (or the previous `mtp_step`, when chaining)
    /// together with `token` = x_{t+1}, runs the draft block at RoPE position
    /// `pos` = t, and writes the logits for x_{t+2}. Its own normed output
    /// replaces `h_t`, so the next chained call takes the drafted token and
    /// `pos + 1`.
    ///
    /// The draft block attends over its own KV cache, so every position up to
    /// `pos` must have been through `mtp_step` (run it after each trunk token,
    /// prompt included); a rejected draft is undone by re-running the trunk,
    /// whose `forward` refreshes `h_t`, and the stale rows are overwritten.
    pub fn mtp_step(
        &self,
        token: u32,
        pos: usize,
        logits_out: &mut [f32],
    ) -> Result<(), ModelError> {
        let Some(m) = &self.mtp else {
            return Err(err("model has no usable NextN (MTP) head"));
        };
        let cfg = &self.config;
        let (d, vocab) = (cfg.d_model, cfg.vocab_size);
        if logits_out.len() != vocab {
            return Err(err("logits_out shape mismatch"));
        }
        if pos >= self.n_ctx {
            return Err(err(format!(
                "position {pos} exceeds allocated context {}",
                self.n_ctx
            )));
        }
        let (Some(rms_norm), Some(matmul), Some(mtp_stem)) = (
            self.table.rms_norm.0,
            self.table.matmul.0,
            self.table.mtp_stem.0,
        ) else {
            return Err(err("dispatch table incomplete for the MTP head"));
        };
        let mut guard = self.state.lock().map_err(|_| err("state lock"))?;
        let st = &mut *guard;
        let kctx = self.kctx(st, pos, 1);
        let li = cfg.n_layers;

        let mut emb = vec![0f32; d];
        self.embed(token, &mut emb)?;
        st.emb
            .upload(f32_bytes(&emb))
            .map_err(|e| err(e.to_string()))?;
        let emb_t = f32_tensor(&st.emb, d);
        let mut pack_t = f32_tensor(&st.pack, 2 * d);
        let mut h_t = f32_tensor(&st.h, d);
        let mut n_t = f32_tensor(&st.n, d);
        let mut hid_t = f32_tensor(&st.hid, d);
        let mut logits_t = f32_tensor(&st.logits, vocab);

        // SAFETY: every tensor points at live memory owned by `st` / `self`.
        unsafe {
            rc(
                mtp_stem(
                    &mut pack_t,
                    &emb_t,
                    &hid_t,
                    &m.enorm,
                    &m.hnorm,
                    cfg.norm_eps,
                    &kctx,
                ),
                "mtp_stem",
                li,
            )?;
            rc(matmul(&mut h_t, &pack_t, &m.eh_proj, &kctx), "matmul", li)?;
        }
        self.attn_block(st, &m.layer, li, &mut h_t, &mut n_t, &kctx)?;
        self.ffn_block(&m.layer.ffn, li, &mut h_t, &mut n_t, &kctx)?;
        // SAFETY: as above.
        unsafe {
            rc(
                rms_norm(&mut hid_t, &h_t, &m.head_norm, cfg.norm_eps, &kctx),
                "rms_norm",
                li,
            )?;
            rc(matmul(&mut logits_t, &hid_t, &m.head, &kctx), "matmul", li)?;
        }
        st.logits
            .download(f32_bytes_mut(logits_out))
            .map_err(|e| err(e.to_string()))
    }

    fn kctx(&self, st: &HState, pos: usize, n_threads: c_int) -> SpiteCtx {
        SpiteCtx {
            n_ctx: self.n_ctx as c_int,
            n_batch: 1,
            n_threads,
            pos: pos as c_int,
            n_heads: self.config.n_heads as c_int,
            n_kv_heads: self.config.n_kv_heads as c_int,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: st.scratch.as_ptr().cast(),
            scratchpad_bytes: st.scratch.size,
        }
    }

    /// `h += attention_ex(rms_norm(h))` for one gated full-attention block.
    fn attn_block(
        &self,
        st: &HState,
        a: &AttnLayer,
        li: usize,
        h_t: &mut SpiteTensor,
        n_t: &mut SpiteTensor,
        kctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let (Some(rms_norm), Some(attention_ex)) =
            (self.table.rms_norm.0, self.table.attention_ex.0)
        else {
            return Err(err("dispatch table incomplete for hybrid decoder"));
        };
        let cfg = &self.config;
        let kv_row = cfg.n_kv_heads * self.attn.head_dim as usize;
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
                rms_norm(n_t, h_t, &a.norm, cfg.norm_eps, kctx),
                "rms_norm",
                li,
            )?;
            rc(
                attention_ex(
                    h_t,
                    n_t,
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
                    kctx,
                ),
                "attention_ex",
                li,
            )
        }
    }

    /// `h += ffn(rms_norm(h))`, dense SwiGLU or MoE.
    fn ffn_block(
        &self,
        ffn_w: &LayerFfn,
        li: usize,
        h_t: &mut SpiteTensor,
        n_t: &mut SpiteTensor,
        kctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let Some(rms_norm) = self.table.rms_norm.0 else {
            return Err(err("rms_norm op missing"));
        };
        let eps = self.config.norm_eps;
        match ffn_w {
            LayerFfn::Dense(d) => {
                let Some(ffn) = self.table.ffn.0 else {
                    return Err(err("ffn op missing"));
                };
                // SAFETY: as in `attn_block`.
                unsafe {
                    rc(rms_norm(n_t, h_t, &d.norm, eps, kctx), "rms_norm", li)?;
                    rc(
                        ffn(
                            h_t,
                            n_t,
                            &d.gate,
                            &d.up,
                            &d.down,
                            FfnActivation::SiluGate,
                            kctx,
                        ),
                        "ffn",
                        li,
                    )
                }
            }
            LayerFfn::Moe(m) => {
                let Some(moe_ffn) = self.table.moe_ffn.0 else {
                    return Err(err("moe_ffn op missing"));
                };
                let opt = |t: &Option<SpiteTensor>| {
                    t.as_ref().map_or(std::ptr::null(), |t| t as *const _)
                };
                // SAFETY: as in `attn_block`.
                unsafe {
                    rc(rms_norm(n_t, h_t, &m.norm, eps, kctx), "rms_norm", li)?;
                    rc(
                        moe_ffn(
                            h_t,
                            n_t,
                            &m.w_gate_inp,
                            &m.w_up_exps,
                            &m.w_gate_exps,
                            &m.w_down_exps,
                            opt(&m.w_up_shexp),
                            opt(&m.w_gate_shexp),
                            opt(&m.w_down_shexp),
                            &m.params,
                            kctx,
                        ),
                        "moe_ffn",
                        li,
                    )
                }
            }
        }
    }
}
