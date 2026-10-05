//! CUDA dense-decoder forward: weights resident in VRAM, ops via kernels.
//!
//! Generic over dense decoder-only archs (embed → per layer rmsnorm →
//! attention → rmsnorm → gated FFN → final norm → LM head). Every GGUF tensor
//! is uploaded **as stored** (Q8_0 stays packed) — no host F32 dequant.
//! Ops come from the [`DispatchTable`]; this module never launches GPU code
//! itself, so swapping in a faster kernel `.so` needs no Rust change.
//!
//! ABI v4 semantics relied on: `attention` and `ffn` accumulate into `out`
//! (residual fused), `matmul` overwrites.

use std::collections::HashMap;
use std::ffi::c_int;
use std::path::Path;
use std::sync::Mutex;

use spite_abi::{FfnActivation, SpiteCtx, SpiteKvCache, SpiteTensor, SpiteType};
use spite_compute::dequant::dequant_to_f32;
use spite_dispatch::{DispatchBuilder, DispatchTable, KernelSpec};
use spite_gpu::{DeviceBuffer, GpuBackend, cuda};
use spite_loader::GgufModel;

use crate::dense::{Activation, packed_bytes};
use crate::{ModelArch, ModelConfig, ModelError};

/// VRAM kept free beyond the computed need (driver, fragmentation).
const VRAM_HEADROOM: usize = 512 << 20;

fn err(msg: impl Into<String>) -> ModelError {
    ModelError::Forward(msg.into())
}

/// One weight tensor in VRAM.
struct DeviceWeight {
    buf: DeviceBuffer,
    ne: [u32; 4],
    kind: SpiteType,
}

impl DeviceWeight {
    fn tensor(&self) -> SpiteTensor {
        SpiteTensor {
            data: self.buf.as_ptr().cast(),
            ne: self.ne,
            nb: SpiteTensor::contiguous_strides(self.kind, &self.ne),
            kind: self.kind,
        }
    }
}

/// F32 device vector view of `n` elements.
fn f32_tensor(buf: &DeviceBuffer, n: usize) -> SpiteTensor {
    let ne = [n as u32, 1, 1, 1];
    SpiteTensor {
        data: buf.as_ptr().cast(),
        ne,
        nb: SpiteTensor::contiguous_strides(SpiteType::F32, &ne),
        kind: SpiteType::F32,
    }
}

/// All device-side state; behind a mutex because `forward` takes `&self`.
struct GpuState {
    weights: HashMap<String, DeviceWeight>,
    k_cache: Vec<DeviceBuffer>,
    v_cache: Vec<DeviceBuffer>,
    scratch: DeviceBuffer,
    h: DeviceBuffer,
    n: DeviceBuffer,
    logits: DeviceBuffer,
}

/// Dense decoder running entirely on one CUDA device.
pub struct GpuDense {
    config: ModelConfig,
    table: DispatchTable,
    activation: Activation,
    apply_qk_norm: bool,
    n_ctx: usize,
    head_dim: usize,
    /// Raw (packed) token embedding rows on the host; one row is dequantized
    /// per token. Avoids a device gather op the ABI does not define.
    embd_host: Vec<u8>,
    embd_kind: SpiteType,
    state: Mutex<GpuState>,
}

/// Bytes resident in VRAM after a successful load, for reporting.
pub struct VramReport {
    pub weights_bytes: usize,
    pub kv_bytes: usize,
    pub scratch_bytes: usize,
    pub free_after: usize,
    pub total: usize,
}

impl GpuDense {
    /// Resolve a dispatch table whose rms_norm/attention/ffn/matmul all come
    /// from a CUDA (`sm_*`) kernel. `None` when CUDA or the kernel is absent.
    pub fn resolve_table(arch: &str, gpu_arch: &str, kernels_dir: &Path) -> Option<DispatchTable> {
        if !gpu_arch.starts_with("sm_") || GpuBackend::detect() != GpuBackend::Cuda {
            return None;
        }
        let table = DispatchBuilder::new(kernels_dir, KernelSpec::from_arch(arch, gpu_arch))
            .build()
            .ok()?;
        let on_gpu = |src: &spite_dispatch::OpSource| src.gpu_arch.starts_with("sm_");
        let ok = table.rms_norm.0.is_some()
            && table.attention.0.is_some()
            && table.ffn.0.is_some()
            && table.matmul.0.is_some()
            && on_gpu(&table.rms_norm.1)
            && on_gpu(&table.attention.1)
            && on_gpu(&table.ffn.1)
            && on_gpu(&table.matmul.1);
        ok.then_some(table)
    }

    /// Upload every tensor of `gguf` to VRAM and allocate KV/scratch for
    /// `n_ctx` positions.
    pub fn load(
        config: ModelConfig,
        gguf: &GgufModel,
        table: DispatchTable,
        n_ctx: usize,
        apply_qk_norm: bool,
    ) -> Result<(Self, VramReport), ModelError> {
        let d = config.d_model;
        let n_heads = config.n_heads;
        let n_kv = config.n_kv_heads;
        let n_ctx = n_ctx.clamp(1, config.max_seq_len.max(1));

        // head_dim from the Q projection (Qwen3 small variants have
        // head_dim != d_model / n_heads).
        let wq0 = gguf.tensor("blk.0.attn_q.weight");
        if wq0.is_null() {
            return Err(ModelError::MissingWeight("blk.0.attn_q.weight".into()));
        }
        let head_dim = wq0.ne[1] as usize / n_heads.max(1);
        let kv_row = n_kv * head_dim;

        // ── Size everything first so we fail before allocating. ──────────
        let mut sizes = Vec::new();
        let mut weights_bytes = 0usize;
        for name in gguf.tensor_names() {
            let t = gguf.tensor(name);
            let n_elem: usize = t.ne.iter().map(|&x| x.max(1) as usize).product();
            let bytes = packed_bytes(t.kind, n_elem)
                .ok_or_else(|| err(format!("unsupported dtype {:?} for {name}", t.kind)))?;
            weights_bytes += bytes;
            sizes.push((name.to_owned(), t, bytes));
        }
        let kv_bytes = 2 * config.n_layers * n_ctx * kv_row * 4;
        let attn_scratch = (2 * n_heads * head_dim + kv_row + n_heads * n_ctx) * 4;
        let ffn_scratch = 2 * config.d_ffn * 4;
        let scratch_bytes = attn_scratch.max(ffn_scratch);
        let act_bytes = (2 * d + config.vocab_size) * 4;
        let need = weights_bytes + kv_bytes + scratch_bytes + act_bytes;
        let (free, total) = cuda::mem_info().map_err(|e| err(e.to_string()))?;
        if need + VRAM_HEADROOM > free {
            return Err(err(format!(
                "model needs {:.2} GiB VRAM (weights {:.2} + KV {:.2} @ {n_ctx} ctx), \
                 only {:.2} GiB free — reduce --ctx",
                gib(need),
                gib(weights_bytes),
                gib(kv_bytes),
                gib(free)
            )));
        }

        // ── Upload weights as stored. ─────────────────────────────────────
        let mut weights = HashMap::with_capacity(sizes.len());
        for (name, t, bytes) in sizes {
            let mut buf = DeviceBuffer::alloc(GpuBackend::Cuda, bytes)
                .map_err(|e| err(format!("alloc {name}: {e}")))?;
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

        let alloc = |n: usize| {
            DeviceBuffer::alloc(GpuBackend::Cuda, n).map_err(|e| err(format!("alloc: {e}")))
        };
        let mut k_cache = Vec::with_capacity(config.n_layers);
        let mut v_cache = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            k_cache.push(alloc(n_ctx * kv_row * 4)?);
            v_cache.push(alloc(n_ctx * kv_row * 4)?);
        }
        let state = GpuState {
            weights,
            k_cache,
            v_cache,
            scratch: alloc(scratch_bytes)?,
            h: alloc(d * 4)?,
            n: alloc(d * 4)?,
            logits: alloc(config.vocab_size * 4)?,
        };

        // Host copy of the packed embedding table for per-token row dequant.
        let embd = gguf.tensor("token_embd.weight");
        if embd.is_null() {
            return Err(ModelError::MissingWeight("token_embd.weight".into()));
        }
        let embd_elems: usize = embd.ne.iter().map(|&x| x.max(1) as usize).product();
        let embd_bytes = packed_bytes(embd.kind, embd_elems)
            .ok_or_else(|| err("unsupported token_embd dtype"))?;
        // SAFETY: as above — a view into the mmap, copied out.
        let embd_host =
            unsafe { std::slice::from_raw_parts(embd.data as *const u8, embd_bytes) }.to_vec();

        let (free_after, _) = cuda::mem_info().map_err(|e| err(e.to_string()))?;
        let report = VramReport {
            weights_bytes,
            kv_bytes,
            scratch_bytes,
            free_after,
            total,
        };
        Ok((
            Self {
                config,
                table,
                activation: Activation::SwiGlu,
                apply_qk_norm,
                n_ctx,
                head_dim,
                embd_host,
                embd_kind: embd.kind,
                state: Mutex::new(state),
            },
            report,
        ))
    }

    /// Dequantize embedding row `tok` into `out` (len d_model).
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

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn rc(code: c_int, op: &str, layer: usize) -> Result<(), ModelError> {
    if code == 0 {
        Ok(())
    } else {
        Err(err(format!(
            "{op} kernel failed (rc={code}) at layer {layer}"
        )))
    }
}

impl ModelArch for GpuDense {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// KV rows are overwritten by position, so nothing to clear.
    fn reset_cache(&self) {}

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
        let (Some(rms_norm), Some(attention), Some(ffn), Some(matmul)) = (
            self.table.rms_norm.0,
            self.table.attention.0,
            self.table.ffn.0,
            self.table.matmul.0,
        ) else {
            return Err(err("CUDA dispatch table incomplete"));
        };
        let st = self.state.lock().map_err(|_| err("gpu state lock"))?;
        let w = |name: &str| {
            st.weights
                .get(name)
                .map(DeviceWeight::tensor)
                .ok_or_else(|| ModelError::MissingWeight(name.into()))
        };
        let opt = |name: &str| st.weights.get(name).map(DeviceWeight::tensor);

        let mut h_t = f32_tensor(&st.h, d);
        let mut n_t = f32_tensor(&st.n, d);
        let mut logits_t = f32_tensor(&st.logits, vocab);
        let out_norm = w("output_norm.weight")?;
        // Tied embeddings: fall back to token_embd as the LM head.
        let out_w = w("output.weight").or_else(|_| w("token_embd.weight"))?;
        let kv_row = cfg.n_kv_heads * self.head_dim;
        let act = match self.activation {
            Activation::SwiGlu => FfnActivation::SiluGate,
            Activation::GeGlu => FfnActivation::GeluGate,
        };

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
            cuda::upload(st.h.as_ptr(), f32_bytes(&emb)).map_err(|e| err(e.to_string()))?;

            for layer in 0..cfg.n_layers {
                let b = format!("blk.{layer}");
                let attn_norm = w(&format!("{b}.attn_norm.weight"))?;
                let (wq, wk, wv, wo) = (
                    w(&format!("{b}.attn_q.weight"))?,
                    w(&format!("{b}.attn_k.weight"))?,
                    w(&format!("{b}.attn_v.weight"))?,
                    w(&format!("{b}.attn_output.weight"))?,
                );
                let (qn, kn) = if self.apply_qk_norm {
                    (
                        opt(&format!("{b}.attn_q_norm.weight")),
                        opt(&format!("{b}.attn_k_norm.weight")),
                    )
                } else {
                    (None, None)
                };
                let ffn_norm = w(&format!("{b}.ffn_norm.weight"))?;
                let (wg, wu, wd) = (
                    w(&format!("{b}.ffn_gate.weight"))?,
                    w(&format!("{b}.ffn_up.weight"))?,
                    w(&format!("{b}.ffn_down.weight"))?,
                );
                let kv_ne = [kv_row as u32, self.n_ctx as u32, 1, 1];
                let mut kv = SpiteKvCache {
                    k: SpiteTensor {
                        data: st.k_cache[layer].as_ptr().cast(),
                        ne: kv_ne,
                        nb: SpiteTensor::contiguous_strides(SpiteType::F32, &kv_ne),
                        kind: SpiteType::F32,
                    },
                    v: SpiteTensor {
                        data: st.v_cache[layer].as_ptr().cast(),
                        ne: kv_ne,
                        nb: SpiteTensor::contiguous_strides(SpiteType::F32, &kv_ne),
                        kind: SpiteType::F32,
                    },
                    layer: layer as c_int,
                };
                let null = std::ptr::null::<SpiteTensor>();
                // SAFETY: every tensor points at live device memory owned by
                // `st`; the kernel ABI is version-checked at load.
                unsafe {
                    rc(
                        rms_norm(&mut n_t, &h_t, &attn_norm, cfg.norm_eps, &kctx),
                        "rms_norm",
                        layer,
                    )?;
                    rc(
                        attention(
                            &mut h_t,
                            &n_t,
                            &wq,
                            &wk,
                            &wv,
                            &wo,
                            qn.as_ref().map_or(null, |t| t as *const _),
                            kn.as_ref().map_or(null, |t| t as *const _),
                            cfg.norm_eps,
                            &mut kv,
                            cfg.rope_theta,
                            &kctx,
                        ),
                        "attention",
                        layer,
                    )?;
                    rc(
                        rms_norm(&mut n_t, &h_t, &ffn_norm, cfg.norm_eps, &kctx),
                        "rms_norm",
                        layer,
                    )?;
                    rc(ffn(&mut h_t, &n_t, &wg, &wu, &wd, act, &kctx), "ffn", layer)?;
                }
            }

            // SAFETY: as above.
            unsafe {
                rc(
                    rms_norm(&mut n_t, &h_t, &out_norm, cfg.norm_eps, &kctx),
                    "rms_norm",
                    cfg.n_layers,
                )?;
                rc(
                    matmul(&mut logits_t, &n_t, &out_w, &kctx),
                    "matmul",
                    cfg.n_layers,
                )?;
            }
            let dst = &mut logits_out[ti * vocab..(ti + 1) * vocab];
            // Blocking D2H copy on the default stream also syncs the kernels.
            cuda::download(st.logits.as_ptr(), f32_bytes_mut(dst))
                .map_err(|e| err(e.to_string()))?;
        }
        Ok(())
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; u8 alignment is 1.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: as above; every bit pattern is a valid f32.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), std::mem::size_of_val(v)) }
}
