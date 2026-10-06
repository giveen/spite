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
use std::ops::Range;
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

/// One pipeline stage's residual stream and scratch, on that stage's device.
struct StageBufs {
    device: usize,
    scratch: DeviceBuffer,
    h: DeviceBuffer,
    n: DeviceBuffer,
}

struct HState {
    /// Owns the device memory every `SpiteTensor` in `layers` points into.
    _weights: HashMap<String, DeviceWeight>,
    kv: Vec<KvPair>,
    gdn: Vec<GdnState>,
    /// Pipeline stages in layer order; the last one also runs the LM head and
    /// the MTP block, and owns `hid`, `emb`, `pack` and `logits`.
    stages: Vec<StageBufs>,
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
    /// `(free, total)` memory of the first stage's device after load; `None` off CUDA.
    pub mem: Option<(usize, usize)>,
    /// Pipeline stages in layer order (one entry on a single device).
    pub stages: Vec<StageReport>,
}

/// One loaded pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageReport {
    /// CUDA ordinal (0 off CUDA).
    pub device: usize,
    /// Trunk layers this stage runs.
    pub layers: Range<usize>,
    /// Device bytes the stage allocated: weights, KV, recurrent state,
    /// activations, scratch, plus the LM head and MTP block on the last stage.
    pub bytes: usize,
}

/// How the trunk is spread over CUDA devices (pipeline parallelism).
///
/// Consecutive layers form a stage on one device; only the `d_model` hidden
/// state crosses a stage boundary, through host memory, so the devices need
/// no peer access (PCIe-only boxes work). The last stage also holds the LM
/// head and the NextN block.
///
/// The default is the current device only, exactly the single-GPU path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayerSplit {
    /// CUDA ordinals to use, in layer order. Empty: the current device.
    pub devices: Vec<usize>,
    /// Relative layer shares, one per device (e.g. `[33, 32]`). Empty: stay
    /// on the first device when the model fits there, otherwise spread over
    /// every listed device in proportion to its free memory, balancing bytes
    /// rather than layer counts.
    pub shares: Vec<u32>,
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
    /// Pipeline stage (index into `HState::stages`) of each layer.
    layer_stage: Vec<usize>,
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

/// Block index of a `blk.N.*` tensor name.
fn block_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Resolve `split` to `(device per stage, layers per stage)`.
///
/// `layer_bytes` is each trunk layer's device footprint, `head_bytes` what
/// the last stage carries on top, `stage_fixed` every stage's scratch and
/// activations, and `free_on` a device's free bytes.
fn plan_split(
    split: &LayerSplit,
    layer_bytes: &[usize],
    head_bytes: usize,
    stage_fixed: usize,
    cuda_on: bool,
    free_on: &dyn Fn(usize) -> Result<usize, ModelError>,
) -> Result<(Vec<usize>, Vec<usize>), ModelError> {
    let n_layers = layer_bytes.len();
    let mut devices = split.devices.clone();
    if cuda_on {
        if devices.is_empty() {
            devices.push(cuda::current_device().map_err(|e| err(e.to_string()))?);
        }
        let n = cuda::device_count().map_err(|e| err(e.to_string()))?;
        if let Some(&bad) = devices.iter().find(|&&dev| dev >= n) {
            return Err(err(format!("GPU {bad} does not exist ({n} visible)")));
        }
    } else if devices.is_empty() {
        devices.push(0);
    }

    if !split.shares.is_empty() {
        if split.shares.len() != devices.len() {
            return Err(err(format!(
                "layer split has {} shares for {} GPUs",
                split.shares.len(),
                devices.len()
            )));
        }
        let counts = share_counts(&split.shares, n_layers)
            .ok_or_else(|| err("layer split shares must not all be zero"))?;
        if counts.contains(&0) {
            return Err(err(format!(
                "a layer split share is too small to get any of the {n_layers} layers"
            )));
        }
        return Ok((devices, counts));
    }

    let mut uniq: Vec<usize> = Vec::with_capacity(devices.len());
    for dev in devices {
        if !uniq.contains(&dev) {
            uniq.push(dev);
        }
    }
    let total = layer_bytes.iter().sum::<usize>() + head_bytes + stage_fixed;
    if uniq.len() == 1 || total + VRAM_HEADROOM <= free_on(uniq[0])? {
        return Ok((vec![uniq[0]], vec![n_layers]));
    }
    let n_st = uniq.len();
    let budgets = uniq
        .iter()
        .enumerate()
        .map(|(i, &dev)| {
            let reserved = VRAM_HEADROOM + stage_fixed + if i + 1 == n_st { head_bytes } else { 0 };
            Ok(free_on(dev)?.saturating_sub(reserved))
        })
        .collect::<Result<Vec<_>, ModelError>>()?;
    let counts = balance_layers(layer_bytes, &budgets);
    // A device whose budget earned no layer is dropped; the last stage stays
    // even when empty because it carries the head.
    Ok(uniq
        .into_iter()
        .zip(counts)
        .enumerate()
        .filter(|&(i, (_, c))| c > 0 || i + 1 == n_st)
        .map(|(_, sc)| sc)
        .unzip())
}

/// Split `n` layers in proportion to `shares` (largest remainder, ties to
/// the earlier stage). `None` when every share is zero.
fn share_counts(shares: &[u32], n: usize) -> Option<Vec<usize>> {
    let total: u64 = shares.iter().map(|&s| u64::from(s)).sum();
    if total == 0 {
        return None;
    }
    let scaled = |s: u32| u64::from(s) * n as u64;
    let mut counts: Vec<usize> = shares
        .iter()
        .map(|&s| (scaled(s) / total) as usize)
        .collect();
    let mut rem: Vec<(u64, usize)> = shares
        .iter()
        .enumerate()
        .map(|(i, &s)| (scaled(s) % total, i))
        .collect();
    rem.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let short = n - counts.iter().sum::<usize>();
    for &(_, i) in rem.iter().take(short) {
        counts[i] += 1;
    }
    Some(counts)
}

/// Contiguous layer counts per stage so each stage's bytes track its share
/// of `budgets`. A layer joins a stage while at least half of it fits under
/// the stage's cumulative target; the last stage takes the rest.
fn balance_layers(layer_bytes: &[usize], budgets: &[usize]) -> Vec<usize> {
    let total: u128 = layer_bytes.iter().map(|&b| b as u128).sum();
    let cap: u128 = budgets.iter().map(|&b| b as u128).sum::<u128>().max(1);
    let mut counts = vec![0usize; budgets.len()];
    let (mut l, mut acc, mut target) = (0usize, 0u128, 0u128);
    for (s, &b) in budgets.iter().enumerate() {
        if s + 1 == budgets.len() {
            counts[s] = layer_bytes.len() - l;
            break;
        }
        target += total * b as u128 / cap;
        while l < layer_bytes.len() && acc + layer_bytes[l] as u128 / 2 <= target {
            acc += layer_bytes[l] as u128;
            l += 1;
            counts[s] += 1;
        }
    }
    counts
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

    /// Upload every trunk tensor of `gguf` to the current device, allocate KV,
    /// GDN state and scratch.
    ///
    /// # Errors
    /// See [`Self::load_split`].
    pub fn load(
        config: ModelConfig,
        gguf: &GgufModel,
        table: DispatchTable,
        backend: GpuBackend,
        n_ctx: usize,
    ) -> Result<(Self, HybridReport), ModelError> {
        Self::load_split(config, gguf, table, backend, n_ctx, &LayerSplit::default())
    }

    /// Like [`Self::load`], spreading the trunk over devices as `split` says.
    ///
    /// Off CUDA the device ordinals are ignored but the stages are still
    /// built, so the stage hand-off can be exercised on the CPU backend.
    ///
    /// # Errors
    /// Malformed GDN geometry, missing or unsupported weights, an invalid
    /// `split` (unknown ordinal, share count not matching the devices, a share
    /// too small to get a layer), or a device without room for its stage.
    pub fn load_split(
        config: ModelConfig,
        gguf: &GgufModel,
        table: DispatchTable,
        backend: GpuBackend,
        n_ctx: usize,
        split: &LayerSplit,
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
        // Embedding lookups run on the host copy, so `token_embd` only goes to
        // the device when it doubles as the LM head (tied embeddings).
        let tied = !has_tensor("output.weight".into());
        let is_loaded = |name: &str| match block_of(name) {
            Some(i) => i < n_blocks,
            None => tied || name != "token_embd.weight",
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

        // Device bytes per trunk layer, of the head (everything outside the
        // trunk: LM head, final norm, MTP block and its KV, hid/logits/emb/pack)
        // and of each stage's own scratch + h + n.
        let mut layer_bytes: Vec<usize> = (0..n_layers)
            .map(|l| {
                if config.recurrent_layers[l] {
                    (gdn.conv_hist_floats() + gdn.state_floats()) * 4
                } else {
                    2 * kv_side
                }
            })
            .collect();
        let mut head_bytes = (d + config.vocab_size + if has_mtp { 3 * d } else { 0 }) * 4
            + if has_mtp { 2 * kv_side } else { 0 };
        for (name, _, bytes) in &sizes {
            match block_of(name) {
                Some(i) if i < n_layers => layer_bytes[i] += bytes,
                _ => head_bytes += bytes,
            }
        }
        let stage_fixed = scratch_bytes + 2 * d * 4;
        let cuda_on = backend == GpuBackend::Cuda;
        let free_on = |dev: usize| -> Result<usize, ModelError> {
            if !cuda_on {
                return Ok(usize::MAX);
            }
            cuda::with_device(dev, cuda::mem_info)
                .map(|(free, _)| free)
                .map_err(|e| err(e.to_string()))
        };

        let (devices, counts) = plan_split(
            split,
            &layer_bytes,
            head_bytes,
            stage_fixed,
            cuda_on,
            &free_on,
        )?;
        let n_stages = devices.len();
        let layer_stage: Vec<usize> = counts
            .iter()
            .enumerate()
            .flat_map(|(s, &c)| std::iter::repeat_n(s, c))
            .collect();
        let mut stage_reports = Vec::with_capacity(n_stages);
        let mut first = 0;
        for (s, (&device, &c)) in devices.iter().zip(&counts).enumerate() {
            let bytes = layer_bytes[first..first + c].iter().sum::<usize>()
                + stage_fixed
                + if s + 1 == n_stages { head_bytes } else { 0 };
            stage_reports.push(StageReport {
                device,
                layers: first..first + c,
                bytes,
            });
            first += c;
        }

        // Stages that share a device (e.g. a split test on one GPU) add up.
        if cuda_on {
            let mut per_dev: Vec<(usize, usize)> = Vec::new();
            for r in &stage_reports {
                match per_dev.iter_mut().find(|(dev, _)| *dev == r.device) {
                    Some((_, b)) => *b += r.bytes,
                    None => per_dev.push((r.device, r.bytes)),
                }
            }
            for &(dev, need) in &per_dev {
                let free = free_on(dev)?;
                if need + VRAM_HEADROOM <= free {
                    continue;
                }
                return Err(err(if per_dev.len() == 1 {
                    format!(
                        "model needs {:.2} GiB VRAM on GPU {dev} (weights {:.2} + KV {:.2} @ {n_ctx} ctx \
                         + state {:.2}), only {:.2} GiB free — reduce --ctx or spread the layers over \
                         more GPUs (--gpus)",
                        gib(need),
                        gib(weights_bytes),
                        gib(kv_bytes),
                        gib(state_bytes),
                        gib(free)
                    )
                } else {
                    format!(
                        "GPU {dev} needs {:.2} GiB for its pipeline stages, only {:.2} GiB free \
                         — reduce --ctx, add GPUs (--gpus) or rebalance (--layer-split)",
                        gib(need),
                        gib(free)
                    )
                }));
            }
        }
        let head_dev = devices[n_stages - 1];

        let alloc = |dev: usize, n: usize| {
            DeviceBuffer::alloc_on(backend, dev, n.max(1)).map_err(|e| err(format!("alloc: {e}")))
        };
        let mut weights = HashMap::with_capacity(sizes.len());
        for (name, t, bytes) in sizes {
            let dev = match block_of(&name) {
                Some(i) if i < n_layers => devices[layer_stage[i]],
                _ => head_dev,
            };
            let mut buf = alloc(dev, bytes)?;
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
        // KV pairs and GDN states in layer order, matching `AttnLayer::kv` and
        // `GdnLayer::st`, each on its layer's device; the MTP KV comes last.
        let mut kv = Vec::with_capacity(n_kv_layers);
        let mut gdn_st = Vec::with_capacity(n_gdn);
        for l in 0..n_layers {
            let dev = devices[layer_stage[l]];
            if config.recurrent_layers[l] {
                let mut g = GdnState {
                    conv_hist: alloc(dev, gdn.conv_hist_floats() * 4)?,
                    state: alloc(dev, gdn.state_floats() * 4)?,
                };
                zero(&mut g.conv_hist)?;
                zero(&mut g.state)?;
                gdn_st.push(g);
            } else {
                kv.push(KvPair {
                    k: alloc(dev, kv_side)?,
                    v: alloc(dev, kv_side)?,
                });
            }
        }
        if has_mtp {
            kv.push(KvPair {
                k: alloc(head_dev, kv_side)?,
                v: alloc(head_dev, kv_side)?,
            });
        }
        let mut stages = Vec::with_capacity(n_stages);
        for &device in &devices {
            stages.push(StageBufs {
                device,
                scratch: alloc(device, scratch_bytes)?,
                h: alloc(device, d * 4)?,
                n: alloc(device, d * 4)?,
            });
        }
        let st = HState {
            _weights: weights,
            kv,
            gdn: gdn_st,
            stages,
            hid: alloc(head_dev, d * 4)?,
            emb: alloc(head_dev, if has_mtp { d * 4 } else { 0 })?,
            pack: alloc(head_dev, if has_mtp { 2 * d * 4 } else { 0 })?,
            logits: alloc(head_dev, config.vocab_size * 4)?,
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
            mem: cuda_on
                .then(|| cuda::with_device(devices[0], cuda::mem_info).ok())
                .flatten(),
            stages: stage_reports,
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
                layer_stage,
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

/// `[cols, rows]` F32 view of a device buffer (one column per token) — the
/// batched-prefill activation shape.
fn f32_tensor2(buf: &DeviceBuffer, cols: usize, rows: usize) -> SpiteTensor {
    let ne = [cols as u32, rows as u32, 1, 1];
    SpiteTensor {
        data: buf.as_ptr().cast(),
        ne,
        nb: SpiteTensor::contiguous_strides(SpiteType::F32, &ne),
        kind: SpiteType::F32,
    }
}

/// `[cols, 1]` F32 view of column `t` of a `[cols, m]` buffer.
fn f32_tensor_col(buf: &DeviceBuffer, cols: usize, t: usize) -> SpiteTensor {
    // SAFETY: the buffer holds cols*m floats and t < m.
    let data = unsafe { buf.as_ptr().add(t * cols * 4) };
    let ne = [cols as u32, 1, 1, 1];
    SpiteTensor {
        data: data.cast(),
        ne,
        nb: SpiteTensor::contiguous_strides(SpiteType::F32, &ne),
        kind: SpiteType::F32,
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
        // A multi-token prompt on a batch-capable (generic) kernel goes through
        // the layer-major batched path; decode (one token) and GPU kernels
        // without batch support keep the per-token path.
        if tokens.len() > 1 && self.batch_capable() {
            return self.forward_batch(tokens, logits_out, ctx);
        }
        let (Some(rms_norm), Some(_), Some(_), Some(matmul)) = (
            self.table.rms_norm.0,
            self.table.attention_ex.0,
            self.table.linear_attn.0,
            self.table.matmul.0,
        ) else {
            return Err(err("dispatch table incomplete for hybrid decoder"));
        };
        let mut guard = self.state.lock().map_err(|_| err("state lock"))?;
        let st = &mut *guard;
        let last = st.stages.len() - 1;

        let mut hid_t = f32_tensor(&st.hid, d);
        let mut logits_t = f32_tensor(&st.logits, vocab);

        // Host bounce buffer: the token embedding, then each stage hand-off.
        let mut host = vec![0f32; d];
        for (ti, &tok) in tokens.iter().enumerate() {
            let pos = ctx.pos as usize + ti;
            if pos >= self.n_ctx {
                return Err(err(format!(
                    "position {pos} exceeds allocated context {}",
                    self.n_ctx
                )));
            }
            self.embed(tok, &mut host)?;
            st.stages[0]
                .h
                .upload(f32_bytes(&host))
                .map_err(|e| err(e.to_string()))?;
            let mut cur = 0;
            self.enter(st, cur)?;
            let (mut h_t, mut n_t, mut kctx) = self.stage_view(st, cur, pos, ctx.n_threads);

            for (li, layer) in self.layers.iter().enumerate() {
                let s = self.layer_stage[li];
                if s != cur {
                    self.hand_off(st, cur, s, &mut host)?;
                    cur = s;
                    (h_t, n_t, kctx) = self.stage_view(st, cur, pos, ctx.n_threads);
                }
                let ffn_w = match layer {
                    Layer::Attn(a) => {
                        self.attn_block(st, a, li, &mut h_t, &mut n_t, &kctx)?;
                        &a.ffn
                    }
                    Layer::Gdn(g) => {
                        self.gdn_block(st, g, li, &mut h_t, &mut n_t, &kctx)?;
                        &g.ffn
                    }
                };
                self.ffn_block(ffn_w, li, &mut h_t, &mut n_t, &kctx)?;
            }
            if cur != last {
                self.hand_off(st, cur, last, &mut host)?;
                (h_t, _, kctx) = self.stage_view(st, last, pos, ctx.n_threads);
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
        // The MTP block, its KV and `hid` all live on the last stage.
        let last = st.stages.len() - 1;
        self.enter(st, last)?;
        let (mut h_t, mut n_t, kctx) = self.stage_view(st, last, pos, 1);
        let li = cfg.n_layers;

        let mut emb = vec![0f32; d];
        self.embed(token, &mut emb)?;
        st.emb
            .upload(f32_bytes(&emb))
            .map_err(|e| err(e.to_string()))?;
        let emb_t = f32_tensor(&st.emb, d);
        let mut pack_t = f32_tensor(&st.pack, 2 * d);
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

    /// True when the resolved ops can take a batched `[d, m]` activation and the
    /// decoder is on a single stage. The generic reference implements batching;
    /// the per-card CUDA kernels do not yet, so they keep the per-token path.
    fn batch_capable(&self) -> bool {
        let batch = |s: &spite_dispatch::OpSource| s.caps & spite_dispatch::CAP_BATCH != 0;
        self.layer_stage.iter().all(|&s| s == 0)
            && batch(&self.table.rms_norm.1)
            && batch(&self.table.ffn.1)
            && batch(&self.table.matmul.1)
            && self
                .table
                .attention_ex
                .0
                .is_none_or(|_| batch(&self.table.attention_ex.1))
            && self
                .table
                .linear_attn
                .0
                .is_none_or(|_| batch(&self.table.linear_attn.1))
            && (self.config.n_expert == 0
                || self
                    .table
                    .moe_ffn
                    .0
                    .is_none_or(|_| batch(&self.table.moe_ffn.1)))
    }

    /// Layer-major batched prefill: one batched op call per layer for `m`
    /// tokens (columns), instead of m sequential passes. The KV cache and the
    /// GDN conv/state advance in token order, so the result matches the
    /// per-token path. Requires [`Self::batch_capable`].
    fn forward_batch(
        &self,
        tokens: &[u32],
        logits_out: &mut [f32],
        ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let cfg = &self.config;
        let (d, vocab) = (cfg.d_model, cfg.vocab_size);
        let m = tokens.len();
        let (Some(rms_norm), Some(matmul)) = (self.table.rms_norm.0, self.table.matmul.0) else {
            return Err(err("dispatch table incomplete for batched prefill"));
        };
        let mut guard = self.state.lock().map_err(|_| err("state lock"))?;
        let st = &mut *guard;
        let dev = st.stages[0].device;

        // Batched residual stream and norm buffer, one column per token.
        let mut hbuf = DeviceBuffer::alloc_on(self.backend, dev, d * m * 4)
            .map_err(|e| err(format!("alloc: {e}")))?;
        let nbuf = DeviceBuffer::alloc_on(self.backend, dev, d * m * 4)
            .map_err(|e| err(format!("alloc: {e}")))?;

        // Token embeddings, token-major host buffer [d, m].
        let mut host = vec![0f32; d * m];
        for (t, &tok) in tokens.iter().enumerate() {
            self.embed(tok, &mut host[t * d..(t + 1) * d])?;
        }
        hbuf.upload(f32_bytes(&host))
            .map_err(|e| err(e.to_string()))?;

        let mut h_t = f32_tensor2(&hbuf, d, m);
        let mut n_t = f32_tensor2(&nbuf, d, m);
        // Batched scratch: the single-token size times m bounds every batched op
        // (the per-token fixed part scales with m; the workspace is <= its *m).
        let scratch_floats = [
            2 * cfg.d_ffn,
            self.attn
                .scratch_floats(cfg.n_heads, cfg.n_kv_heads, self.n_ctx),
            self.gdn.scratch_floats(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let scratch = DeviceBuffer::alloc_on(self.backend, dev, (scratch_floats * 4 * m).max(1))
            .map_err(|e| err(format!("alloc scratch: {e}")))?;
        let kctx = SpiteCtx {
            n_ctx: self.n_ctx as c_int,
            n_batch: m as c_int,
            n_threads: ctx.n_threads,
            pos: ctx.pos,
            n_heads: cfg.n_heads as c_int,
            n_kv_heads: cfg.n_kv_heads as c_int,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: scratch.as_ptr().cast(),
            scratchpad_bytes: scratch.size,
        };

        for (li, layer) in self.layers.iter().enumerate() {
            let ffn_w = match layer {
                Layer::Attn(a) => {
                    self.attn_block(st, a, li, &mut h_t, &mut n_t, &kctx)?;
                    &a.ffn
                }
                Layer::Gdn(g) => {
                    self.gdn_block(st, g, li, &mut h_t, &mut n_t, &kctx)?;
                    &g.ffn
                }
            };
            self.ffn_block(ffn_w, li, &mut h_t, &mut n_t, &kctx)?;
        }

        // Final norm + LM head, one column at a time (logits are per token).
        let mut hid = f32_tensor(&st.hid, d);
        let mut logits = f32_tensor(&st.logits, vocab);
        for t in 0..m {
            let ht = f32_tensor_col(&hbuf, d, t);
            // SAFETY: every tensor points at live memory owned by `st` / `self`.
            unsafe {
                rc(
                    rms_norm(&mut hid, &ht, &self.out_norm, cfg.norm_eps, &kctx),
                    "rms_norm",
                    cfg.n_layers,
                )?;
                rc(
                    matmul(&mut logits, &hid, &self.out_w, &kctx),
                    "matmul",
                    cfg.n_layers,
                )?;
            }
            st.logits
                .download(f32_bytes_mut(&mut logits_out[t * vocab..(t + 1) * vocab]))
                .map_err(|e| err(e.to_string()))?;
        }
        Ok(())
    }

    /// Stage `s`'s residual stream, norm buffer and kernel context.
    fn stage_view(
        &self,
        st: &HState,
        s: usize,
        pos: usize,
        n_threads: c_int,
    ) -> (SpiteTensor, SpiteTensor, SpiteCtx) {
        let d = self.config.d_model;
        let sb = &st.stages[s];
        let kctx = SpiteCtx {
            n_ctx: self.n_ctx as c_int,
            n_batch: 1,
            n_threads,
            pos: pos as c_int,
            n_heads: self.config.n_heads as c_int,
            n_kv_heads: self.config.n_kv_heads as c_int,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: sb.scratch.as_ptr().cast(),
            scratchpad_bytes: sb.scratch.size,
        };
        (f32_tensor(&sb.h, d), f32_tensor(&sb.n, d), kctx)
    }

    /// Make stage `s`'s device current, so its kernels launch there.
    fn enter(&self, st: &HState, s: usize) -> Result<(), ModelError> {
        if self.backend == GpuBackend::Cuda {
            cuda::set_device(st.stages[s].device).map_err(|e| err(e.to_string()))?;
        }
        Ok(())
    }

    /// Move the residual stream from stage `from` to stage `to` through
    /// `host`, then make `to` current.
    ///
    /// The blocking download waits for `from`'s kernels; the upload is
    /// ordered before `to`'s kernels on its legacy default stream. No peer
    /// access is needed.
    fn hand_off(
        &self,
        st: &mut HState,
        from: usize,
        to: usize,
        host: &mut [f32],
    ) -> Result<(), ModelError> {
        st.stages[from]
            .h
            .download(f32_bytes_mut(host))
            .map_err(|e| err(e.to_string()))?;
        st.stages[to]
            .h
            .upload(f32_bytes(host))
            .map_err(|e| err(e.to_string()))?;
        self.enter(st, to)
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

    /// `h += linear_attn(rms_norm(h))` for one Gated Delta Net block.
    ///
    /// `h`/`n` are `[d, 1]` for one token or `[d, m]` for a batch of columns;
    /// the conv history and delta-rule state carry across the columns in order.
    fn gdn_block(
        &self,
        st: &HState,
        g: &GdnLayer,
        li: usize,
        h_t: &mut SpiteTensor,
        n_t: &mut SpiteTensor,
        kctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let (Some(rms_norm), Some(linear_attn)) = (self.table.rms_norm.0, self.table.linear_attn.0)
        else {
            return Err(err("dispatch table incomplete for hybrid decoder"));
        };
        let gs = &st.gdn[g.st];
        let mut conv_hist = f32_tensor(&gs.conv_hist, self.gdn.conv_hist_floats());
        let mut state = f32_tensor(&gs.state, self.gdn.state_floats());
        // SAFETY: every tensor points at live memory owned by `st` / `self`;
        // the kernel ABI version is checked at load.
        unsafe {
            rc(
                rms_norm(n_t, h_t, &g.norm, self.config.norm_eps, kctx),
                "rms_norm",
                li,
            )?;
            rc(
                linear_attn(
                    h_t,
                    n_t,
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
                    kctx,
                ),
                "linear_attn",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn share_counts_largest_remainder() {
        assert_eq!(share_counts(&[33, 32], 65), Some(vec![33, 32]));
        assert_eq!(
            share_counts(&[1, 1, 1, 1, 1, 1], 65),
            Some(vec![11, 11, 11, 11, 11, 10])
        );
        assert_eq!(share_counts(&[20, 12], 64), Some(vec![40, 24]));
        assert_eq!(share_counts(&[1, 0], 4), Some(vec![4, 0]));
        assert_eq!(share_counts(&[0, 0], 4), None);
    }

    #[test]
    fn balance_tracks_bytes_not_counts() {
        // Every 4th layer is 3x heavier (KV); equal budgets must split bytes
        // evenly, not layer counts.
        let bytes: Vec<usize> = (0..64)
            .map(|l| if l % 4 == 3 { 300 } else { 100 })
            .collect();
        let counts = balance_layers(&bytes, &[1, 1]);
        assert_eq!(counts.iter().sum::<usize>(), 64);
        let first: usize = bytes[..counts[0]].iter().sum();
        let total: usize = bytes.iter().sum();
        assert!(first.abs_diff(total - first) <= 300, "{counts:?}");
    }

    #[test]
    fn balance_follows_budgets_and_keeps_all_layers() {
        let bytes = vec![10usize; 65];
        assert_eq!(balance_layers(&bytes, &[3, 1]), vec![49, 16]);
        assert_eq!(balance_layers(&bytes, &[1; 6]).iter().sum::<usize>(), 65);
        // No room anywhere: everything lands on the last stage, and the
        // per-device check reports it.
        assert_eq!(balance_layers(&bytes, &[0, 0]), vec![0, 65]);
    }

    #[test]
    fn auto_split_stays_on_one_device_when_it_fits() {
        let bytes = vec![1usize << 20; 8];
        let split = LayerSplit {
            devices: vec![0, 1],
            shares: Vec::new(),
        };
        let roomy = |_: usize| Ok(usize::MAX / 2);
        let (devs, counts) = plan_split(&split, &bytes, 0, 0, false, &roomy).unwrap();
        assert_eq!((devs, counts), (vec![0], vec![8]));

        // 8 MiB of layers plus a 3 MiB head; 6 MiB per device after headroom.
        // The head leaves the last device 3 MiB of layer budget: 8 * 6/9 -> 5/3.
        let tight = |_: usize| Ok(VRAM_HEADROOM + (6 << 20));
        let (devs, counts) = plan_split(&split, &bytes, 3 << 20, 0, false, &tight).unwrap();
        assert_eq!(devs, vec![0, 1]);
        assert_eq!(counts, vec![5, 3]);
    }

    /// Qwen3.8-27B Q6_K trunk (~18.6 GiB of weights): 48 GDN layers at
    /// ~293 MiB and 16 full-attention layers at ~284 MiB. Sizes are the real
    /// per-block tensor footprint from the GGUF, projected to 6.5625 bits/weight.
    fn qwen38_q6k_trunk() -> Vec<usize> {
        const MIB: usize = 1 << 20;
        (0..64)
            .map(|l| {
                if (l + 1) % 4 == 0 {
                    284 * MIB
                } else {
                    293 * MIB
                }
            })
            .collect()
    }

    #[test]
    fn two_p100_pcie_hold_qwen38_q6k() {
        // 2× P100-PCIE-16GB (PHB): the model does not fit one card, so it must
        // pipeline across both, every layer placed, nothing left on the floor.
        let layers = qwen38_q6k_trunk();
        let head = 324 * (1 << 20); // NextN/MTP block
        let split = LayerSplit {
            devices: vec![0, 1],
            shares: Vec::new(),
        };
        let free = VRAM_HEADROOM + 15 * (1 << 30);
        let free_on = |_: usize| Ok(free);
        let (devs, counts) = plan_split(&split, &layers, head, 0, false, &free_on).unwrap();
        assert_eq!(devs, vec![0, 1]);
        assert_eq!(counts.iter().sum::<usize>(), 64);
        assert!(counts.iter().all(|&c| c > 0));
    }

    #[test]
    fn four_p100_pcie_hold_qwen38_q6k_with_the_mtp_head() {
        // 4× P100-PCIE-16GB (2× PHB pairs, SYS between): same splitter, four
        // stages; the last stage carries the LM head and the MTP block.
        let layers = qwen38_q6k_trunk();
        let head = 324 * (1 << 20);
        let fixed = 2 * 5120 * 4; // per-stage h + n residual buffers
        let split = LayerSplit {
            devices: vec![0, 1, 2, 3],
            shares: Vec::new(),
        };
        let free = VRAM_HEADROOM + 15 * (1 << 30);
        let free_on = |_: usize| Ok(free);
        let (devs, counts) = plan_split(&split, &layers, head, fixed, false, &free_on).unwrap();
        assert_eq!(devs, vec![0, 1, 2, 3]);
        assert_eq!(
            counts.iter().sum::<usize>(),
            64,
            "every layer must be placed"
        );

        // Recompute each stage's footprint exactly as `load_split` does and
        // confirm it fits 16 GiB minus headroom.
        let mut first = 0;
        for (s, &c) in counts.iter().enumerate() {
            let bytes = layers[first..first + c].iter().sum::<usize>()
                + fixed
                + if s + 1 == counts.len() { head } else { 0 };
            assert!(
                bytes + VRAM_HEADROOM <= free,
                "stage {s} overflows: {} MiB",
                bytes >> 20
            );
            first += c;
        }
    }
}
