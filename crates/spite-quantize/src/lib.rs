//! GGUF quantization tooling.
//!
//! Converts F32/F16 models to quantized formats. The output is a new GGUF
//! file with updated tensor types and data sections; all metadata is
//! carried over verbatim.
//!
//! Supported target types:
//!   Q8_0   — 8-bit, fast, minimal quality loss
//!   Q4_0   — 4-bit, simple, lower quality
//!   Q4KM   — 4-bit K-quants mixed, best quality/size for 4-bit
//!   Q4KS   — 4-bit K-quants small, smaller than Q4KM
//!   Q5KM   — 5-bit K-quants mixed
//!   Q6K    — 6-bit K-quants, near-lossless

pub mod gguf_write;
pub mod mxfp4;
pub mod q4k;
pub mod q8_0;

use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantizeError {
    #[error("unsupported source dtype: {0}")]
    UnsupportedSource(String),
    #[error("tensor {tensor}: {msg}")]
    TensorError { tensor: String, msg: String },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("GGUF error: {0}")]
    Gguf(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    Q8_0,
    Q4_0,
    Q4KM,
    Q4KS,
    Q5KM,
    Q6K,
    /// PXA's PXQ4 tier: MXFP4 (32-element blocks, E8M0 scale, e2m1 codes).
    MXFP4,
}

impl QuantType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Q8_0 => "Q8_0",
            Self::Q4_0 => "Q4_0",
            Self::Q4KM => "Q4_K_M",
            Self::Q4KS => "Q4_K_S",
            Self::Q5KM => "Q5_K_M",
            Self::Q6K => "Q6_K",
            Self::MXFP4 => "MXFP4",
        }
    }

    /// Approximate bits per weight (including block overhead).
    pub fn bpw(self) -> f32 {
        match self {
            Self::Q8_0 => 8.5,
            Self::Q4_0 => 4.5,
            Self::Q4KM => 4.85,
            Self::Q4KS => 4.58,
            Self::Q5KM => 5.68,
            Self::Q6K => 6.57,
            Self::MXFP4 => 4.25,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Target quantization type for most tensors.
    pub target: QuantType,
    pub n_threads: usize,
    /// Tensor names (prefix match) to keep in F32 or F16 (embeddings, output head).
    pub keep_f32: Vec<String>,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            target: QuantType::Q4KM,
            n_threads: 4,
            keep_f32: vec!["token_embd".into(), "output.weight".into()],
        }
    }
}

/// Quantize `n_elem` F32 values from `src` into `dst` using `kind`.
///
/// `dst` must be pre-allocated to the correct block-packed byte size.
pub fn quantize_f32(
    src: &[f32],
    dst: &mut [u8],
    kind: QuantType,
    n_elem: usize,
) -> Result<(), QuantizeError> {
    match kind {
        QuantType::Q8_0 => {
            q8_0::quantize(src, dst, n_elem);
            Ok(())
        }
        QuantType::Q4KM | QuantType::Q4KS => {
            q4k::quantize_q4k(src, dst, n_elem);
            Ok(())
        }
        QuantType::MXFP4 => {
            mxfp4::quantize(src, dst, n_elem);
            Ok(())
        }
        _ => Err(QuantizeError::UnsupportedSource(format!(
            "{} not yet implemented",
            kind.name()
        ))),
    }
}

/// Read `src_path` (any supported GGUF), quantize tensors per `cfg`, write
/// `dst_path` as a new GGUF v3 file (metadata carried over verbatim).
///
/// Tensors whose name matches a `keep_f32` prefix are written as F32; the rest
/// are dequantized to F32 and requantized to `cfg.target`.
pub fn quantize_model(
    src_path: &Path,
    dst_path: &Path,
    cfg: &QuantizeConfig,
) -> Result<(), QuantizeError> {
    let model =
        spite_loader::GgufModel::open(src_path).map_err(|e| QuantizeError::Gguf(e.to_string()))?;

    let mut names: Vec<String> = model.tensor_names().map(str::to_owned).collect();
    names.sort(); // deterministic output

    let mut outs: Vec<gguf_write::OutTensor> = Vec::with_capacity(names.len());
    for name in &names {
        let t = model.tensor(name);
        let n: usize = t.ne.iter().map(|&x| x.max(1) as usize).product();
        let blk = t.kind.block_elements() as usize;
        if !n.is_multiple_of(blk) {
            return Err(QuantizeError::TensorError {
                tensor: name.clone(),
                msg: format!("{n} elements is not a multiple of the {blk}-element block"),
            });
        }
        let nbytes = n / blk * t.kind.block_bytes() as usize;
        // SAFETY: `t.data` points at `nbytes` of the loader's mmap.
        let src = unsafe { std::slice::from_raw_parts(t.data as *const u8, nbytes) };
        let mut f32buf = vec![0f32; n];
        spite_compute::dequant::dequant_to_f32(src, t.kind, n, &mut f32buf).map_err(|e| {
            QuantizeError::TensorError {
                tensor: name.clone(),
                msg: e.to_string(),
            }
        })?;

        let keep = cfg.keep_f32.iter().any(|p| name.starts_with(p.as_str()))
            // 1-D tensors are parameters the ops read as F32 (GDN dt/a/norm,
            // conv weights, RMS norm weights) and cannot be block-quantized;
            // a 2-D weight whose row length is not a multiple of the block is
            // equally unquantizable.
            || model.tensor_rank(name) < 2
            || !(t.ne[0] as usize).is_multiple_of(target_type(cfg.target).block_elements() as usize);
        let (kind, bytes) = if keep {
            (spite_abi::SpiteType::F32, f32_to_bytes(&f32buf))
        } else {
            let mut b = vec![0u8; block_bytes(cfg.target, n)];
            quantize_f32(&f32buf, &mut b, cfg.target, n)?;
            (target_type(cfg.target), b)
        };
        outs.push(gguf_write::OutTensor {
            name: name.clone(),
            ne: t.ne,
            ndim: model.tensor_rank(name).max(1),
            type_id: kind as u32,
            bytes,
        });
    }

    gguf_write::write(dst_path, &model.meta, &outs)?;
    Ok(())
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; v.len() * 4];
    for (i, x) in v.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
    b
}

/// The SpiteType (== GGUF type id) a `QuantType` writes.
fn target_type(q: QuantType) -> spite_abi::SpiteType {
    use spite_abi::SpiteType as T;
    match q {
        QuantType::Q8_0 => T::Q8_0,
        QuantType::Q4_0 => T::Q4_0,
        QuantType::Q4KM | QuantType::Q4KS => T::Q4K,
        QuantType::Q5KM => T::Q5K,
        QuantType::Q6K => T::Q6K,
        QuantType::MXFP4 => T::Mxfp4,
    }
}

/// Required output buffer size in bytes for `n_elem` elements of `kind`.
pub fn block_bytes(kind: QuantType, n_elem: usize) -> usize {
    let (block_elems, block_bytes) = match kind {
        QuantType::Q8_0 => (32usize, 34usize), // 2B scale + 32×i8
        QuantType::Q4_0 => (32, 18),           // 2B scale + 16×u8
        QuantType::Q4KM | QuantType::Q4KS => (256, 144), // 2+2+12+128 bytes
        QuantType::Q5KM => (256, 176),
        QuantType::Q6K => (256, 210),
        QuantType::MXFP4 => (32, 17), // PXA PXQ4
    };
    let n_blocks = n_elem.div_ceil(block_elems);
    n_blocks * block_bytes
}
