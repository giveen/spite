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
}

impl QuantType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Q8_0 => "Q8_0",
            Self::Q4_0 => "Q4_0",
            Self::Q4KM => "Q4_K_M",
            Self::Q4KS => "Q4_K_S",
            Self::Q5KM => "Q5_K_M",
            Self::Q6K  => "Q6_K",
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
            Self::Q6K  => 6.57,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Target quantization type for most tensors.
    pub target:    QuantType,
    pub n_threads: usize,
    /// Tensor names (prefix match) to keep in F32 or F16 (embeddings, output head).
    pub keep_f32:  Vec<String>,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            target:   QuantType::Q4KM,
            n_threads: 4,
            keep_f32:  vec!["token_embd".into(), "output.weight".into()],
        }
    }
}

/// Quantize `n_elem` F32 values from `src` into `dst` using `kind`.
///
/// `dst` must be pre-allocated to the correct block-packed byte size.
pub fn quantize_f32(
    _src:    &[f32],
    _dst:    &mut [u8],
    _kind:   QuantType,
    _n_elem: usize,
) -> Result<(), QuantizeError> {
    // TODO: dispatch to quantize_q8_0 / quantize_q4k / etc.
    Err(QuantizeError::UnsupportedSource("quantization not yet implemented".into()))
}

/// Read `src_path` (F32 GGUF), quantize tensors per `cfg`, write `dst_path`.
pub fn quantize_model(
    _src_path: &Path,
    _dst_path: &Path,
    _cfg:      &QuantizeConfig,
) -> Result<(), QuantizeError> {
    // TODO:
    // 1. Open src GGUF with spite-loader
    // 2. Write GGUF header + all metadata verbatim
    // 3. For each tensor:
    //    - if name matches keep_f32 prefix → copy as F32
    //    - else: dequant to F32 if needed → quantize_f32(data, buf, cfg.target)
    // 4. Write tensor index + data section
    Err(QuantizeError::Gguf("quantize_model not yet implemented".into()))
}

/// Required output buffer size in bytes for `n_elem` elements of `kind`.
pub fn block_bytes(kind: QuantType, n_elem: usize) -> usize {
    let (block_elems, block_bytes) = match kind {
        QuantType::Q8_0 => (32usize, 34usize),  // 2B scale + 32×i8
        QuantType::Q4_0 => (32, 18),             // 2B scale + 16×u8
        QuantType::Q4KM | QuantType::Q4KS => (256, 144), // 2+2+12+128 bytes
        QuantType::Q5KM => (256, 176),
        QuantType::Q6K  => (256, 210),
    };
    let n_blocks = n_elem.div_ceil(block_elems);
    n_blocks * block_bytes
}
