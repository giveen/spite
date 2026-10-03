//! Block dequantization — GGUF quantized formats → F32.
//!
//! Each format stores weights in fixed-size blocks with per-block scales.
//! Reference: ggml/src/ggml-quants.c in the llama.cpp repository.

use spite_abi::SpiteType;
use crate::{ComputeError, f16_to_f32};

/// Dequantize `n_elem` elements from `src` (packed blocks) into `dst` (F32).
pub fn dequant_to_f32(
    src:    &[u8],
    kind:   SpiteType,
    n_elem: usize,
    dst:    &mut [f32],
) -> Result<(), ComputeError> {
    if dst.len() < n_elem { return Err(ComputeError::ShapeMismatch("dst too small".into())); }
    match kind {
        SpiteType::F32  => {
            let src_f32 = bytemuck_cast(src, n_elem)?;
            dst[..n_elem].copy_from_slice(src_f32);
            Ok(())
        }
        SpiteType::Q8_0 => dequant_q8_0(src, n_elem, dst),
        SpiteType::Q4_0 => dequant_q4_0(src, n_elem, dst),
        SpiteType::Q4K  => dequant_q4k(src, n_elem, dst),
        SpiteType::Q5K  => dequant_q5k(src, n_elem, dst),
        SpiteType::Q6K  => dequant_q6k(src, n_elem, dst),
        _ => Err(ComputeError::UnsupportedDtype),
    }
}

/// Q8_0 block: `{ u16 d; i8 qs[32] }` — 34 bytes per 32 elements.
fn dequant_q8_0(src: &[u8], n_elem: usize, dst: &mut [f32]) -> Result<(), ComputeError> {
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 2 + BLOCK; // d(u16) + 32×i8
    let n_blocks = n_elem / BLOCK;
    if src.len() < n_blocks * BLOCK_BYTES {
        return Err(ComputeError::ShapeMismatch("q8_0 src too small".into()));
    }
    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
        for i in 0..BLOCK {
            dst[b * BLOCK + i] = (src[off + 2 + i] as i8 as f32) * d;
        }
    }
    Ok(())
}

/// Q4_0 block: `{ u16 d; u8 qs[16] }` — 18 bytes per 32 elements.
fn dequant_q4_0(src: &[u8], n_elem: usize, dst: &mut [f32]) -> Result<(), ComputeError> {
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 2 + BLOCK / 2;
    let n_blocks = n_elem / BLOCK;
    if src.len() < n_blocks * BLOCK_BYTES {
        return Err(ComputeError::ShapeMismatch("q4_0 src too small".into()));
    }
    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
        for i in 0..16 {
            let byte = src[off + 2 + i];
            dst[b * BLOCK + i * 2]     = ((byte & 0x0F) as i32 - 8) as f32 * d;
            dst[b * BLOCK + i * 2 + 1] = ((byte >> 4)  as i32 - 8) as f32 * d;
        }
    }
    Ok(())
}

/// Q4_K block: 256 elements, per-superblock d/dmin, per-group scales.
fn dequant_q4k(_src: &[u8], _n_elem: usize, _dst: &mut [f32]) -> Result<(), ComputeError> {
    // TODO: block_q4_K { d: u16, dmin: u16, scales[12]: u8, qs[128]: u8 }
    //       256 elements per block, 8 groups of 32, each group has its own scale
    Ok(())
}

fn dequant_q5k(_src: &[u8], _n_elem: usize, _dst: &mut [f32]) -> Result<(), ComputeError> {
    // TODO: block_q5_K — like Q4_K with an extra high-bit array for the 5th bit
    Ok(())
}

fn dequant_q6k(_src: &[u8], _n_elem: usize, _dst: &mut [f32]) -> Result<(), ComputeError> {
    // TODO: block_q6_K — 256 elements, ql[128] (4-bit) + qh[64] (2-bit high)
    Ok(())
}

fn bytemuck_cast(src: &[u8], n: usize) -> Result<&[f32], ComputeError> {
    if src.len() < n * 4 { return Err(ComputeError::ShapeMismatch("f32 src too small".into())); }
    // SAFETY: alignment is caller's responsibility; this is a reference path only.
    let ptr = src.as_ptr() as *const f32;
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
}
