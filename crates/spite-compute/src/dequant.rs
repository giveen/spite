//! Block dequantization - any GGUF quantized format to F32.
//!
//! The implementation is `core/quant.c`, vendored from ggml and shared with the
//! C/CUDA kernels, so the host and every kernel decode a given type identically.

use crate::ComputeError;
use spite_abi::SpiteType;
use std::ffi::c_void;

unsafe extern "C" {
    fn spite_dequantize_row(kind: u32, src: *const c_void, dst: *mut f32, n: i64) -> i32;
}

/// Dequantize `n_elem` elements from `src` (packed blocks) into `dst` (F32).
pub fn dequant_to_f32(
    src: &[u8],
    kind: SpiteType,
    n_elem: usize,
    dst: &mut [f32],
) -> Result<(), ComputeError> {
    if dst.len() < n_elem {
        return Err(ComputeError::ShapeMismatch("dst too small".into()));
    }
    let blk = kind.block_elements() as usize;
    if !n_elem.is_multiple_of(blk) {
        return Err(ComputeError::ShapeMismatch(format!(
            "{n_elem} elements is not a multiple of the {kind:?} block size {blk}"
        )));
    }
    let need = n_elem / blk * kind.block_bytes() as usize;
    if src.len() < need {
        return Err(ComputeError::ShapeMismatch(format!(
            "{kind:?} src has {} bytes, need {need}",
            src.len()
        )));
    }
    // SAFETY: `src` holds at least `need` bytes (= n_elem/blk blocks), `dst` holds
    // at least `n_elem` floats, and the C side reads/writes exactly those ranges.
    let rc = unsafe {
        spite_dequantize_row(
            kind as u32,
            src.as_ptr().cast(),
            dst.as_mut_ptr(),
            n_elem as i64,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(ComputeError::UnsupportedDtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q8_0 block: f16 scale 0.5 (0x3800) + qs = -3..=28 -> exactly qs * 0.5.
    #[test]
    fn q8_0_block_decodes() {
        let mut src = vec![0x00, 0x38];
        src.extend((0..32).map(|i| (i as i8 - 3) as u8));
        let mut dst = [0f32; 32];
        dequant_to_f32(&src, SpiteType::Q8_0, 32, &mut dst).unwrap();
        for (i, v) in dst.iter().enumerate() {
            assert_eq!(*v, (i as f32 - 3.0) * 0.5);
        }
    }

    #[test]
    fn rejects_truncated_and_misaligned_input() {
        let mut dst = [0f32; 256];
        assert!(dequant_to_f32(&[0; 10], SpiteType::Q4K, 256, &mut dst).is_err());
        assert!(dequant_to_f32(&[0; 144], SpiteType::Q4K, 100, &mut dst).is_err());
    }

    /// Every type's layout agrees with what the C side consumes: decoding one
    /// zero-filled block must succeed (zero scales => all zeros).
    #[test]
    fn every_type_decodes_a_zero_block() {
        for kind in SpiteType::ALL {
            let blk = kind.block_elements() as usize;
            let src = vec![0u8; kind.block_bytes() as usize];
            let mut dst = vec![1f32; blk];
            dequant_to_f32(&src, kind, blk, &mut dst).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        }
    }
}
