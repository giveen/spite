//! Q8_0 block quantization.
//!
//! Block format: `{ d: f16, qs: [i8; 32] }` — 34 bytes per 32 elements.
//! `d` is the block scale; each weight w = d × qs[i].
//!
//! This is the fastest quantization format — near-lossless, 8.5 bpw.

/// Bytes per block.
pub const BLOCK_SIZE: usize = 32;
pub const BLOCK_BYTES: usize = 2 + BLOCK_SIZE; // 2 bytes f16 scale + 32 i8

/// Quantize `n_elem` F32 values from `src` into `dst` (Q8_0 blocks).
///
/// `dst` must be at least `block_bytes(n_elem)` bytes.
pub fn quantize(src: &[f32], dst: &mut [u8], n_elem: usize) {
    let n_blocks = n_elem.div_ceil(BLOCK_SIZE);
    for b in 0..n_blocks {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(n_elem);
        let block = &src[start..end];

        // Find abs-max for the block scale.
        let amax = block.iter().cloned().map(f32::abs).fold(0f32, f32::max);
        let d = amax / 127.0;
        let d_inv = if d > 0.0 { 1.0 / d } else { 0.0 };

        let off = b * BLOCK_BYTES;
        let d_bits = f32_to_f16_bits(d);
        dst[off] = (d_bits & 0xFF) as u8;
        dst[off + 1] = (d_bits >> 8) as u8;

        for (i, &x) in block.iter().enumerate() {
            dst[off + 2 + i] = (x * d_inv).round().clamp(-128.0, 127.0) as i8 as u8;
        }
        // Pad last block with zeros if shorter than BLOCK_SIZE.
        for i in block.len()..BLOCK_SIZE {
            dst[off + 2 + i] = 0;
        }
    }
}

/// Required output bytes for `n_elem` elements.
pub fn block_bytes(n_elem: usize) -> usize {
    n_elem.div_ceil(BLOCK_SIZE) * BLOCK_BYTES
}

fn f32_to_f16_bits(x: f32) -> u16 {
    // Fast F32→F16 conversion using bit manipulation.
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        return sign as u16;
    }
    if exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    (sign | ((exp as u32) << 10) | mant) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dequant_q8_0_local(src: &[u8], n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        let block_size = 32;
        let block_bytes = 2 + block_size;
        for b in 0..(n / block_size) {
            let off = b * block_bytes;
            let d_bits = u16::from_le_bytes([src[off], src[off + 1]]);
            let d = {
                let sign = ((d_bits >> 15) as u32) << 31;
                let exp = ((d_bits >> 10) & 0x1F) as u32;
                let mant = (d_bits & 0x3FF) as u32;
                f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
            };
            for i in 0..block_size {
                out[b * block_size + i] = (src[off + 2 + i] as i8 as f32) * d;
            }
        }
        out
    }

    #[test]
    fn roundtrip() {
        let src: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let mut dst = vec![0u8; block_bytes(32)];
        quantize(&src, &mut dst, 32);
        let out = dequant_q8_0_local(&dst, 32);
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.5, "q8_0 roundtrip: {a} vs {b}");
        }
    }
}
