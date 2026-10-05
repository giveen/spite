//! Block codecs for KV-cache quantization.
//!
//! The KV cache stores one row per token position; each row is the attention
//! key (or value) for every KV head at that position, flattened. These codecs
//! pack a row of `f32` into the fixed-size blocks GGML uses for the same
//! formats, so a quantized KV cache is byte-compatible with the kernels that
//! expect `q8_0` / `q5_1` / `q4_0` layouts.
//!
//! Block layouts (32 elements per block):
//!
//! | tier  | bytes/block | layout                                            |
//! |-------|-------------|---------------------------------------------------|
//! | q8_0  | 34          | `f16 d; i8 qs[32]`                                |
//! | q5_1  | 24          | `f16 d; f16 m; u32 qh; u8 qs[16]`                 |
//! | q4_0  | 18          | `f16 d; u8 qs[16]`                                |
//!
//! `f16` and `f32` are element-wise and carry no block overhead.

use crate::KvQuant;

/// Elements per quantized block.
pub const BLOCK: usize = 32;

/// Packed byte length of `n_elem` elements stored at `q`.
///
/// For `f32`/`f16` this is exact; for block types it rounds up to a whole
/// number of blocks, matching GGML's allocation for a tensor of that length.
pub fn packed_bytes(q: KvQuant, n_elem: usize) -> usize {
    match q {
        KvQuant::F32 => n_elem * 4,
        KvQuant::F16 => n_elem * 2,
        _ => n_elem.div_ceil(BLOCK) * block_bytes(q),
    }
}

/// Bytes per 32-element block for a block-quantized tier.
fn block_bytes(q: KvQuant) -> usize {
    match q {
        KvQuant::Q8 => 34,   // f16 d + 32×i8
        KvQuant::Q5_1 => 24, // f16 d + f16 m + u32 qh + 16×u8
        KvQuant::Q4 => 18,   // f16 d + 16×u8
        _ => unreachable!("block_bytes called on a non-block tier"),
    }
}

/// Quantize `src` into `dst`.
///
/// `dst` must be at least `packed_bytes(q, src.len())` bytes; padded bytes of
/// a trailing partial block are zero-filled.
pub fn quantize(q: KvQuant, src: &[f32], dst: &mut [u8]) {
    debug_assert!(dst.len() >= packed_bytes(q, src.len()));
    match q {
        KvQuant::F32 => {
            for (i, &x) in src.iter().enumerate() {
                dst[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        KvQuant::F16 => {
            for (i, &x) in src.iter().enumerate() {
                dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_f16(x).to_le_bytes());
            }
        }
        KvQuant::Q8 => quant_q8(src, dst),
        KvQuant::Q5_1 => quant_q5_1(src, dst),
        KvQuant::Q4 => quant_q4(src, dst),
    }
}

/// Dequantize `n_elem` elements from `src` into `dst` (`dst.len() >= n_elem`).
pub fn dequantize(q: KvQuant, src: &[u8], n_elem: usize, dst: &mut [f32]) {
    debug_assert!(dst.len() >= n_elem);
    match q {
        KvQuant::F32 => {
            for (i, d) in dst.iter_mut().take(n_elem).enumerate() {
                *d = f32::from_le_bytes([
                    src[i * 4],
                    src[i * 4 + 1],
                    src[i * 4 + 2],
                    src[i * 4 + 3],
                ]);
            }
        }
        KvQuant::F16 => {
            for (i, d) in dst.iter_mut().take(n_elem).enumerate() {
                *d = f16_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]));
            }
        }
        KvQuant::Q8 => dequant_q8(src, n_elem, dst),
        KvQuant::Q5_1 => dequant_q5_1(src, n_elem, dst),
        KvQuant::Q4 => dequant_q4(src, n_elem, dst),
    }
}

// ── q8_0 ───────────────────────────────────────────────────────────────────

fn quant_q8(src: &[f32], dst: &mut [u8]) {
    const BB: usize = 34;
    for b in 0..src.len().div_ceil(BLOCK) {
        let blk = &src[b * BLOCK..(b * BLOCK + BLOCK).min(src.len())];
        let amax = blk.iter().map(|x| x.abs()).fold(0f32, f32::max);
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let off = b * BB;
        dst[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for i in 0..BLOCK {
            let v = blk
                .get(i)
                .map_or(0, |&x| (x * inv).round().clamp(-128.0, 127.0) as i8);
            dst[off + 2 + i] = v as u8;
        }
    }
}

fn dequant_q8(src: &[u8], n_elem: usize, dst: &mut [f32]) {
    const BB: usize = 34;
    for b in 0..n_elem.div_ceil(BLOCK) {
        let off = b * BB;
        let d = f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
        for i in 0..BLOCK {
            let idx = b * BLOCK + i;
            if idx >= n_elem {
                break;
            }
            dst[idx] = (src[off + 2 + i] as i8 as f32) * d;
        }
    }
}

// ── q5_1 ───────────────────────────────────────────────────────────────────

fn quant_q5_1(src: &[f32], dst: &mut [u8]) {
    const BB: usize = 24;
    for b in 0..src.len().div_ceil(BLOCK) {
        let blk = &src[b * BLOCK..(b * BLOCK + BLOCK).min(src.len())];
        let min = blk.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = blk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let d = (max - min) / 31.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let off = b * BB;
        dst[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        dst[off + 2..off + 4].copy_from_slice(&f32_to_f16(min).to_le_bytes());

        let mut qh: u32 = 0;
        let mut qs = [0u8; 16];
        for i in 0..BLOCK {
            let q = blk
                .get(i)
                .map_or(0, |&x| ((x - min) * inv).round().clamp(0.0, 31.0) as u32);
            if i < 16 {
                qs[i] |= (q & 0x0F) as u8;
            } else {
                qs[i - 16] |= ((q & 0x0F) as u8) << 4;
            }
            if q & 0x10 != 0 {
                qh |= 1 << i;
            }
        }
        dst[off + 4..off + 8].copy_from_slice(&qh.to_le_bytes());
        dst[off + 8..off + 24].copy_from_slice(&qs);
    }
}

fn dequant_q5_1(src: &[u8], n_elem: usize, dst: &mut [f32]) {
    const BB: usize = 24;
    for b in 0..n_elem.div_ceil(BLOCK) {
        let off = b * BB;
        let d = f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([src[off + 2], src[off + 3]]));
        let qh = u32::from_le_bytes([src[off + 4], src[off + 5], src[off + 6], src[off + 7]]);
        for i in 0..BLOCK {
            let idx = b * BLOCK + i;
            if idx >= n_elem {
                break;
            }
            let byte = src[off + 8 + (i % 16)];
            let lo = if i < 16 { byte & 0x0F } else { byte >> 4 };
            let q = (lo as u32) | (((qh >> i) & 1) << 4);
            dst[idx] = q as f32 * d + m;
        }
    }
}

// ── q4_0 ───────────────────────────────────────────────────────────────────

fn quant_q4(src: &[f32], dst: &mut [u8]) {
    const BB: usize = 18;
    for b in 0..src.len().div_ceil(BLOCK) {
        let blk = &src[b * BLOCK..(b * BLOCK + BLOCK).min(src.len())];
        let amax = blk.iter().map(|x| x.abs()).fold(0f32, f32::max);
        let d = amax / 8.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let off = b * BB;
        dst[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        let encode = |x: f32| -> u8 {
            let q = (x * inv).round().clamp(-8.0, 7.0) as i32;
            ((q + 8) & 0x0F) as u8
        };
        for i in 0..16 {
            let lo = blk.get(i).map_or(8, |&x| encode(x));
            let hi = blk.get(i + 16).map_or(8, |&x| encode(x));
            dst[off + 2 + i] = lo | (hi << 4);
        }
    }
}

fn dequant_q4(src: &[u8], n_elem: usize, dst: &mut [f32]) {
    const BB: usize = 18;
    for b in 0..n_elem.div_ceil(BLOCK) {
        let off = b * BB;
        let d = f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
        for i in 0..16 {
            let byte = src[off + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            let i0 = b * BLOCK + i;
            let i1 = b * BLOCK + i + 16;
            if i0 < n_elem {
                dst[i0] = lo as f32 * d;
            }
            if i1 < n_elem {
                dst[i1] = hi as f32 * d;
            }
        }
    }
}

// ── f16 helpers ────────────────────────────────────────────────────────────

/// Round-to-nearest-even f32 → f16 bit pattern.
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let man = b & 0x007F_FFFF;
    if b & 0x7F80_0000 == 0x7F80_0000 {
        // Inf/NaN: preserve payload class.
        return sign | 0x7C00 | if man != 0 { 0x0200 } else { 0 };
    }
    if exp >= 0x1F {
        return sign | 0x7C00; // overflow saturates to +inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflows to signed zero
        }
        let man = man | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let h = (man >> shift) as u16;
        let rem = man & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let round = (rem > half) || (rem == half && (h & 1) == 1);
        return sign | (h + round as u16);
    }
    let h = (((exp as u32) << 10) | (man >> 13)) as u16;
    let rem = man & 0x1FFF;
    let round = (rem > 0x1000) || (rem == 0x1000 && (h & 1) == 1);
    sign | (h + round as u16)
}

/// f16 bit pattern → f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let man = (bits & 0x3FF) as u32;
    match exp {
        0 => {
            if man == 0 {
                f32::from_bits(sign)
            } else {
                // Subnormal f16: value = man * 2^-24. `man <= 1023` and the
                // 2^-24 scale are both exact in f32, so the product rounds to
                // nothing and matches CUDA's `__half2float` bit-for-bit.
                let v = (man as f32) * f32::from_bits(0x3380_0000);
                if sign != 0 { -v } else { v }
            }
        }
        0x1F => f32::from_bits(sign | 0x7F80_0000 | (man << 13)),
        _ => f32::from_bits(sign | ((exp + 127 - 15) << 23) | (man << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(q: KvQuant, tol: f32) {
        // A representative row: smooth ramp plus a spike, like attention K/V.
        let src: Vec<f32> = (0..128)
            .map(|i| ((i as f32) * 0.07).sin() * 3.0 + if i == 63 { 10.0 } else { 0.0 })
            .collect();
        let mut packed = vec![0u8; packed_bytes(q, src.len())];
        quantize(q, &src, &mut packed);
        let mut back = vec![0f32; src.len()];
        dequantize(q, &packed, src.len(), &mut back);
        for (a, b) in src.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() <= tol,
                "{q:?} roundtrip {a} vs {b} (tol {tol})"
            );
        }
    }

    #[test]
    fn roundtrip_f32_is_lossless() {
        roundtrip(KvQuant::F32, 0.0);
    }

    #[test]
    fn roundtrip_f16() {
        roundtrip(KvQuant::F16, 0.01);
    }

    #[test]
    fn roundtrip_q8() {
        roundtrip(KvQuant::Q8, 0.06);
    }

    #[test]
    fn roundtrip_q5_1() {
        roundtrip(KvQuant::Q5_1, 0.5);
    }

    #[test]
    fn roundtrip_q4() {
        roundtrip(KvQuant::Q4, 1.5);
    }

    #[test]
    fn packed_bytes_matches_tier_footprints() {
        assert_eq!(packed_bytes(KvQuant::F32, 32), 128);
        assert_eq!(packed_bytes(KvQuant::F16, 32), 64);
        assert_eq!(packed_bytes(KvQuant::Q8, 32), 34);
        assert_eq!(packed_bytes(KvQuant::Q5_1, 32), 24);
        assert_eq!(packed_bytes(KvQuant::Q4, 32), 18);
        // Partial final block still allocates a full block.
        assert_eq!(packed_bytes(KvQuant::Q8, 33), 68);
    }

    #[test]
    fn f16_subnormals_roundtrip_exactly() {
        // Subnormal f16 patterns used to decode 2^14x too large, which
        // corrupted the whole cache whenever a block's scale or a K/V value
        // fell below 2^-14. Check the exact decode against IEEE-754:
        // subnormal f16 = mantissa * 2^-24.
        let mut checked = 0;
        for man in 1u16..0x400 {
            let bits = man; // exp == 0, sign == 0
            let expect = (man as f32) * 2f32.powi(-24);
            let got = f16_to_f32(bits);
            assert_eq!(got, expect, "subnormal {bits:#06x}");
            assert_eq!(f32_to_f16(expect), bits, "re-encode {bits:#06x}");
            checked += 1;
        }
        assert_eq!(checked, 1023);
    }

    #[test]
    fn q8_scale_in_f16_subnormal_range_still_decodes() {
        // A block whose scale `d = amax/127` lands in the f16 subnormal range
        // (2^-24 <= d < 2^-14) must decode at the right magnitude rather than
        // being amplified by 2^14.
        let src: Vec<f32> = (0..32).map(|i| ((i as f32) - 15.5) * 6e-6).collect();
        let mut packed = vec![0u8; packed_bytes(KvQuant::Q8, src.len())];
        quantize(KvQuant::Q8, &src, &mut packed);
        let d = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
        assert!(d > 0.0 && d < 2f32.powi(-14), "scale {d} not subnormal");
        let mut back = vec![0f32; src.len()];
        dequantize(KvQuant::Q8, &packed, src.len(), &mut back);
        for (a, b) in src.iter().zip(back.iter()) {
            assert!((a - b).abs() <= 1e-5, "q8 tiny block {a} vs {b}");
        }
    }

    #[test]
    fn q5_1_scale_and_min_are_recovered() {
        // Constant block: d == 0, min == value, all q == 0.
        let src = vec![1.25f32; 32];
        let mut packed = vec![0u8; packed_bytes(KvQuant::Q5_1, 32)];
        quantize(KvQuant::Q5_1, &src, &mut packed);
        let mut back = vec![0f32; 32];
        dequantize(KvQuant::Q5_1, &packed, 32, &mut back);
        for b in back {
            assert!((b - 1.25).abs() < 0.01);
        }
    }
}
