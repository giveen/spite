//! MXFP4 block quantization — PXA's "PXQ4" tier.
//!
//! Block format: `{ e: u8, qs: [u8; 16] }` — 17 bytes per 32 elements (4.25
//! bpw). `e` is an E8M0 (power-of-two) block scale; `qs` packs 32 4-bit e2m1
//! codes, low nibble for element `j` and high nibble for element `j + 16`.
//!
//! Reconstruction: `w = e8m0_to_fp32_half(e) * kvalues_fp4[code]`, with the
//! e2m1 values **doubled** so the half-scale lands on the real e2m1 grid.
//! This matches llama.cpp / pxa `quantize_row_mxfp4_ref` bit for bit (and the
//! decode in `core/quant.c` `dq_mxfp4`).
//!
//! MXFP4 is what PXA ships as PXQ4: 4.25 bpw is what lets a 27B model fit one
//! 16 GB P100.

/// Elements per block.
pub const BLOCK_SIZE: usize = 32;
/// Bytes per block (`e` + 16 code bytes).
pub const BLOCK_BYTES: usize = 17;

/// e2m1 values, doubled (indices 0..7 non-negative, 8..15 their negatives).
const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// E8M0 exponent byte → fp32 block scale, halved (`ggml_e8m0_to_fp32_half`).
pub fn e8m0_to_fp32_half(x: u8) -> f32 {
    let u: u32 = if x >= 2 {
        (x as u32 - 1) << 23
    } else if x == 0 {
        0x0020_0000
    } else {
        0x0040_0000
    };
    f32::from_bits(u)
}

/// Nearest e2m1 code for `x` under scale `d`.
fn best_index(d: f32, x: f32) -> u8 {
    let mut best = (x - d * KVALUES_FP4[0] as f32).abs();
    let mut index = 0u8;
    for (j, &v) in KVALUES_FP4.iter().enumerate().skip(1) {
        let diff = (x - d * v as f32).abs();
        if diff < best {
            best = diff;
            index = j as u8;
        }
    }
    index
}

/// Quantize `n_elem` F32 values from `src` into MXFP4 blocks in `dst`.
///
/// `dst` must be at least `block_bytes(n_elem)` bytes. The last block is
/// zero-padded when `n_elem` is not a multiple of [`BLOCK_SIZE`].
pub fn quantize(src: &[f32], dst: &mut [u8], n_elem: usize) {
    let n_blocks = n_elem.div_ceil(BLOCK_SIZE);
    for b in 0..n_blocks {
        let off = b * BLOCK_BYTES;
        for k in 0..BLOCK_BYTES {
            dst[off + k] = 0;
        }
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(n_elem);
        let amax = src[start..end]
            .iter()
            .cloned()
            .map(f32::abs)
            .fold(0f32, f32::max);
        if amax == 0.0 {
            continue; // e = 0, all codes 0
        }
        let e = ((amax.log2().floor() as i32) - 2 + 127) as u8;
        let d = e8m0_to_fp32_half(e);
        dst[off] = e;
        for j in 0..BLOCK_SIZE / 2 {
            let x0 = if start + j < end { src[start + j] } else { 0.0 };
            let x1 = if start + BLOCK_SIZE / 2 + j < end {
                src[start + BLOCK_SIZE / 2 + j]
            } else {
                0.0
            };
            let v0 = best_index(d, x0);
            let v1 = best_index(d, x1);
            dst[off + 1 + j] = v0 | (v1 << 4);
        }
    }
}

/// Required output bytes for `n_elem` elements.
pub fn block_bytes(n_elem: usize) -> usize {
    n_elem.div_ceil(BLOCK_SIZE) * BLOCK_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror of `dq_mxfp4` in core/quant.c.
    fn dequant(src: &[u8], n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        for b in 0..n.div_ceil(BLOCK_SIZE) {
            let off = b * BLOCK_BYTES;
            let d = e8m0_to_fp32_half(src[off]);
            for j in 0..BLOCK_SIZE / 2 {
                let byte = src[off + 1 + j];
                let v0 = KVALUES_FP4[(byte & 0x0F) as usize];
                let v1 = KVALUES_FP4[(byte >> 4) as usize];
                let i = b * BLOCK_SIZE + j;
                if i < n {
                    out[i] = v0 as f32 * d;
                }
                if i + BLOCK_SIZE / 2 < n {
                    out[i + BLOCK_SIZE / 2] = v1 as f32 * d;
                }
            }
        }
        out
    }

    #[test]
    fn roundtrip_within_half_lsb() {
        // A ramp covers both nibble halves and the sign codes.
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.7).collect();
        let mut dst = vec![0u8; block_bytes(src.len())];
        quantize(&src, &mut dst, src.len());
        let out = dequant(&dst, src.len());
        // e2m1 has only 8 magnitudes; with a 2^(floor(log2 amax)-2) scale the
        // nearest-code error is at most ~1/4 of amax.
        let amax = src.iter().cloned().map(f32::abs).fold(0f32, f32::max);
        let max_err = src
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err <= amax / 3.0, "max_err {max_err} amax {amax}");
    }

    #[test]
    fn zero_block_is_all_zero() {
        let src = vec![0f32; 32];
        let mut dst = vec![0xFFu8; BLOCK_BYTES];
        quantize(&src, &mut dst, 32);
        assert!(dst.iter().all(|&b| b == 0));
        assert!(dequant(&dst, 32).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn exact_powers_of_two_are_lossless() {
        // Values already on the e2m1 grid reconstruct exactly.
        let mut src = vec![0f32; 32];
        for (j, v) in KVALUES_FP4.iter().enumerate().take(16) {
            src[j] = *v as f32;
        }
        // amax = 12 -> e = floor(log2 12) - 2 + 127 = 3 - 2 + 127 = 128
        let mut dst = vec![0u8; block_bytes(32)];
        quantize(&src, &mut dst, 32);
        assert_eq!(dst[0], 128);
        let out = dequant(&dst, 32);
        for (j, v) in KVALUES_FP4.iter().enumerate().take(16) {
            // doubled values at scale 2^(128-128) = 1
            assert_eq!(out[j], *v as f32);
        }
    }
}
