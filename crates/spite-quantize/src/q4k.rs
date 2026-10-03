//! Q4_K_M / Q4_K_S block quantization (K-quants).
//!
//! K-quants use a two-level scale hierarchy:
//!   super-block: 256 elements, two u16 scales (d, dmin)
//!   sub-block:   8 groups × 32 elements, each with its own 6-bit scale
//!
//! Block format (Q4_K): 144 bytes per 256 elements.
//!   d:      f16  (super-block scale)
//!   dmin:   f16  (super-block min scale, for the minimum)
//!   scales: u8[12] (packed 6-bit per-group scales, 8 groups × 2 values)
//!   qs:     u8[128] (4-bit packed weights: two 4-bit values per byte)
//!
//! "Mixed" (Q4_K_M) assigns Q6_K to every 6th layer (attention output + FFN
//! down-proj) and Q4_K_M to the rest. The scheduler/quantize_model decides
//! the per-tensor strategy; this module only does the math.

pub const BLOCK_SIZE:  usize = 256;
pub const BLOCK_BYTES: usize = 2 + 2 + 12 + 128; // 144 bytes

/// Quantize `n_elem` F32 values into Q4_K blocks.
///
/// Each super-block of 256 elements is split into 8 groups of 32.
/// Groups are quantized independently with shared super-block d/dmin.
pub fn quantize_q4k(src: &[f32], dst: &mut [u8], n_elem: usize) {
    let n_blocks = n_elem.div_ceil(BLOCK_SIZE);
    for b in 0..n_blocks {
        let start = b * BLOCK_SIZE;
        let end   = (start + BLOCK_SIZE).min(n_elem);
        let block = &src[start..end];

        let (d, dmin, scales, qs) = encode_block_q4k(block);

        let off = b * BLOCK_BYTES;
        let d_bits    = f32_to_f16(d);
        let dmin_bits = f32_to_f16(dmin);
        dst[off]     = (d_bits    & 0xFF) as u8;
        dst[off + 1] = (d_bits    >> 8)   as u8;
        dst[off + 2] = (dmin_bits & 0xFF) as u8;
        dst[off + 3] = (dmin_bits >> 8)   as u8;
        dst[off + 4  ..off + 16 ].copy_from_slice(&scales);
        dst[off + 16 ..off + 144].copy_from_slice(&qs);
    }
}

fn encode_block_q4k(block: &[f32]) -> (f32, f32, [u8; 12], [u8; 128]) {
    // Find global max and min across the super-block.
    let max = block.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = block.iter().cloned().fold(f32::INFINITY,     f32::min);
    let min = min.min(0.0); // clamp min to ≤ 0

    let d    = (max - min) / 15.0;
    let dmin = -min / 15.0;
    let d_inv    = if d    > 0.0 { 1.0 / d    } else { 0.0 };
    let dmin_inv = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

    let mut scales = [0u8; 12];
    let mut qs     = [0u8; 128];

    // 8 sub-groups of 32 elements
    for g in 0..8usize {
        let gs = g * 32;
        let ge = gs.saturating_add(32).min(block.len());
        let group = &block[gs..ge];

        let gmax = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let gmin = group.iter().cloned().fold(f32::INFINITY,     f32::min).min(0.0);

        let sc    = ((gmax - gmin) * d_inv).round().clamp(0.0, 63.0) as u8;
        let sc_m  = (-gmin * dmin_inv).round().clamp(0.0, 63.0) as u8;

        // Pack two 6-bit scales into the scales array (3 bytes per 2 groups).
        // Simplified: store lower 4 bits in byte layout — full K-quant packing
        // is more complex; this is a readable stub.
        let byte_idx = g * 3 / 2;
        if g % 2 == 0 {
            scales[byte_idx] = sc | ((sc_m & 0xF) << 4);
        } else {
            scales[byte_idx + 1] = sc_m >> 4;
        }

        // Quantize group elements to 4 bits.
        let group_d    = if sc    > 0 { (gmax - gmin) / sc    as f32 } else { 0.0 };
        let group_dmin = if sc_m  > 0 { -gmin          / sc_m as f32 } else { 0.0 };

        for i in 0..ge.saturating_sub(gs) {
            let x = group[i];
            let q = if group_d > 0.0 { ((x - gmin) / group_d).round().clamp(0.0, 15.0) as u8 }
                    else { 0 };
            let qs_idx = (gs + i) / 2;
            if (gs + i) % 2 == 0 {
                qs[qs_idx] = q;
            } else {
                qs[qs_idx] |= q << 4;
            }
        }
        let _ = group_dmin; // used in dequant path
    }

    (d, dmin, scales, qs)
}

pub fn block_bytes(n_elem: usize) -> usize {
    n_elem.div_ceil(BLOCK_SIZE) * BLOCK_BYTES
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp  = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0  { return sign as u16; }
    if exp >= 31 { return (sign | 0x7C00) as u16; }
    (sign | ((exp as u32) << 10) | mant) as u16
}
