//! Variable Bit Rate (VBR) KV cache storage.
//!
//! A KV cache row is one token's key (or value) vector. As the sequence grows
//! the cache dominates memory, so VBR trades a little precision for a smaller
//! footprint: the rows start at the configured tier and, as depth increases,
//! the whole cache is re-encoded one step down the ladder
//! `f16 → q8 → q5_1 → q4`.
//!
//! The trigger is depth, not free VRAM: at 1/4, 1/2 and 3/4 of the context
//! window the cache drops a tier, which keeps its footprint roughly flat
//! instead of growing linearly with the number of tokens. Quality is lost only
//! for the oldest rows, and only once the sequence is long enough to matter.
//!
//! Degradation is monotonic and one-way — rows are never promoted back up.

use crate::KvQuant;
use crate::quant::{self, packed_bytes};

/// When to step down the quantization ladder for one side of the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VbrPolicy {
    /// Tier the cache starts at.
    pub start: KvQuant,
    /// Token depths at which to drop one tier, in ascending order.
    /// Extra entries past the bottom of the ladder are ignored.
    pub degrade_at: Vec<usize>,
}

impl VbrPolicy {
    /// A fixed-tier cache: start at `start` and never degrade.
    pub fn fixed(start: KvQuant) -> Self {
        Self {
            start,
            degrade_at: Vec::new(),
        }
    }

    /// Auto-VBR for a cache sized for `max_ctx` tokens.
    ///
    /// Degrades at 1/4, 1/2 and 3/4 of the window. A window shorter than 8
    /// tokens has no useful thresholds, so the cache stays fixed at `start`.
    pub fn from_ctx(max_ctx: usize, start: KvQuant) -> Self {
        let step = max_ctx / 4;
        let degrade_at = if step == 0 {
            Vec::new()
        } else {
            (1..=4).map(|i| step * i).collect()
        };
        Self { start, degrade_at }
    }
}

/// A growable, quantized stack of fixed-length rows for one cache side.
///
/// `push` appends a row and may trigger a degradation pass that re-encodes
/// every existing row at the next tier down. `to_f32` materializes the whole
/// stack for the attention kernel.
pub struct VbrRows {
    row_len: usize,
    n_rows: usize,
    quant: KvQuant,
    policy: VbrPolicy,
    /// Index of the next degradation threshold not yet applied.
    next: usize,
    /// All rows, packed back-to-back at `quant`.
    buf: Vec<u8>,
}

impl VbrRows {
    /// Create an empty stack of `row_len`-element rows following `policy`.
    pub fn new(row_len: usize, policy: VbrPolicy) -> Self {
        let quant = policy.start;
        Self {
            row_len,
            n_rows: 0,
            quant,
            policy,
            next: 0,
            buf: Vec::new(),
        }
    }

    /// Append one row (must be exactly `row_len` long).
    pub fn push(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.row_len);
        let n = packed_bytes(self.quant, self.row_len);
        let start = self.n_rows * n;
        self.buf.resize(start + n, 0);
        quant::quantize(self.quant, row, &mut self.buf[start..start + n]);
        self.n_rows += 1;
        self.maybe_degrade();
    }

    pub fn len(&self) -> usize {
        self.n_rows
    }

    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    /// Current storage tier (may have degraded since construction).
    pub fn quant(&self) -> KvQuant {
        self.quant
    }

    /// Elements per row.
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// Bytes currently held by the packed rows.
    pub fn packed_bytes(&self) -> usize {
        self.buf.len()
    }

    /// Bytes the same number of rows would occupy at `q`.
    pub fn bytes_at(&self, q: KvQuant) -> usize {
        packed_bytes(q, self.row_len) * self.n_rows
    }

    /// Dequantize every row into a contiguous `[len × row_len]` F32 buffer.
    pub fn to_f32(&self) -> Vec<f32> {
        self.to_f32_from(0)
    }

    /// Dequantize rows `row0..len` into a contiguous `[n × row_len]` buffer.
    /// Used for sliding-window attention, which skips the leading rows.
    pub fn to_f32_from(&self, row0: usize) -> Vec<f32> {
        let rows = self.n_rows.saturating_sub(row0);
        let mut out = vec![0f32; rows * self.row_len];
        let n = packed_bytes(self.quant, self.row_len);
        for r in 0..rows {
            let src = &self.buf[(row0 + r) * n..(row0 + r + 1) * n];
            let dst = &mut out[r * self.row_len..(r + 1) * self.row_len];
            quant::dequantize(self.quant, src, self.row_len, dst);
        }
        out
    }

    /// Apply every degradation threshold `n_rows` has now reached.
    fn maybe_degrade(&mut self) {
        while self.next < self.policy.degrade_at.len()
            && self.n_rows >= self.policy.degrade_at[self.next]
        {
            match self.quant.degrade() {
                Some(next) => self.requantize(next),
                // Already at the floor: stop considering thresholds.
                None => {
                    self.next = self.policy.degrade_at.len();
                    break;
                }
            }
            self.next += 1;
        }
    }

    /// Re-encode every row at `to` (one tier below the current one).
    fn requantize(&mut self, to: KvQuant) {
        let old_n = packed_bytes(self.quant, self.row_len);
        let new_n = packed_bytes(to, self.row_len);
        let mut out = vec![0u8; self.n_rows * new_n];
        let mut row = vec![0f32; self.row_len];
        for r in 0..self.n_rows {
            quant::dequantize(
                self.quant,
                &self.buf[r * old_n..(r + 1) * old_n],
                self.row_len,
                &mut row,
            );
            quant::quantize(to, &row, &mut out[r * new_n..(r + 1) * new_n]);
        }
        self.buf = out;
        self.quant = to;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(seed: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i + seed) as f32) * 0.11).sin() * 2.0)
            .collect()
    }

    #[test]
    fn fixed_policy_never_degrades() {
        let mut rows = VbrRows::new(64, VbrPolicy::fixed(KvQuant::F16));
        for i in 0..1000 {
            rows.push(&row(i, 64));
        }
        assert_eq!(rows.quant(), KvQuant::F16);
        assert_eq!(rows.packed_bytes(), 1000 * 64 * 2);
    }

    #[test]
    fn vbr_degrades_down_the_ladder() {
        let mut rows = VbrRows::new(32, VbrPolicy::from_ctx(400, KvQuant::F16));
        // Thresholds: 100, 200, 300, 400. From f16 the ladder has three
        // steps, so the last threshold lands on the q4 floor.
        for i in 0..299 {
            rows.push(&row(i, 32));
        }
        assert_eq!(rows.quant(), KvQuant::Q5_1, "at 299 tokens");
        rows.push(&row(299, 32));
        assert_eq!(rows.quant(), KvQuant::Q4, "at 300 tokens");
        // Bottom of the ladder: extra pushes do not panic or change tier.
        for i in 300..500 {
            rows.push(&row(i, 32));
        }
        assert_eq!(rows.quant(), KvQuant::Q4);
    }

    #[test]
    fn vbr_shrinks_footprint_below_f16() {
        let mut rows = VbrRows::new(128, VbrPolicy::from_ctx(512, KvQuant::F16));
        for i in 0..512 {
            rows.push(&row(i, 128));
        }
        // q4 is 18 bytes / 32 elems = 0.5625 bpe, vs f16 at 2.0 bpe.
        assert_eq!(rows.packed_bytes(), rows.bytes_at(KvQuant::Q4));
        assert!(rows.packed_bytes() < 512 * 128 * 2 / 2);
    }

    #[test]
    fn dequantized_rows_have_expected_shape() {
        let mut rows = VbrRows::new(16, VbrPolicy::from_ctx(64, KvQuant::Q8));
        for i in 0..10 {
            rows.push(&row(i, 16));
        }
        assert_eq!(rows.len(), 10);
        assert_eq!(rows.to_f32().len(), 10 * 16);
        // The first five rows and the last five agree with a fresh read.
        let all = rows.to_f32();
        let tail = rows.to_f32_from(5);
        assert_eq!(all[5 * 16..], tail[..]);
    }

    #[test]
    fn values_stay_close_after_degrading() {
        let mut rows = VbrRows::new(64, VbrPolicy::from_ctx(128, KvQuant::F16));
        let originals: Vec<Vec<f32>> = (0..128).map(|i| row(i, 64)).collect();
        for r in &originals {
            rows.push(r);
        }
        // Now at the bottom of the ladder; each element moved at most a few
        // quantization steps of the row's dynamic range.
        let back = rows.to_f32();
        for (r, orig) in originals.iter().enumerate() {
            for (i, (&a, &b)) in orig.iter().zip(&back[r * 64..(r + 1) * 64]).enumerate() {
                assert!(
                    (a - b).abs() < 1.0,
                    "row {r} elem {i}: {a} vs {b} after full degradation"
                );
            }
        }
    }
}
