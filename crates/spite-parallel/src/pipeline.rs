//! Pipeline parallelism: split transformer layers across GPUs.

/// Assign layers to pipeline stages.
///
/// Returns a vec of `(start_layer, end_layer)` ranges, one per stage.
/// Layers are distributed as evenly as possible; the last stage absorbs
/// any remainder.
pub fn stage_ranges(n_layers: usize, n_stages: usize) -> Vec<(usize, usize)> {
    if n_stages == 0 || n_stages > n_layers {
        return vec![(0, n_layers)];
    }
    let base = n_layers / n_stages;
    let extra = n_layers % n_stages;
    let mut ranges = Vec::with_capacity(n_stages);
    let mut start = 0;
    for i in 0..n_stages {
        let count = base + if i < extra { 1 } else { 0 };
        ranges.push((start, start + count));
        start += count;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_split() {
        let r = stage_ranges(32, 4);
        assert_eq!(r, vec![(0, 8), (8, 16), (16, 24), (24, 32)]);
    }

    #[test]
    fn uneven_split() {
        let r = stage_ranges(33, 4);
        // 33 = 9+8+8+8 → extra=1, first stage gets +1
        assert_eq!(r[0], (0, 9));
        assert_eq!(r[1], (9, 17));
        assert_eq!(r[2], (17, 25));
        assert_eq!(r[3], (25, 33));
    }
}
