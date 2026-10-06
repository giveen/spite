//! P100 SXM2 NVLink tensor-parallelism topology detection.
//!
//! The Tesla P100 SXM2 (DGX-1) supports up to 4-GPU full-mesh NVLink 1.0
//! (160 GB/s bidirectional per GPU).  This module detects the NVLink
//! connectivity and returns the optimal shard count for tensor parallelism.
//!
//! The kernel-side all-reduce is in
//! `kernels/qwen/qwen3_8/nvidia/sm_60/tesla_p100/multi_gpu.cuh`.
//! The Rust host calls `p100_allreduce_f32` (via FFI) after each attention
//! and FFN op.

use crate::ShardStrategy;

/// Maximum tensor-parallel degree for the P100 NVLink ring.
pub const P100_TP_MAX_SHARDS: usize = 4;

/// NVLink topology for a set of P100 SXM2 GPUs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P100Topology {
    /// Number of GPUs that can participate in NVLink all-reduce.
    pub n_linked: usize,
}

impl P100Topology {
    /// Detect NVLink peer connectivity from the environment.
    ///
    /// Reads `SPITE_P100_SHARDS` (1, 2, or 4) when set.
    /// Otherwise falls back to `SPITE_GPU_COUNT` clamped to
    /// `P100_TP_MAX_SHARDS`.
    pub fn detect() -> Self {
        let explicit: Option<usize> = std::env::var("SPITE_P100_SHARDS")
            .ok()
            .and_then(|s| s.parse().ok());
        if let Some(n) = explicit {
            return Self {
                n_linked: n.clamp(1, P100_TP_MAX_SHARDS),
            };
        }
        let gpu_count: usize = std::env::var("SPITE_GPU_COUNT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        Self {
            n_linked: gpu_count.clamp(1, P100_TP_MAX_SHARDS),
        }
    }

    /// Returns true if tensor parallelism is available (≥2 linked GPUs).
    pub fn is_multi_gpu(&self) -> bool {
        self.n_linked >= 2
    }
}

/// Per-GPU configuration for a P100 tensor-parallel run.
#[derive(Debug, Clone)]
pub struct P100TpConfig {
    pub n_shards: usize,
    pub shard_idx: usize,
}

impl P100TpConfig {
    /// Build configs for all shards in a topology.
    pub fn for_topology(topo: &P100Topology) -> Vec<Self> {
        (0..topo.n_linked)
            .map(|i| Self {
                n_shards: topo.n_linked,
                shard_idx: i,
            })
            .collect()
    }
}

/// Returns the optimal `ShardStrategy` for P100 NVLink topology.
///
/// Picks the largest power-of-two shard count ≤ detected GPU count
/// and ≤ `P100_TP_MAX_SHARDS`.  Returns `ShardStrategy::None` for a
/// single GPU.
pub fn p100_optimal_shard_count() -> ShardStrategy {
    let topo = P100Topology::detect();
    if !topo.is_multi_gpu() {
        return ShardStrategy::None;
    }
    // Round down to nearest power of two (1, 2, or 4).
    let n = topo.n_linked;
    let n_pow2 = if n >= 4 {
        4
    } else if n >= 2 {
        2
    } else {
        1
    };
    if n_pow2 <= 1 {
        ShardStrategy::None
    } else {
        ShardStrategy::Tensor { n_shards: n_pow2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_gpu_no_tp() {
        // With no env vars, defaults to 1 GPU → no tensor parallelism.
        unsafe {
            std::env::remove_var("SPITE_P100_SHARDS");
            std::env::remove_var("SPITE_GPU_COUNT");
        }
        let topo = P100Topology::detect();
        assert_eq!(topo.n_linked, 1);
        assert!(!topo.is_multi_gpu());
        assert_eq!(p100_optimal_shard_count(), ShardStrategy::None);
    }

    #[test]
    fn four_gpu_tp() {
        unsafe { std::env::set_var("SPITE_P100_SHARDS", "4") };
        let topo = P100Topology::detect();
        assert_eq!(topo.n_linked, 4);
        assert!(topo.is_multi_gpu());
        assert_eq!(
            p100_optimal_shard_count(),
            ShardStrategy::Tensor { n_shards: 4 }
        );
        unsafe { std::env::remove_var("SPITE_P100_SHARDS") };
    }

    #[test]
    fn clamp_above_max() {
        unsafe { std::env::set_var("SPITE_P100_SHARDS", "8") };
        let topo = P100Topology::detect();
        assert_eq!(topo.n_linked, P100_TP_MAX_SHARDS);
        unsafe { std::env::remove_var("SPITE_P100_SHARDS") };
    }

    #[test]
    fn configs_match_topology() {
        let topo = P100Topology { n_linked: 2 };
        let cfgs = P100TpConfig::for_topology(&topo);
        assert_eq!(cfgs.len(), 2);
        assert_eq!(cfgs[0].shard_idx, 0);
        assert_eq!(cfgs[1].shard_idx, 1);
        assert!(cfgs.iter().all(|c| c.n_shards == 2));
    }
}
