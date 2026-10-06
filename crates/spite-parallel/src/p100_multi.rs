//! Tesla P100 tensor-parallel shard policy.
//!
//! P100 SXM2 groups (DGX-1 quads) are fully connected over NVLink 1.0, so
//! Megatron-style tensor parallelism across up to 4 cards is viable; PCIe
//! P100s have no NVLink and should not be tensor-sharded.
//!
//! The GPU count comes from the environment (`SPITE_P100_SHARDS`, else
//! `SPITE_GPU_COUNT`); there is no NVLink topology probe yet, so a PCIe box
//! must leave both unset or set `SPITE_P100_SHARDS=1`.
//!
//! The kernel-side all-reduce is exported by the P100 card kernel as
//! `spite_p100_enable_peer_access` / `spite_p100_allreduce_f32`
//! (`kernels/qwen/qwen3_5/nvidia/sm_60/tesla_p100/kernel.cu`). The executor
//! does not call it yet: per-device weight sharding and the per-op
//! all-reduce still have to be wired into the host.

use crate::ShardStrategy;

/// Maximum tensor-parallel degree for a P100 NVLink group.
pub const P100_TP_MAX_SHARDS: usize = 4;

/// GPUs available to a P100 tensor-parallel group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P100Topology {
    pub n_linked: usize,
}

impl P100Topology {
    /// Read the group size from `SPITE_P100_SHARDS`, else `SPITE_GPU_COUNT`.
    pub fn detect() -> Self {
        Self::from_env_values(
            std::env::var("SPITE_P100_SHARDS").ok().as_deref(),
            std::env::var("SPITE_GPU_COUNT").ok().as_deref(),
        )
    }

    /// `detect()` without touching the process environment. An explicit
    /// shard count wins over the GPU count; unparsable values are ignored.
    pub fn from_env_values(p100_shards: Option<&str>, gpu_count: Option<&str>) -> Self {
        let parse = |v: Option<&str>| v.and_then(|s| s.trim().parse::<usize>().ok());
        let n = parse(p100_shards).or(parse(gpu_count)).unwrap_or(1);
        Self {
            n_linked: n.clamp(1, P100_TP_MAX_SHARDS),
        }
    }

    /// True when tensor parallelism is possible (two or more GPUs).
    pub fn is_multi_gpu(&self) -> bool {
        self.n_linked >= 2
    }

    /// Largest power-of-two shard count that fits the group, so KV heads and
    /// FFN columns split evenly across shards.
    pub fn shard_strategy(&self) -> ShardStrategy {
        match self.n_linked {
            n if n >= 4 => ShardStrategy::Tensor { n_shards: 4 },
            2 | 3 => ShardStrategy::Tensor { n_shards: 2 },
            _ => ShardStrategy::None,
        }
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

/// `ShardStrategy` for the P100 group described by the environment.
pub fn p100_optimal_shard_count() -> ShardStrategy {
    P100Topology::detect().shard_strategy()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_gpu_no_tp() {
        let topo = P100Topology::from_env_values(None, None);
        assert_eq!(topo.n_linked, 1);
        assert!(!topo.is_multi_gpu());
        assert_eq!(topo.shard_strategy(), ShardStrategy::None);
    }

    #[test]
    fn explicit_shards_win_over_gpu_count() {
        let topo = P100Topology::from_env_values(Some("4"), Some("2"));
        assert_eq!(topo.n_linked, 4);
        assert_eq!(topo.shard_strategy(), ShardStrategy::Tensor { n_shards: 4 });
    }

    #[test]
    fn gpu_count_fallback_and_clamp() {
        assert_eq!(P100Topology::from_env_values(None, Some("8")).n_linked, 4);
        assert_eq!(P100Topology::from_env_values(Some("0"), None).n_linked, 1);
        assert_eq!(
            P100Topology::from_env_values(Some("x"), Some("2")).n_linked,
            2
        );
    }

    #[test]
    fn three_gpus_round_down_to_two() {
        let topo = P100Topology::from_env_values(Some("3"), None);
        assert_eq!(topo.shard_strategy(), ShardStrategy::Tensor { n_shards: 2 });
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
