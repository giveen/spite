//! Tesla P100 tensor-parallel shard policy.
//!
//! P100 **SXM2** groups (DGX-1 quads) are fully connected over NVLink 1.0, so
//! Megatron-style tensor parallelism across up to 4 cards is viable.
//!
//! P100 **PCIe** cards have no NVLink. Peer copies fall back to PCIe (PHB when
//! both cards sit under one root complex, SYS across QPI on dual-socket boards),
//! which is far too bandwidth-bound to tensor-shard a dense model, and this
//! policy refuses it: with no NVLink, [`P100Topology::shard_strategy`] returns
//! [`ShardStrategy::Pipeline`], never [`ShardStrategy::Tensor`]. The dense
//! layer-wise pipeline in `spite-models`/`spite-parallel::pipeline` moves only
//! the hidden state through host memory, so it is the right split for PCIe/QPI
//! boxes (2× and 4× P100-PCIE-16GB included).
//!
//! The GPU count comes from the environment (`SPITE_P100_SHARDS`, else
//! `SPITE_GPU_COUNT`). NVLink is probed at runtime by
//! [`P100Topology::detect`] via `spite-gpu`'s `cuda::p2p_kind` (peer access plus
//! the SXM form factor in the device name); `SPITE_P100_NVLINK=1` (or `=0`)
//! overrides the probe. Enumerating the environment without CUDA (all of the
//! `from_*` constructors used in tests) assumes **no** NVLink.
//!
//! The kernel-side all-reduce is exported by the P100 card kernel as
//! `spite_p100_enable_peer_access` / `spite_p100_allreduce_f32`
//! (`kernels/qwen/qwen3_5/nvidia/sm_60/tesla_p100/kernel.cu`). The executor does
//! not call it yet: per-device weight sharding and the per-op all-reduce still
//! have to be wired into the host.

use spite_gpu::{P2pKind, cuda};

use crate::ShardStrategy;

/// Maximum tensor-parallel degree for a P100 NVLink group.
pub const P100_TP_MAX_SHARDS: usize = 4;

/// GPUs available to a P100 tensor-parallel group, and whether they are linked
/// by NVLink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P100Topology {
    pub n_linked: usize,
    /// True only when the group has peer access over an SXM/NVLink link. A
    /// PCIe group is `false` even when the cards can address each other.
    pub nvlink: bool,
}

impl P100Topology {
    /// Read the group size from `SPITE_P100_SHARDS`, else `SPITE_GPU_COUNT`,
    /// and probe the `0 -> 1` link for NVLink (env override
    /// `SPITE_P100_NVLINK`, else `cuda::p2p_kind`).
    pub fn detect() -> Self {
        let mut topo = Self::from_parts(
            std::env::var("SPITE_P100_SHARDS").ok().as_deref(),
            std::env::var("SPITE_GPU_COUNT").ok().as_deref(),
            false,
        );
        if topo.n_linked >= 2 {
            topo.nvlink = probe_nvlink();
        }
        topo
    }

    /// `detect()` without touching the environment beyond the explicit
    /// override, and without probing. An explicit shard count wins over the
    /// GPU count; unparsable values are ignored; the count clamps to
    /// [`P100_TP_MAX_SHARDS`]. `nvlink` is taken as given.
    pub fn from_parts(p100_shards: Option<&str>, gpu_count: Option<&str>, nvlink: bool) -> Self {
        let parse = |v: Option<&str>| v.and_then(|s| s.trim().parse::<usize>().ok());
        let n = parse(p100_shards).or(parse(gpu_count)).unwrap_or(1);
        let n_linked = n.clamp(1, P100_TP_MAX_SHARDS);
        Self {
            n_linked,
            nvlink: nvlink && n_linked >= 2,
        }
    }

    /// `from_parts` with no NVLink (the conservative default when the link is
    /// unknown or unprobed).
    pub fn from_env_values(p100_shards: Option<&str>, gpu_count: Option<&str>) -> Self {
        Self::from_parts(p100_shards, gpu_count, false)
    }

    /// True when there is more than one GPU to spread the model over.
    pub fn is_multi_gpu(&self) -> bool {
        self.n_linked >= 2
    }

    /// True when tensor parallelism is both possible and worthwhile (NVLink).
    pub fn can_tensor_parallel(&self) -> bool {
        self.nvlink
    }

    /// Best split for the group, so KV heads and FFN columns divide evenly.
    ///
    /// * one GPU, or no NVLink → [`ShardStrategy::None`] / pipeline only;
    /// * NVLink → the largest power-of-two tensor degree that fits.
    pub fn shard_strategy(&self) -> ShardStrategy {
        match self.n_linked {
            n if n < 2 => ShardStrategy::None,
            n if !self.nvlink => ShardStrategy::Pipeline { n_stages: n },
            n if n >= 4 => ShardStrategy::Tensor { n_shards: 4 },
            _ => ShardStrategy::Tensor { n_shards: 2 },
        }
    }
}

/// Runtime NVLink probe for devices 0 and 1, with a `SPITE_P100_NVLINK`
/// override (`1/true/yes/on` → NVLink, anything else non-empty → PCIe).
fn probe_nvlink() -> bool {
    if let Ok(v) = std::env::var("SPITE_P100_NVLINK") {
        let t = v.trim().to_ascii_lowercase();
        if !t.is_empty() {
            return matches!(t.as_str(), "1" | "true" | "yes" | "on");
        }
    }
    matches!(cuda::p2p_kind(0, 1), Ok(P2pKind::Nvlink))
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
        assert!(!topo.can_tensor_parallel());
        assert_eq!(topo.shard_strategy(), ShardStrategy::None);
    }

    #[test]
    fn explicit_shards_win_over_gpu_count() {
        let topo = P100Topology::from_env_values(Some("4"), Some("2"));
        assert_eq!(topo.n_linked, 4);
        // No NVLink assumed by `from_env_values`: never tensor-shard PCIe.
        assert_eq!(
            topo.shard_strategy(),
            ShardStrategy::Pipeline { n_stages: 4 }
        );
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
    fn nvlink_enables_tensor_parallel() {
        let topo = P100Topology::from_parts(Some("4"), None, true);
        assert!(topo.can_tensor_parallel());
        assert_eq!(topo.shard_strategy(), ShardStrategy::Tensor { n_shards: 4 });
        // Three NVLink cards still round down to a power of two.
        let three = P100Topology::from_parts(Some("3"), None, true);
        assert_eq!(
            three.shard_strategy(),
            ShardStrategy::Tensor { n_shards: 2 }
        );
    }

    #[test]
    fn pcie_falls_back_to_pipeline() {
        // Two PCIe P100s (PHB/SYS) must be a pipeline, not a tensor group.
        let topo = P100Topology::from_parts(Some("2"), None, false);
        assert!(topo.is_multi_gpu());
        assert!(!topo.can_tensor_parallel());
        assert_eq!(
            topo.shard_strategy(),
            ShardStrategy::Pipeline { n_stages: 2 }
        );
    }

    #[test]
    fn nvlink_with_one_gpu_is_impossible() {
        let topo = P100Topology::from_parts(Some("1"), None, true);
        assert!(!topo.nvlink);
        assert_eq!(topo.shard_strategy(), ShardStrategy::None);
    }

    #[test]
    fn configs_match_topology() {
        let topo = P100Topology {
            n_linked: 2,
            nvlink: true,
        };
        let cfgs = P100TpConfig::for_topology(&topo);
        assert_eq!(cfgs.len(), 2);
        assert_eq!(cfgs[0].shard_idx, 0);
        assert_eq!(cfgs[1].shard_idx, 1);
        assert!(cfgs.iter().all(|c| c.n_shards == 2));
    }
}
