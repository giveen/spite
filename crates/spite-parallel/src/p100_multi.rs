//! Tesla P100 tensor-parallel shard policy.
//!
//! Tensor parallelism shards each layer's weight matrices across the cards and
//! needs one all-reduce per attention/FFN op, so it is only worth it when the
//! cards can exchange data directly. The gate is **peer access**
//! (`cudaDeviceCanAccessPeer`), not the link medium:
//!
//! * P100 **SXM2** groups (DGX-1 quads) are fully connected over NVLink 1.0.
//! * P100 **PCIe** cards have no NVLink, but a pair under one root complex
//!   (**PHB**) still has PCIe peer access, and pxa measures a real win there:
//!   `-sm tensor` on 2× P100-PCIE is +17–27% decode and +44% prefill over the
//!   layer split (pxa `docs/DEFAULTS.md`). A peer-capable PCIe pair is a tensor
//!   group, not a pipeline.
//! * A pair with **no** peer access (SYS across QPI, or a platform that blocks
//!   PCIe P2P) must use the layer-wise pipeline, which moves only the hidden
//!   state through host memory.
//!
//! [`P100Topology::detect`] probes the `0 -> 1` link with `spite-gpu`'s
//! `cuda::p2p_kind` (peer access + the SXM form factor in the device name).
//! `SPITE_P100_LINK=none|pcie|nvlink` overrides the probe; the legacy
//! `SPITE_P100_NVLINK=1|0` is accepted as `nvlink|pcie`. The GPU count comes
//! from `SPITE_P100_SHARDS`, else `SPITE_GPU_COUNT`. Enumerating the
//! environment without CUDA (the `from_*` constructors used in tests) assumes
//! **no** link.
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

/// GPUs available to a P100 tensor-parallel group, and the link between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P100Topology {
    pub n_linked: usize,
    /// Direct device link: `None` (no peer access), `Pcie` (peer access over
    /// PCIe — PHB/SYS), or `Nvlink` (SXM2). Only `None` rules out tensor
    /// parallelism.
    pub link: P2pKind,
}

impl P100Topology {
    /// Read the group size from `SPITE_P100_SHARDS`, else `SPITE_GPU_COUNT`,
    /// and probe the `0 -> 1` link (env override `SPITE_P100_LINK` /
    /// `SPITE_P100_NVLINK`, else `cuda::p2p_kind`).
    pub fn detect() -> Self {
        let mut topo = Self::from_parts(
            std::env::var("SPITE_P100_SHARDS").ok().as_deref(),
            std::env::var("SPITE_GPU_COUNT").ok().as_deref(),
            P2pKind::None,
        );
        if topo.n_linked >= 2 {
            topo.link = probe_link();
        }
        topo
    }

    /// `detect()` without probing. An explicit shard count wins over the GPU
    /// count; unparsable values are ignored; the count clamps to
    /// [`P100_TP_MAX_SHARDS`]. `link` is taken as given.
    pub fn from_parts(p100_shards: Option<&str>, gpu_count: Option<&str>, link: P2pKind) -> Self {
        let parse = |v: Option<&str>| v.and_then(|s| s.trim().parse::<usize>().ok());
        let n = parse(p100_shards).or(parse(gpu_count)).unwrap_or(1);
        let n_linked = n.clamp(1, P100_TP_MAX_SHARDS);
        Self {
            n_linked,
            link: if n_linked >= 2 { link } else { P2pKind::None },
        }
    }

    /// `from_parts` with no link (the conservative default when the link is
    /// unknown or unprobed).
    pub fn from_env_values(p100_shards: Option<&str>, gpu_count: Option<&str>) -> Self {
        Self::from_parts(p100_shards, gpu_count, P2pKind::None)
    }

    /// True when there is more than one GPU to spread the model over.
    pub fn is_multi_gpu(&self) -> bool {
        self.n_linked >= 2
    }

    /// True when the cards can exchange data directly, so tensor parallelism
    /// is possible (NVLink **or** PCIe peer access).
    pub fn can_tensor_parallel(&self) -> bool {
        self.link != P2pKind::None
    }

    /// Best split for the group, so KV heads and FFN columns divide evenly.
    ///
    /// * one GPU, or no peer access → [`ShardStrategy::None`] / pipeline only;
    /// * any direct link → the largest power-of-two tensor degree that fits.
    pub fn shard_strategy(&self) -> ShardStrategy {
        if self.n_linked < 2 {
            return ShardStrategy::None;
        }
        if self.link == P2pKind::None {
            return ShardStrategy::Pipeline {
                n_stages: self.n_linked,
            };
        }
        if self.n_linked >= 4 {
            ShardStrategy::Tensor { n_shards: 4 }
        } else {
            ShardStrategy::Tensor { n_shards: 2 }
        }
    }
}

/// Runtime `0 -> 1` link probe with a `SPITE_P100_LINK` override
/// (`none|pcie|nvlink`); `SPITE_P100_NVLINK=1|0` is the legacy alias.
fn probe_link() -> P2pKind {
    if let Ok(v) = std::env::var("SPITE_P100_LINK") {
        match v.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "0" => return P2pKind::None,
            "pcie" | "pci" => return P2pKind::Pcie,
            "nvlink" | "sxm" => return P2pKind::Nvlink,
            _ => {}
        }
    }
    if let Ok(v) = std::env::var("SPITE_P100_NVLINK") {
        let t = v.trim().to_ascii_lowercase();
        if !t.is_empty() {
            return if matches!(t.as_str(), "1" | "true" | "yes" | "on") {
                P2pKind::Nvlink
            } else {
                P2pKind::Pcie
            };
        }
    }
    cuda::p2p_kind(0, 1).unwrap_or(P2pKind::None)
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
        assert_eq!(topo.link, P2pKind::None);
        assert_eq!(topo.shard_strategy(), ShardStrategy::None);
    }

    #[test]
    fn explicit_shards_win_over_gpu_count() {
        let topo = P100Topology::from_env_values(Some("4"), Some("2"));
        assert_eq!(topo.n_linked, 4);
        // No link assumed by `from_env_values`: pipeline, not tensor.
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
        let topo = P100Topology::from_parts(Some("4"), None, P2pKind::Nvlink);
        assert!(topo.can_tensor_parallel());
        assert_eq!(topo.shard_strategy(), ShardStrategy::Tensor { n_shards: 4 });
        // Three cards still round down to a power of two.
        let three = P100Topology::from_parts(Some("3"), None, P2pKind::Nvlink);
        assert_eq!(
            three.shard_strategy(),
            ShardStrategy::Tensor { n_shards: 2 }
        );
    }

    #[test]
    fn pcie_peer_pair_is_a_tensor_group() {
        // pxa measures `-sm tensor` on 2x P100-PCIE (PHB) at +17-27% decode
        // over the layer split, so peer-capable PCIe is tensor, not pipeline.
        let topo = P100Topology::from_parts(Some("2"), None, P2pKind::Pcie);
        assert!(topo.is_multi_gpu());
        assert!(topo.can_tensor_parallel());
        assert_eq!(topo.shard_strategy(), ShardStrategy::Tensor { n_shards: 2 });
    }

    #[test]
    fn no_peer_access_falls_back_to_pipeline() {
        // SYS across QPI (or a platform without PCIe P2P): pipeline only.
        let topo = P100Topology::from_parts(Some("2"), None, P2pKind::None);
        assert!(topo.is_multi_gpu());
        assert!(!topo.can_tensor_parallel());
        assert_eq!(
            topo.shard_strategy(),
            ShardStrategy::Pipeline { n_stages: 2 }
        );
    }

    #[test]
    fn one_gpu_has_no_link() {
        let topo = P100Topology::from_parts(Some("1"), None, P2pKind::Nvlink);
        assert_eq!(topo.link, P2pKind::None);
        assert_eq!(topo.shard_strategy(), ShardStrategy::None);
    }

    #[test]
    fn configs_match_topology() {
        let topo = P100Topology {
            n_linked: 2,
            link: P2pKind::Pcie,
        };
        let cfgs = P100TpConfig::for_topology(&topo);
        assert_eq!(cfgs.len(), 2);
        assert_eq!(cfgs[0].shard_idx, 0);
        assert_eq!(cfgs[1].shard_idx, 1);
        assert!(cfgs.iter().all(|c| c.n_shards == 2));
    }
}
