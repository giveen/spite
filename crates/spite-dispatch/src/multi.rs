//! Multi-GPU dispatch: one `DispatchTable` per GPU node, layer ranges auto-assigned.
//!
//! spite runs **pipeline parallelism** across heterogeneous GPUs:
//! each node handles a contiguous range of transformer layers and passes
//! activations to the next node.  Every node gets its own `DispatchTable`
//! built against its specific GPU arch, so a mixed fleet like
//! `RTX 5070 (sm_120) + RTX 3090 (sm_86)` uses Blackwell kernels on the
//! first GPU and Ampere kernels on the second — no compromises.
//!
//! # Communication topology
//!
//! ```text
//! Same-vendor NVLink / XGMI:
//!   GPU 0 ──NVLink──▶ GPU 1           (900 GB/s — fast)
//!
//! Same-vendor PCIe (no NVLink):
//!   GPU 0 ──PCIe──▶ GPU 1             (~64 GB/s — OK)
//!
//! Cross-vendor (CUDA + ROCm):
//!   GPU 0 ──PCIe──▶ CPU RAM ──PCIe──▶ GPU 1   (~32 GB/s per leg)
//! ```
//!
//! spite surfaces the detected topology so users can make informed decisions.
//! Cross-vendor pipeline parallelism works — it is just bandwidth-limited to
//! PCIe speeds.  For large activations (long sequences or large batch sizes)
//! consider assigning more layers to the faster GPU to reduce transfer volume.
//!
//! # Layer assignment
//!
//! By default layers are split proportional to each GPU's VRAM.  A GPU with
//! unknown VRAM (Apple Unified Memory, any card not in the card database)
//! counts as 1 part in the proportional split.  Users can override with an
//! explicit `layer_counts` slice.

use std::ops::Range;
use std::path::Path;

use crate::{DispatchBuilder, DispatchError, DispatchTable};
use crate::resolve::KernelSpec;
use super::{card_spec, normalize_card_name};

// ── GPU node ───────────────────────────────────────────────────────────────

/// One physical GPU in the pipeline.
#[derive(Debug, Clone)]
pub struct GpuNode {
    /// Normalised card id: "rtx_5090", "mi300x", …
    pub card_id:  String,
    /// GPU arch string used in the kernel directory tree.
    pub gpu_arch: String,
    /// Dedicated VRAM in GiB (0 = unknown / Apple Unified Memory).
    pub vram_gib: u32,
    /// Transformer layers this GPU is responsible for (assigned later).
    pub layers:   Range<usize>,
    /// How this GPU hands activations to the *next* node in the pipeline.
    pub link_out: CommLink,
}

impl GpuNode {
    /// Build a `GpuNode` from a raw card name string.
    pub fn from_card(raw: &str) -> Self {
        let card_id = normalize_card_name(raw);
        let spec    = card_spec(&card_id);
        Self {
            card_id,
            gpu_arch: spec.gpu_arch.to_owned(),
            vram_gib: spec.vram_gib,
            layers:   0..0,   // filled in by MultiGpuSpec::assign_layers
            link_out: CommLink::None,
        }
    }
}

// ── Communication topology ─────────────────────────────────────────────────

/// How activations travel from one pipeline stage to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommLink {
    /// Last node in the pipeline — no outbound transfer.
    None,
    /// Same vendor, NVLink or XGMI / Infinity Fabric (fast device-to-device).
    DeviceLink,
    /// Same vendor, PCIe only (no NVLink bridge detected).
    PcieDirectP2P,
    /// Cross-vendor: activations must transit host CPU memory.
    /// NVIDIA→Host→AMD or AMD→Host→NVIDIA (slow but correct).
    HostCopy,
}

impl CommLink {
    pub fn label(self) -> &'static str {
        match self {
            Self::None          => "—",
            Self::DeviceLink    => "NVLink / XGMI",
            Self::PcieDirectP2P => "PCIe P2P",
            Self::HostCopy      => "PCIe via host RAM (cross-vendor)",
        }
    }

    /// Estimated peak bandwidth in GB/s (conservative values for planning).
    pub fn bandwidth_gbs(self) -> f32 {
        match self {
            Self::None          =>   0.0,
            Self::DeviceLink    => 900.0,   // NVLink 4.0 / XGMI gen4
            Self::PcieDirectP2P =>  64.0,   // PCIe 5.0 x16 bidir
            Self::HostCopy      =>  28.0,   // PCIe 5.0 one-direction (2 legs)
        }
    }
}

// ── Multi-GPU spec ─────────────────────────────────────────────────────────

/// A fully-described multi-GPU pipeline configuration.
pub struct MultiGpuSpec {
    pub nodes: Vec<GpuNode>,
}

impl MultiGpuSpec {
    /// Build from a slice of raw card name strings (e.g. `["RTX 5070", "RTX 3090"]`).
    ///
    /// Detects the communication topology automatically from vendor strings.
    pub fn from_cards(raw_cards: &[&str]) -> Self {
        let mut nodes: Vec<GpuNode> = raw_cards.iter().map(|s| GpuNode::from_card(s)).collect();
        assign_comm_links(&mut nodes);
        Self { nodes }
    }

    /// Assign transformer layer ranges to each node proportional to VRAM.
    ///
    /// Nodes with `vram_gib == 0` (unknown / Apple) each receive one share.
    /// Call this once the model's total layer count is known.
    ///
    /// To override the auto-assignment, pass `Some(counts)` where `counts[i]`
    /// is the number of layers you want node `i` to handle.  The counts are
    /// renormalised so their sum equals `n_layers`.
    pub fn assign_layers(&mut self, n_layers: usize, override_counts: Option<&[u32]>) {
        let n = self.nodes.len();
        if n == 0 { return; }

        // Build weight vector (VRAM or user-supplied).
        let weights: Vec<f64> = match override_counts {
            Some(w) if w.len() == n => w.iter().map(|&v| v as f64).collect(),
            _ => self.nodes.iter().map(|g| {
                if g.vram_gib > 0 { g.vram_gib as f64 } else { 1.0 }
            }).collect(),
        };

        let total_w: f64 = weights.iter().sum();
        let mut start = 0usize;
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let end = if i == n - 1 {
                n_layers
            } else {
                let frac = weights[i] / total_w;
                (start + (frac * n_layers as f64).round() as usize).min(n_layers)
            };
            node.layers = start..end;
            start = end;
        }
    }

    /// Build one `DispatchTable` per GPU node.
    pub fn build_tables(
        &self,
        model_arch:  &str,
        kernels_dir: &Path,
    ) -> Vec<Result<DispatchTable, DispatchError>> {
        self.nodes.iter().map(|node| {
            let mut spec = KernelSpec::from_arch(model_arch, &node.gpu_arch);
            spec.card_id = node.card_id.clone();
            DispatchBuilder::new(kernels_dir, spec).build()
        }).collect()
    }

    /// Human-readable summary of this configuration.
    pub fn print_summary(&self) {
        println!("multi-GPU pipeline ({} nodes):", self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let vram_s = if node.vram_gib > 0 {
                format!("{} GiB", node.vram_gib)
            } else {
                "shared".into()
            };
            let layers_s = if node.layers.is_empty() {
                "layers TBD".into()
            } else {
                format!("layers {}–{}", node.layers.start, node.layers.end - 1)
            };
            println!(
                "  [{i}] {:<20}  {:<10}  {vram_s:<10}  {layers_s}",
                node.card_id, node.gpu_arch
            );
            if node.link_out != CommLink::None {
                println!(
                    "      → node {}: {}  ({:.0} GB/s)",
                    i + 1,
                    node.link_out.label(),
                    node.link_out.bandwidth_gbs()
                );
            }
        }
        if self.has_cross_vendor() {
            println!();
            println!("  note: cross-vendor pair detected — activations transit host");
            println!("        RAM between those nodes (~{:.0} GB/s limit).",
                CommLink::HostCopy.bandwidth_gbs());
        }
    }

    /// `true` if any two adjacent nodes use different GPU backends.
    pub fn has_cross_vendor(&self) -> bool {
        self.nodes.windows(2).any(|w| vendor(&w[0].gpu_arch) != vendor(&w[1].gpu_arch))
    }

    /// Collect all unique GPU arch strings across nodes.
    pub fn gpu_arches(&self) -> Vec<&str> {
        let mut seen = Vec::new();
        for n in &self.nodes {
            if !seen.contains(&n.gpu_arch.as_str()) {
                seen.push(n.gpu_arch.as_str());
            }
        }
        seen
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Vendor string derived from GPU arch for topology detection.
fn vendor(gpu_arch: &str) -> &'static str {
    if gpu_arch.starts_with("sm_")                           { return "nvidia"; }
    if gpu_arch.starts_with("rdna") || gpu_arch.starts_with("cdna") { return "amd"; }
    if gpu_arch == "metal"                                   { return "apple"; }
    if gpu_arch.starts_with("xe")                            { return "intel"; }
    "generic"
}

/// Assign `CommLink` values to every node based on vendor adjacency.
fn assign_comm_links(nodes: &mut Vec<GpuNode>) {
    let n = nodes.len();
    for i in 0..n {
        if i == n - 1 {
            nodes[i].link_out = CommLink::None;
            continue;
        }
        let v_a = vendor(&nodes[i].gpu_arch.clone());
        let v_b = vendor(&nodes[i + 1].gpu_arch.clone());
        nodes[i].link_out = if v_a != v_b {
            CommLink::HostCopy
        } else if v_a == "nvidia" || v_a == "amd" {
            // Optimistic default: assume direct link available.
            // TODO: detect actual NVLink / XGMI topology via driver API.
            CommLink::DeviceLink
        } else {
            CommLink::PcieDirectP2P
        };
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_vendor_layer_split() {
        let mut spec = MultiGpuSpec::from_cards(&["RTX_5070", "RTX_3090"]);
        spec.assign_layers(32, None);
        // 5070 = 12 GiB, 3090 = 24 GiB → 1:2 ratio
        // 5070 gets ~11 layers, 3090 gets ~21 layers (rounded)
        assert_eq!(spec.nodes[0].layers.start, 0);
        assert_eq!(spec.nodes[1].layers.end, 32);
        assert!(!spec.nodes[0].layers.is_empty());
        assert!(!spec.nodes[1].layers.is_empty());
        // Same vendor → device link
        assert_eq!(spec.nodes[0].link_out, CommLink::DeviceLink);
        assert_eq!(spec.nodes[1].link_out, CommLink::None);
    }

    #[test]
    fn cross_vendor_uses_host_copy() {
        let spec = MultiGpuSpec::from_cards(&["RTX_4090", "RX_7900_XTX"]);
        assert_eq!(spec.nodes[0].link_out, CommLink::HostCopy);
        assert!(spec.has_cross_vendor());
    }

    #[test]
    fn explicit_layer_override() {
        let mut spec = MultiGpuSpec::from_cards(&["RTX_5090", "RTX_5090"]);
        spec.assign_layers(40, Some(&[30, 10]));
        assert_eq!(spec.nodes[0].layers, 0..30);
        assert_eq!(spec.nodes[1].layers, 30..40);
    }

    #[test]
    fn single_gpu_is_valid() {
        let mut spec = MultiGpuSpec::from_cards(&["MI300X"]);
        spec.assign_layers(64, None);
        assert_eq!(spec.nodes[0].layers, 0..64);
        assert_eq!(spec.nodes[0].link_out, CommLink::None);
    }

    #[test]
    fn arches_deduplicated() {
        let spec = MultiGpuSpec::from_cards(&["RTX_5070", "RTX_3090"]);
        let arches = spec.gpu_arches();
        assert_eq!(arches.len(), 2);
        assert!(arches.contains(&"sm_120"));
        assert!(arches.contains(&"sm_86"));
    }
}
