//! Three-tier weight storage: VRAM → system RAM → disk (mmap).
//!
//! spite already uses mmap for GGUF loading, so the "disk tier" is free:
//! the OS page cache handles NVMe/SSD/HDD streaming transparently.
//! This crate provides the tier planner and per-layer placement metadata
//! that the executor uses to decide where each layer lives.
//!
//! # Memory hierarchy
//!
//! ```text
//!  NVMe / SSD / HDD  ←—mmap—→  System RAM  ←—PCIe DMA—→  VRAM
//!       ~7 GB/s                  ~100 GB/s                ~900 GB/s
//! ```
//!
//! # Vision encoder offload
//!
//! Vision encoders (SigLIP, CLIP, InternViT) are a separate weight block
//! that only runs during image prefill — not during text decode.  After each
//! image is processed the encoder can be evicted back to its storage tier,
//! freeing VRAM for the backbone.
//!
//! Pass a [`VisionSpec`] to [`TieredPlacement::plan`] and the planner
//! places the encoder automatically after the backbone:
//!
//! ```rust,ignore
//! let placement = TieredPlacement::plan(
//!     8 * GIB, 2048 * GIB, 500 * MIB, 80,
//!     &OffloadConfig::default(),
//!     Some(&VisionSpec { encoder_bytes: 2 * GIB, policy: VisionOffloadPolicy::Auto }),
//! );
//! placement.print_summary(8, 500 * MIB);
//! ```
//!
//! # Example: RTX 2080 (8 GB) + 2 TB RAM
//!
//! For a 70B model at Q4_K_M (~40 GB):
//!   - ~6 GB usable VRAM (after 2 GB KV-cache reserve) → ~12 layers on GPU
//!   - ~34 GB needed for the rest → fits entirely in RAM
//!   - disk tier: 0 layers (model fits in RAM; disk only when RAM is also tight)

use thiserror::Error;

const GIB: u64 = 1 << 30;

// ── Memory tier ────────────────────────────────────────────────────────────

/// Where a transformer layer's weights live at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Weights resident in GPU VRAM — full-speed compute.
    Vram,
    /// Weights locked into system RAM — DMA'd to GPU per forward pass.
    SystemRam,
    /// Weights mmap'd from disk — OS page cache pages them in on demand.
    Disk,
}

impl MemoryTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Vram => "VRAM",
            Self::SystemRam => "RAM",
            Self::Disk => "disk (mmap)",
        }
    }

    /// Conservative peak bandwidth for planning (GB/s).
    pub fn bandwidth_gbs(self) -> f32 {
        match self {
            Self::Vram => 900.0,
            Self::SystemRam => 100.0,
            Self::Disk => 7.0,
        }
    }
}

// ── Config ─────────────────────────────────────────────────────────────────

/// Offload policy configuration.
#[derive(Debug, Clone)]
pub struct OffloadConfig {
    /// Bytes to reserve in VRAM for KV cache and activation buffers.
    /// Default: 2 GiB.
    pub vram_reserved_bytes: u64,

    /// Maximum system RAM to consume for model weights.
    /// `u64::MAX` (default) means "use all available".
    pub ram_budget_bytes: u64,

    /// Number of layers to prefetch into RAM ahead of the current GPU layer.
    /// Higher values trade RAM for smoother GPU utilisation.
    pub prefetch_ahead: usize,

    /// Call `madvise(MADV_SEQUENTIAL)` on the mmap region when the disk tier
    /// is active.  Lets the kernel read-ahead aggressively from NVMe.
    pub madvise_sequential: bool,

    /// Call `mlock` on RAM-tier weights to prevent the OS from swapping them.
    /// Requires sufficient `RLIMIT_MEMLOCK` (or run as root).
    pub mlock_ram_weights: bool,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            vram_reserved_bytes: 2 * GIB,
            ram_budget_bytes: u64::MAX,
            prefetch_ahead: 1,
            madvise_sequential: true,
            mlock_ram_weights: false,
        }
    }
}

// ── Layer placement ────────────────────────────────────────────────────────

/// Resolved placement for a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerPlacement {
    pub tier: MemoryTier,
}

// ── Vision encoder offload ─────────────────────────────────────────────────

/// Where to keep the vision encoder weights between image calls.
///
/// The encoder only runs during image prefill; after each image is processed
/// it can be evicted back to its storage tier, freeing VRAM for the backbone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisionOffloadPolicy {
    /// Planner decides: VRAM if budget remains after backbone, otherwise
    /// RAM, otherwise disk.  This is the right choice for almost all cases.
    #[default]
    Auto,
    /// Always keep the encoder resident in VRAM (fastest per-image call).
    PinVram,
    /// Keep in system RAM; DMA to GPU during prefill, evict after.
    PinRam,
    /// Stream from disk each image call (mmap); evict after.
    StreamDisk,
}

/// Vision encoder sizing and placement policy passed to the tier planner.
#[derive(Debug, Clone)]
pub struct VisionSpec {
    /// Total bytes of the vision encoder's weights (SigLIP-400M ≈ 1.7 GiB,
    /// CLIP-ViT-L ≈ 0.9 GiB, InternViT-300M ≈ 0.6 GiB).
    pub encoder_bytes: u64,
    /// How to place the encoder weights.  Default: `Auto`.
    pub policy: VisionOffloadPolicy,
}

/// Resolved placement for the vision encoder component.
#[derive(Debug, Clone)]
pub struct VisionEncoderPlacement {
    pub tier: MemoryTier,
    pub encoder_bytes: u64,
    /// When `true` the encoder is evicted back to its storage tier after
    /// each image prefill, freeing VRAM for the backbone decode loop.
    pub evict_after_prefill: bool,
}

impl VisionEncoderPlacement {
    fn resolve(spec: &VisionSpec, vram_remaining: u64, ram_bytes: u64) -> Self {
        let tier = match spec.policy {
            VisionOffloadPolicy::PinVram => MemoryTier::Vram,
            VisionOffloadPolicy::PinRam => MemoryTier::SystemRam,
            VisionOffloadPolicy::StreamDisk => MemoryTier::Disk,
            VisionOffloadPolicy::Auto => {
                if vram_remaining >= spec.encoder_bytes {
                    MemoryTier::Vram
                } else if ram_bytes >= spec.encoder_bytes {
                    MemoryTier::SystemRam
                } else {
                    MemoryTier::Disk
                }
            }
        };
        let evict_after_prefill = tier != MemoryTier::Vram;
        Self {
            tier,
            encoder_bytes: spec.encoder_bytes,
            evict_after_prefill,
        }
    }

    pub fn print_summary(&self) {
        let evict = if self.evict_after_prefill {
            " (evict after prefill)"
        } else {
            " (resident)"
        };
        println!(
            "  vision enc : {}  (~{:.1} GiB{})",
            self.tier.label(),
            self.encoder_bytes as f64 / GIB as f64,
            evict,
        );
    }
}

// ── Tiered placement plan ──────────────────────────────────────────────────

/// Full placement plan for all layers of a model.
///
/// Layers 0..n_vram_layers run fully on GPU.
/// Layers n_vram_layers..(n_vram_layers+n_ram_layers) are paged from RAM.
/// Remaining layers are streamed from disk via the mmap loader.
/// `vision` is `Some` when the model has a vision encoder.
#[derive(Debug, Clone)]
pub struct TieredPlacement {
    pub placements: Vec<LayerPlacement>,
    pub n_vram_layers: usize,
    pub n_ram_layers: usize,
    pub n_disk_layers: usize,
    /// Resolved placement for the vision encoder, if present.
    pub vision: Option<VisionEncoderPlacement>,
}

#[derive(Debug, Error)]
pub enum OffloadError {
    #[error("bytes_per_layer must be non-zero")]
    ZeroBytesPerLayer,
}

impl TieredPlacement {
    /// Plan layer placement given the available memory budgets.
    ///
    /// - `vram_bytes`: total VRAM capacity in bytes
    /// - `ram_bytes`: total system RAM available in bytes
    /// - `bytes_per_layer`: approximate weight bytes per transformer layer
    /// - `n_layers`: total transformer layers in the model
    /// - `cfg`: offload policy (reserves, mlock, etc.)
    /// - `vision`: optional vision encoder spec; `None` for text-only models
    pub fn plan(
        vram_bytes: u64,
        ram_bytes: u64,
        bytes_per_layer: u64,
        n_layers: usize,
        cfg: &OffloadConfig,
        vision: Option<&VisionSpec>,
    ) -> Result<Self, OffloadError> {
        if bytes_per_layer == 0 {
            return Err(OffloadError::ZeroBytesPerLayer);
        }

        let usable_vram = vram_bytes.saturating_sub(cfg.vram_reserved_bytes);
        let n_vram = ((usable_vram / bytes_per_layer) as usize).min(n_layers);

        let remaining = n_layers - n_vram;
        let ram_budget = cfg.ram_budget_bytes.min(ram_bytes);
        let n_ram = ((ram_budget / bytes_per_layer) as usize).min(remaining);

        let n_disk = remaining - n_ram;

        let mut placements = Vec::with_capacity(n_layers);
        for _ in 0..n_vram {
            placements.push(LayerPlacement {
                tier: MemoryTier::Vram,
            });
        }
        for _ in 0..n_ram {
            placements.push(LayerPlacement {
                tier: MemoryTier::SystemRam,
            });
        }
        for _ in 0..n_disk {
            placements.push(LayerPlacement {
                tier: MemoryTier::Disk,
            });
        }

        // Place vision encoder in remaining VRAM budget after backbone layers.
        let vram_used_by_backbone = n_vram as u64 * bytes_per_layer;
        let vram_remaining = usable_vram.saturating_sub(vram_used_by_backbone);
        let vision_placement =
            vision.map(|v| VisionEncoderPlacement::resolve(v, vram_remaining, ram_bytes));

        Ok(Self {
            placements,
            n_vram_layers: n_vram,
            n_ram_layers: n_ram,
            n_disk_layers: n_disk,
            vision: vision_placement,
        })
    }

    /// Human-readable summary of the tier assignment.
    pub fn print_summary(&self, vram_gib: u32, bytes_per_layer: u64) {
        let total = self.placements.len();
        println!("memory tiers : {} layers total", total);

        let vram_s = if vram_gib > 0 {
            format!("{} GiB VRAM", vram_gib)
        } else {
            "shared VRAM".into()
        };
        if self.n_vram_layers > 0 {
            println!(
                "  VRAM      : layers 0–{}  ({}, ~{} GiB weights)",
                self.n_vram_layers - 1,
                vram_s,
                bytes_per_layer * self.n_vram_layers as u64 / GIB,
            );
        }
        if self.n_ram_layers > 0 {
            let start = self.n_vram_layers;
            println!(
                "  RAM       : layers {}–{}  (~{} GiB weights — paged to GPU per step)",
                start,
                start + self.n_ram_layers - 1,
                bytes_per_layer * self.n_ram_layers as u64 / GIB,
            );
        }
        if self.n_disk_layers > 0 {
            let start = self.n_vram_layers + self.n_ram_layers;
            println!(
                "  disk/mmap : layers {}–{}  (~{} GiB — streamed from file, ~7 GB/s)",
                start,
                start + self.n_disk_layers - 1,
                bytes_per_layer * self.n_disk_layers as u64 / GIB,
            );
        }

        if let Some(ref v) = self.vision {
            v.print_summary();
        }

        // Speed warning when disk tier is active
        if self.n_disk_layers > 0 {
            println!();
            println!("  note: disk-tier layers are limited to ~7 GB/s (NVMe).");
            println!("        more RAM or a faster drive reduces per-token latency.");
        }
    }

    /// `true` if any layer falls below VRAM.
    pub fn has_offload(&self) -> bool {
        self.n_ram_layers > 0 || self.n_disk_layers > 0
    }

    /// `true` if any layer must be read from disk.
    pub fn has_disk_tier(&self) -> bool {
        self.n_disk_layers > 0
    }
}

// ── Quick-estimate helpers ─────────────────────────────────────────────────

/// Estimate bytes-per-layer for a model given its total parameter count
/// and an average bits-per-weight (e.g. 4.5 for Q4_K_M).
pub fn bytes_per_layer_estimate(total_params: u64, bits_per_weight: f32, n_layers: usize) -> u64 {
    if n_layers == 0 {
        return 0;
    }
    let total_bytes = (total_params as f32 * bits_per_weight / 8.0) as u64;
    total_bytes / n_layers as u64
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn all_fits_in_vram() {
        // 24 GiB VRAM, tiny model
        let plan = TieredPlacement::plan(
            24 * GIB,
            64 * GIB,
            500 * MIB,
            10,
            &OffloadConfig::default(),
            None,
        )
        .unwrap();
        assert_eq!(plan.n_vram_layers, 10);
        assert_eq!(plan.n_ram_layers, 0);
        assert_eq!(plan.n_disk_layers, 0);
        assert!(!plan.has_offload());
    }

    #[test]
    fn vram_overflow_into_ram() {
        // 8 GiB VRAM (2 GiB reserved → 6 usable), 500 MiB/layer
        // 6 GiB / 0.5 GiB = 12 layers in VRAM; remaining 68 in RAM
        let plan = TieredPlacement::plan(
            8 * GIB,
            2048 * GIB,
            500 * MIB,
            80,
            &OffloadConfig::default(),
            None,
        )
        .unwrap();
        assert_eq!(plan.n_vram_layers, 12);
        assert_eq!(plan.n_ram_layers, 68);
        assert_eq!(plan.n_disk_layers, 0);
        assert!(plan.has_offload());
        assert!(!plan.has_disk_tier());
    }

    #[test]
    fn ram_overflow_into_disk() {
        // 8 GiB VRAM, only 4 GiB RAM budget, 500 MiB/layer, 80 layers
        let cfg = OffloadConfig {
            ram_budget_bytes: 4 * GIB,
            ..Default::default()
        };
        let plan = TieredPlacement::plan(8 * GIB, 4 * GIB, 500 * MIB, 80, &cfg, None).unwrap();
        assert_eq!(plan.n_vram_layers, 12);
        assert_eq!(plan.n_ram_layers, 8); // 4 GiB / 0.5 GiB
        assert_eq!(plan.n_disk_layers, 60);
        assert!(plan.has_disk_tier());
    }

    #[test]
    fn zero_vram_all_ram() {
        // Apple Silicon: vram_gib=0 → vram_bytes=0, everything in RAM
        let plan =
            TieredPlacement::plan(0, 128 * GIB, 500 * MIB, 32, &OffloadConfig::default(), None)
                .unwrap();
        assert_eq!(plan.n_vram_layers, 0);
        assert_eq!(plan.n_ram_layers, 32);
        assert_eq!(plan.n_disk_layers, 0);
    }

    #[test]
    fn zero_bytes_per_layer_errors() {
        let result =
            TieredPlacement::plan(8 * GIB, 64 * GIB, 0, 80, &OffloadConfig::default(), None);
        assert!(result.is_err());
    }

    #[test]
    fn vision_fits_in_remaining_vram() {
        // 24 GiB VRAM, 10 layers × 500 MiB = 5 GiB backbone.
        // Usable after 2 GiB reserve = 22 GiB → all 10 layers in VRAM, 17 GiB left.
        // Vision encoder at 1.7 GiB fits in remaining VRAM.
        let spec = VisionSpec {
            encoder_bytes: (1700 * MIB),
            policy: VisionOffloadPolicy::Auto,
        };
        let plan = TieredPlacement::plan(
            24 * GIB,
            64 * GIB,
            500 * MIB,
            10,
            &OffloadConfig::default(),
            Some(&spec),
        )
        .unwrap();
        assert_eq!(plan.n_vram_layers, 10);
        let v = plan.vision.unwrap();
        assert_eq!(v.tier, MemoryTier::Vram);
        assert!(!v.evict_after_prefill);
    }

    #[test]
    fn vision_spills_to_ram_when_vram_full() {
        // 8 GiB VRAM → 12 backbone layers use 6 GiB, 0 bytes remaining for vision.
        // Vision encoder (0.9 GiB) goes to RAM.
        let spec = VisionSpec {
            encoder_bytes: 900 * MIB,
            policy: VisionOffloadPolicy::Auto,
        };
        let plan = TieredPlacement::plan(
            8 * GIB,
            2048 * GIB,
            500 * MIB,
            80,
            &OffloadConfig::default(),
            Some(&spec),
        )
        .unwrap();
        assert_eq!(plan.n_vram_layers, 12);
        let v = plan.vision.unwrap();
        assert_eq!(v.tier, MemoryTier::SystemRam);
        assert!(v.evict_after_prefill);
    }

    #[test]
    fn vision_pin_vram_ignores_budget() {
        // Force PinVram even when VRAM is full — caller's explicit request.
        let spec = VisionSpec {
            encoder_bytes: 900 * MIB,
            policy: VisionOffloadPolicy::PinVram,
        };
        let plan = TieredPlacement::plan(
            8 * GIB,
            2048 * GIB,
            500 * MIB,
            80,
            &OffloadConfig::default(),
            Some(&spec),
        )
        .unwrap();
        let v = plan.vision.unwrap();
        assert_eq!(v.tier, MemoryTier::Vram);
        assert!(!v.evict_after_prefill);
    }

    #[test]
    fn vision_stream_disk_policy() {
        let spec = VisionSpec {
            encoder_bytes: 900 * MIB,
            policy: VisionOffloadPolicy::StreamDisk,
        };
        let plan = TieredPlacement::plan(
            24 * GIB,
            64 * GIB,
            500 * MIB,
            10,
            &OffloadConfig::default(),
            Some(&spec),
        )
        .unwrap();
        let v = plan.vision.unwrap();
        assert_eq!(v.tier, MemoryTier::Disk);
        assert!(v.evict_after_prefill);
    }
}
