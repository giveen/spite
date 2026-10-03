//! Kernel path resolution.
//!
//! Produces a priority-ordered list of candidate directories for a given
//! `KernelSpec`. The caller (DispatchBuilder) walks the list and takes the
//! first directory that contains a loadable `.so`.
//!
//! # Model-specific resolution order
//!
//! ```text
//! kernels/<family>/<model>/<arch>/<card>/<quant>/   ← most specific
//! kernels/<family>/<model>/<arch>/<card>/
//! kernels/<family>/<model>/<arch>/<quant>/
//! kernels/<family>/<model>/<arch>/                  ← arch baseline
//! kernels/generic/<arch>/
//! kernels/generic/generic/                          ← always present
//! ```
//!
//! # Engine-level resolution order (cross-model features)
//!
//! ```text
//! kernels/_engine/<feature>/<arch>/<card>/          ← card specialist
//! kernels/_engine/<feature>/<arch>/                 ← arch baseline
//! kernels/_engine/<feature>/generic/                ← always present
//! ```
//!
//! Engine kernels implement ops that are independent of the model architecture:
//! speculative decoding verify, chunked prefill, KV cache quantisation, etc.
//! A card-specific engine kernel (e.g. an RTX 4090 MTP verifier that exploits
//! 72 MB L2 cache) drops into `_engine/<feature>/sm_89/rtx_4090/` and
//! automatically overrides the arch-level `_engine/<feature>/sm_89/`.

use std::path::{Path, PathBuf};

// ── KernelSpec ─────────────────────────────────────────────────────────────

/// All dimensions that determine which kernel binary to load.
#[derive(Debug, Clone, Default)]
pub struct KernelSpec {
    /// Model family directory: "llama", "deepseek", "qwen", …
    pub family:   String,
    /// Model variant directory: "llama3", "v3", "qwen3", …
    pub model:    String,
    /// GPU architecture: "sm_89", "rdna4", "cdna3", "metal", …
    pub gpu_arch: String,
    /// Specific card id: "rtx_4090", "mi300x", "rx_9900_xtx", …
    /// Empty string means "no card-specific override available".
    pub card_id:  String,
    /// Quantisation format: "Q4_K_M", "Q8_0", "F16", …
    /// Empty string means "any quant" (pick up the arch-level kernel).
    pub quant:    String,
}

impl KernelSpec {
    /// Convenience constructor for callers that only know the GGUF arch string
    /// and GPU arch (no family/model split, no card/quant).
    /// Suitable for the CLI and server when no richer spec is available.
    pub fn from_arch(model_arch: &str, gpu_arch: &str) -> Self {
        // Use the arch string as a stand-in for both family and model so that
        // model-specific kernels under kernels/<family>/<model>/ are found if
        // they exist; generic fallbacks still apply when they don't.
        Self {
            family:   model_arch.into(),
            model:    model_arch.into(),
            gpu_arch: gpu_arch.into(),
            card_id:  String::new(),
            quant:    String::new(),
        }
    }

    /// Candidate dirs for **model-specific** kernels, most-specific first.
    pub fn model_candidates(&self, kernels_dir: &Path) -> Vec<PathBuf> {
        let base = kernels_dir
            .join(&self.family)
            .join(&self.model)
            .join(&self.gpu_arch);

        let has_card  = !self.card_id.is_empty();
        let has_quant = !self.quant.is_empty();

        let mut paths = Vec::with_capacity(6);

        if has_card && has_quant {
            paths.push(base.join(&self.card_id).join(&self.quant));
        }
        if has_card {
            paths.push(base.join(&self.card_id));
        }
        if has_quant {
            paths.push(base.join(&self.quant));
        }
        paths.push(base);

        // Vendor-generic fallbacks (no model knowledge, but right vendor backend)
        if self.gpu_arch.starts_with("sm_") {
            paths.push(kernels_dir.join("generic").join("generic_cuda"));
        } else if self.gpu_arch.starts_with("rdna") || self.gpu_arch.starts_with("cdna") {
            paths.push(kernels_dir.join("generic").join("generic_rocm"));
        }

        paths.push(kernels_dir.join("generic").join(&self.gpu_arch));
        paths.push(kernels_dir.join("generic").join("generic"));
        paths
    }

    /// Candidate dirs for **engine-level** feature kernels, most-specific first.
    ///
    /// `feature` is a short identifier: "speculative", "prefill", "kv_quant".
    /// Engine kernels are cross-model — they live under `_engine/<feature>/`
    /// and are selected independently from the model forward-pass kernels.
    pub fn engine_candidates(&self, feature: &str, kernels_dir: &Path) -> Vec<PathBuf> {
        let base = kernels_dir.join("_engine").join(feature).join(&self.gpu_arch);
        let mut paths = Vec::with_capacity(3);

        if !self.card_id.is_empty() {
            paths.push(base.join(&self.card_id));
        }
        paths.push(base);
        paths.push(kernels_dir.join("_engine").join(feature).join("generic"));
        paths
    }
}

// ── Card detection ─────────────────────────────────────────────────────────

/// Map a GPU display name to the canonical `card_id` used in directory names.
///
/// Examples:
/// ```text
/// "NVIDIA GeForce RTX 4090"   → "rtx_4090"
/// "AMD Radeon RX 9900 XTX"    → "rx_9900_xtx"
/// "AMD Instinct MI300X"        → "mi300x"
/// "Intel Arc B580"             → "b580"
/// ```
///
/// Returns `""` if the name cannot be normalised; callers treat that as
/// "no card-specific override".
pub fn detect_card_id(gpu_display_name: &str) -> String {
    if let Ok(v) = std::env::var("SPITE_CARD_ID") {
        return v;
    }
    normalize_card_name(gpu_display_name)
}

fn normalize_card_name(name: &str) -> String {
    // Strip well-known vendor/product prefixes so we get the model designator.
    const PREFIXES: &[&str] = &[
        "NVIDIA GeForce RTX ", "NVIDIA GeForce GTX ",
        "NVIDIA GeForce ", "NVIDIA Quadro RTX ", "NVIDIA Quadro ",
        "NVIDIA RTX ", "NVIDIA ",
        "AMD Radeon RX ", "AMD Radeon Pro ", "AMD Radeon ",
        "AMD Instinct ", "AMD ",
        "Intel Arc ", "Intel Data Center GPU ", "Intel ",
    ];
    let mut s = name;
    for prefix in PREFIXES {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
            break;
        }
    }
    // Lowercase, collapse any run of non-alphanumeric to a single underscore.
    let mut out = String::with_capacity(s.len());
    let mut last_was_sep = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep && !out.is_empty() {
            out.push('_');
            last_was_sep = true;
        }
    }
    out.trim_end_matches('_').to_owned()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spec_full() -> KernelSpec {
        KernelSpec {
            family:   "llama".into(),
            model:    "llama3".into(),
            gpu_arch: "sm_89".into(),
            card_id:  "rtx_4090".into(),
            quant:    "Q4_K_M".into(),
        }
    }

    fn spec_arch_only() -> KernelSpec {
        KernelSpec {
            family:   "deepseek".into(),
            model:    "v3".into(),
            gpu_arch: "sm_89".into(),
            ..Default::default()
        }
    }

    #[test]
    fn model_candidates_full_spec() {
        let root = PathBuf::from("/k");
        let c = spec_full().model_candidates(&root);
        assert_eq!(c[0], PathBuf::from("/k/llama/llama3/sm_89/rtx_4090/Q4_K_M"));
        assert_eq!(c[1], PathBuf::from("/k/llama/llama3/sm_89/rtx_4090"));
        assert_eq!(c[2], PathBuf::from("/k/llama/llama3/sm_89/Q4_K_M"));
        assert_eq!(c[3], PathBuf::from("/k/llama/llama3/sm_89"));
        // generic_cuda before generic/<arch>
        assert_eq!(c[4], PathBuf::from("/k/generic/generic_cuda"));
        assert_eq!(c[5], PathBuf::from("/k/generic/sm_89"));
        assert_eq!(c[6], PathBuf::from("/k/generic/generic"));
    }

    #[test]
    fn model_candidates_arch_only() {
        let root = PathBuf::from("/k");
        let c = spec_arch_only().model_candidates(&root);
        assert_eq!(c[0], PathBuf::from("/k/deepseek/v3/sm_89"));
        assert_eq!(c[1], PathBuf::from("/k/generic/generic_cuda"));
        assert_eq!(c[2], PathBuf::from("/k/generic/sm_89"));
        assert_eq!(c[3], PathBuf::from("/k/generic/generic"));
    }

    #[test]
    fn engine_candidates_with_card() {
        let root = PathBuf::from("/k");
        let spec = KernelSpec {
            gpu_arch: "cdna3".into(),
            card_id:  "mi300x".into(),
            ..Default::default()
        };
        let c = spec.engine_candidates("prefill", &root);
        assert_eq!(c[0], PathBuf::from("/k/_engine/prefill/cdna3/mi300x"));
        assert_eq!(c[1], PathBuf::from("/k/_engine/prefill/cdna3"));
        assert_eq!(c[2], PathBuf::from("/k/_engine/prefill/generic"));
    }

    #[test]
    fn engine_candidates_no_card() {
        let root = PathBuf::from("/k");
        let spec = KernelSpec { gpu_arch: "sm_89".into(), ..Default::default() };
        let c = spec.engine_candidates("speculative", &root);
        assert_eq!(c[0], PathBuf::from("/k/_engine/speculative/sm_89"));
        assert_eq!(c[1], PathBuf::from("/k/_engine/speculative/generic"));
    }

    #[test]
    fn card_id_normalization() {
        assert_eq!(normalize_card_name("NVIDIA GeForce RTX 4090"), "rtx_4090");
        assert_eq!(normalize_card_name("AMD Radeon RX 9900 XTX"),  "rx_9900_xtx");
        assert_eq!(normalize_card_name("AMD Instinct MI300X"),      "mi300x");
        assert_eq!(normalize_card_name("Intel Arc B580"),           "b580");
        assert_eq!(normalize_card_name("AMD Radeon RX 7900 XTX"),  "rx_7900_xtx");
    }
}
