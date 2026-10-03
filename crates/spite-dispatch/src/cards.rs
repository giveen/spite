//! Card knowledge base: friendly GPU card names → GPU arch + VRAM.
//!
//! Users supply `--card RTX_5090` (or `"RTX 5090"`, `"rtx5090"` — all
//! normalise to `"rtx_5090"` via `resolve::normalize_card_name`).
//! `card_spec` maps the normalised id to the GPU arch string used in
//! the kernel directory hierarchy and the VRAM capacity.
//!
//! Adding a new card means adding one line to the match below.

/// Everything the engine needs to know about a GPU at startup.
#[derive(Debug, Clone, Copy)]
pub struct CardSpec {
    /// Kernel-directory arch string: "sm_120", "rdna4", "metal", …
    pub gpu_arch: &'static str,
    /// Dedicated VRAM in GiB (0 = unknown or shared / Apple Unified Memory).
    pub vram_gib: u32,
}

/// Look up a normalised card id (e.g. `"rtx_5090"`) and return its spec.
///
/// Returns `CardSpec { gpu_arch: "generic", vram_gib: 0 }` for unknown cards
/// so the engine always falls through to the generic kernel chain.
pub fn card_spec(card_id: &str) -> CardSpec {
    let (arch, vram) = match card_id {
        // ── NVIDIA Blackwell (sm_120) ─────────────────────────────────────
        "rtx_5090"          => ("sm_120", 32),
        "rtx_5080"          => ("sm_120", 16),
        "rtx_5070_ti"       => ("sm_120", 16),
        "rtx_5070"          => ("sm_120", 12),
        "rtx_5060_ti"       => ("sm_120", 16),
        "rtx_5060"          => ("sm_120",  8),
        // Blackwell data centre
        "b200" | "gb200"    => ("sm_100", 192),
        "b100"              => ("sm_100", 192),

        // ── NVIDIA Ada Lovelace (sm_89) ───────────────────────────────────
        "rtx_4090"          => ("sm_89", 24),
        "rtx_4080_super"    => ("sm_89", 16),
        "rtx_4080"          => ("sm_89", 16),
        "rtx_4070_ti_super" => ("sm_89", 16),
        "rtx_4070_ti"       => ("sm_89", 12),
        "rtx_4070_super"    => ("sm_89", 12),
        "rtx_4070"          => ("sm_89", 12),
        "rtx_4060_ti"       => ("sm_89",  8),
        "rtx_4060"          => ("sm_89",  8),
        // Ada data centre / workstation
        "rtx_6000_ada"      => ("sm_89", 48),
        "rtx_4500_ada"      => ("sm_89", 24),
        "rtx_4000_ada"      => ("sm_89", 20),

        // ── NVIDIA Hopper (sm_90) ─────────────────────────────────────────
        "h200"              => ("sm_90", 141),
        "h100"              => ("sm_90",  80),
        "h800"              => ("sm_90",  80),

        // ── NVIDIA Ampere (sm_86/sm_80) ───────────────────────────────────
        "rtx_3090_ti" | "rtx_3090" => ("sm_86", 24),
        "rtx_3080_ti"       => ("sm_86", 12),
        "rtx_3080"          => ("sm_86", 10),
        "rtx_3070_ti" | "rtx_3070" => ("sm_86",  8),
        "rtx_3060_ti" | "rtx_3060" => ("sm_86",  8),
        "a100"              => ("sm_80", 80),
        "a800"              => ("sm_80", 80),
        "a6000"             => ("sm_86", 48),
        "a5000"             => ("sm_86", 24),
        "a4000"             => ("sm_86", 16),

        // ── NVIDIA Turing (sm_75) ─────────────────────────────────────────
        "rtx_2080_ti"       => ("sm_75", 11),
        "rtx_2080_super" | "rtx_2080" => ("sm_75",  8),
        "rtx_2070_super" | "rtx_2070" => ("sm_75",  8),
        "rtx_2060_super" | "rtx_2060" => ("sm_75",  6),

        // ── NVIDIA Volta (sm_70) ──────────────────────────────────────────
        "v100"              => ("sm_70", 32),

        // ── AMD RDNA 4 ────────────────────────────────────────────────────
        "rx_9900_xtx"       => ("rdna4", 32),
        "rx_9900_xt"        => ("rdna4", 32),
        "rx_9800_xt"        => ("rdna4", 16),
        "rx_9070_xt"        => ("rdna4", 16),
        "rx_9070"           => ("rdna4", 16),
        "rx_9060_xt"        => ("rdna4",  8),

        // ── AMD RDNA 3 ────────────────────────────────────────────────────
        "rx_7900_xtx"       => ("rdna3", 24),
        "rx_7900_xt"        => ("rdna3", 20),
        "rx_7900_gre"       => ("rdna3", 16),
        "rx_7800_xt"        => ("rdna3", 16),
        "rx_7700_xt"        => ("rdna3", 12),
        "rx_7600_xt"        => ("rdna3",  8),
        "rx_7600"           => ("rdna3",  8),

        // ── AMD RDNA 2 ────────────────────────────────────────────────────
        "rx_6950_xt" | "rx_6900_xt" => ("rdna2", 16),
        "rx_6800_xt" | "rx_6800"    => ("rdna2", 16),
        "rx_6700_xt"        => ("rdna2", 12),
        "rx_6600_xt" | "rx_6600"    => ("rdna2",  8),

        // ── AMD CDNA 3 ────────────────────────────────────────────────────
        "mi350x"            => ("cdna3", 288),
        "mi325x"            => ("cdna3", 288),
        "mi300x"            => ("cdna3", 192),
        "mi300a"            => ("cdna3", 128),

        // ── AMD CDNA 2 ────────────────────────────────────────────────────
        "mi250x"            => ("cdna2", 128),
        "mi250"             => ("cdna2",  64),
        "mi210"             => ("cdna2",  64),

        // ── Apple Silicon (Metal, shared VRAM = 0) ────────────────────────
        // M4 family
        "m4_ultra"          => ("metal",   0),
        "m4_max"            => ("metal",   0),
        "m4_pro"            => ("metal",   0),
        "m4"                => ("metal",   0),
        // M3 family
        "m3_max"            => ("metal",   0),
        "m3_pro"            => ("metal",   0),
        "m3"                => ("metal",   0),
        // M2 family
        "m2_ultra"          => ("metal",   0),
        "m2_max"            => ("metal",   0),
        "m2_pro"            => ("metal",   0),
        "m2"                => ("metal",   0),
        // M1 family
        "m1_ultra"          => ("metal",   0),
        "m1_max"            => ("metal",   0),
        "m1_pro"            => ("metal",   0),
        "m1"                => ("metal",   0),
        // M5 (announced)
        "m5_ultra"          => ("metal",   0),
        "m5_max"            => ("metal",   0),
        "m5_pro"            => ("metal",   0),
        "m5"                => ("metal",   0),

        // ── Intel Arc Battlemage (Xe2) ────────────────────────────────────
        "b770"              => ("xe2", 16),
        "b580"              => ("xe2", 12),
        "b580m"             => ("xe2",  8),

        // ── Intel Arc Alchemist (Xe HPG) ──────────────────────────────────
        "a770"              => ("xe_hpg", 16),
        "a750"              => ("xe_hpg",  8),
        "a580"              => ("xe_hpg",  8),
        "a380"              => ("xe_hpg",  6),

        _ => ("generic", 0),
    };
    CardSpec { gpu_arch: arch, vram_gib: vram }
}
