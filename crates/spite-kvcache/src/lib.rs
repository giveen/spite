//! KV cache management.
//!
//! Two cache strategies:
//!
//! ## RingCache — single-session inference
//! Fixed-size contiguous allocation. When the context fills, the oldest
//! tokens are evicted (sliding window). Simple, fast, zero fragmentation.
//! Right choice for single-user interactive inference.
//!
//! ## PrefixCache — server-side inference
//! Caches KV state for shared prefixes (system prompts, few-shot examples).
//! When a new request starts with a prefix that's already cached, the
//! forward pass skips those tokens entirely. Large win when many sessions
//! share the same system prompt.

pub mod persist;
pub mod paged;

use spite_abi::{SpiteKvCache, SpiteTensor};
use thiserror::Error;

// ── KV cache quantization ─────────────────────────────────────────────────

/// Precision level for one side (K or V) of the KV cache.
///
/// These map to the "classic" GGML quant types — no Turbo kernels required.
/// `F16` is the default; it imposes no quality cost and works everywhere.
/// Move to `Q8` first (imperceptible loss, 50% savings), then `Q4` only when
/// VRAM is genuinely tight (some quality degradation at long context).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KvQuant {
    /// Full precision — reference / debug only.
    F32,
    /// 16-bit float — default, no quality cost. (2 bytes/element)
    F16,
    /// 8-bit uniform (q8_0) — 50% savings vs F16, imperceptible quality loss.
    Q8,
    /// 4-bit uniform (q4_0) — 75% savings vs F16, mild loss at long context.
    Q4,
}

impl KvQuant {
    /// Storage bytes per element.
    pub fn bytes_per_elem(self) -> f32 {
        match self {
            Self::F32 => 4.0,
            Self::F16 => 2.0,
            Self::Q8  => 1.0,
            Self::Q4  => 0.5,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8  => "q8_0",
            Self::Q4  => "q4_0",
        }
    }

    /// The next tier down the degradation ladder (`None` if already at floor).
    pub fn degrade(self) -> Option<Self> {
        match self {
            Self::F32 => Some(Self::F16),
            Self::F16 => Some(Self::Q8),
            Self::Q8  => Some(Self::Q4),
            Self::Q4  => None,
        }
    }
}

impl std::fmt::Display for KvQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for KvQuant {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32"            => Ok(Self::F32),
            "f16" | "fp16"   => Ok(Self::F16),
            "q8" | "q8_0"    => Ok(Self::Q8),
            "q4" | "q4_0"    => Ok(Self::Q4),
            other => Err(format!("unknown KV quant type '{other}'; use f16, q8, or q4")),
        }
    }
}

/// Quantization policy for the KV cache.
///
/// The VBR (variable bit-rate) design: K and V can use different quant types,
/// and `dynamic` mode degrades them automatically as VRAM pressure mounts.
///
/// # How to choose
///
/// | Scenario                          | Setting                       |
/// |-----------------------------------|-------------------------------|
/// | Plenty of VRAM (default)          | `KvQuantConfig::default()`    |
/// | Tight VRAM, care about quality    | key=Q8, val=Q8                |
/// | Very tight VRAM                   | key=Q8, val=Q4 (K is sensitive in attention) |
/// | Let the engine decide at runtime  | `dynamic = true`              |
#[derive(Debug, Clone)]
pub struct KvQuantConfig {
    /// Quantization for key tensors.  Default: F16.
    pub key: KvQuant,
    /// Quantization for value tensors.  Default: F16.
    pub val: KvQuant,
    /// VBR dynamic mode: automatically degrade K and V tiers as the context
    /// window fills and KV VRAM pressure grows.
    /// Ladder: F16 → Q8 → Q4 (never below `floor`).
    pub dynamic: bool,
    /// Lowest permitted tier when `dynamic = true`.  Default: Q4.
    pub floor: KvQuant,
    /// Explicit KV VRAM budget in bytes.  0 = auto (infer from free VRAM).
    pub vram_budget_bytes: u64,
}

impl Default for KvQuantConfig {
    fn default() -> Self {
        Self {
            key:               KvQuant::F16,
            val:               KvQuant::F16,
            dynamic:           false,
            floor:             KvQuant::Q4,
            vram_budget_bytes: 0,
        }
    }
}

impl KvQuantConfig {
    /// Parse from a CLI string:
    ///   "f16"      → both K and V at f16
    ///   "q8"       → both K and V at q8_0
    ///   "q8,q4"    → K at q8_0, V at q4_0 (K is more sensitive)
    ///   "auto"     → dynamic VBR degradation (starts at f16)
    pub fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self { dynamic: true, ..Default::default() });
        }
        let parts: Vec<&str> = s.splitn(2, ',').collect();
        let key: KvQuant = parts[0].parse()?;
        let val: KvQuant = parts.get(1).map(|p| p.parse()).transpose()?.unwrap_or(key);
        Ok(Self { key, val, ..Default::default() })
    }

    /// Estimated VRAM bytes for the KV cache given model dimensions.
    pub fn vram_bytes(
        &self,
        n_layers:   usize,
        n_kv_heads: usize,
        head_dim:   usize,
        max_ctx:    usize,
    ) -> u64 {
        let k_bytes = (n_layers * max_ctx * n_kv_heads * head_dim) as f64
            * self.key.bytes_per_elem() as f64;
        let v_bytes = (n_layers * max_ctx * n_kv_heads * head_dim) as f64
            * self.val.bytes_per_elem() as f64;
        (k_bytes + v_bytes) as u64
    }

    pub fn is_default(&self) -> bool {
        self.key == KvQuant::F16 && self.val == KvQuant::F16 && !self.dynamic
    }
}

// ── Cache trait ───────────────────────────────────────────────────────────

/// The pluggable KV cache interface.
///
/// Implement this to swap the cache strategy (ring, prefix, paged, disk-
/// backed, quantized, …) for a specific model or deployment context,
/// without touching the executor or scheduler.
///
/// Register implementations in a `Registry<dyn Cache>`.
pub trait Cache: Send + Sync {
    /// Allocate cache space for `seq_id` to hold `n_tokens` positions.
    /// Called before the prefill pass.
    fn prepare(&mut self, seq_id: u64, n_tokens: usize) -> Result<(), CacheError>;

    /// Commit the most recently written position to `seq_id`.
    /// Called after each decode step.
    fn commit(&mut self, seq_id: u64);

    /// Release all cache pages held by `seq_id`.
    fn free(&mut self, seq_id: u64);

    /// Return the K and V tensors for `(seq_id, layer)` if available.
    fn kv_view(&self, seq_id: u64, layer: usize) -> Option<(&SpiteTensor, &SpiteTensor)>;

    /// Current number of cached tokens for `seq_id`.
    fn len(&self, seq_id: u64) -> usize;
}

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("context length {requested} exceeds cache capacity {capacity}")]
    ContextTooLong { requested: usize, capacity: usize },
    #[error("layer {0} out of range")]
    LayerOutOfRange(usize),
    #[error("allocation failed: {0}")]
    Alloc(String),
}

// ── Cache configuration ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub n_layers:   usize,
    pub n_kv_heads: usize,
    pub head_dim:   usize,
    pub max_ctx:    usize,
    pub strategy:   CacheStrategy,
    /// KV quantization policy.  Default: both K and V at f16.
    pub quant:      KvQuantConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Evict oldest tokens when full. Good for single-user.
    Ring,
    /// Keep a hash-indexed prefix cache. Good for serving.
    Prefix,
}

// ── Ring cache ────────────────────────────────────────────────────────────

pub struct RingCache {
    cfg:     CacheConfig,
    /// Flat GPU buffer: [n_layers, max_ctx, n_kv_heads, head_dim] × 2 (K+V).
    /// Layout is interleaved by layer so each layer's forward pass touches
    /// a contiguous region.
    // TODO: replace with an actual GPU allocation handle
    _buf:    Vec<u8>,
    /// Current write head. Wraps at max_ctx.
    head:    usize,
    /// How many token positions are actually populated.
    len:     usize,
}

impl RingCache {
    pub fn new(cfg: CacheConfig) -> Result<Self, CacheError> {
        let elems = cfg.n_layers * cfg.max_ctx * cfg.n_kv_heads * cfg.head_dim;
        let k_bytes = (elems as f64 * cfg.quant.key.bytes_per_elem() as f64) as usize;
        let v_bytes = (elems as f64 * cfg.quant.val.bytes_per_elem() as f64) as usize;
        let bytes = k_bytes + v_bytes;

        // TODO: allocate on GPU (cudaMalloc / hipMalloc / Metal buffer)
        let _buf = vec![0u8; bytes];

        Ok(Self { cfg, _buf, head: 0, len: 0 })
    }

    /// Returns the SpiteKvCache view for `layer` at the current write head.
    /// The kernel writes K/V into the returned tensors.
    pub fn layer_view(&mut self, layer: usize) -> Result<SpiteKvCache, CacheError> {
        if layer >= self.cfg.n_layers {
            return Err(CacheError::LayerOutOfRange(layer));
        }
        // TODO: compute pointer offset into _buf, wrap SpiteTensor around it
        Ok(SpiteKvCache {
            k:     SpiteTensor::null(),
            v:     SpiteTensor::null(),
            layer: layer as i32,
        })
    }

    /// Advance the write head after a token is committed.
    pub fn commit(&mut self) {
        self.head = (self.head + 1) % self.cfg.max_ctx;
        self.len  = (self.len + 1).min(self.cfg.max_ctx);
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_full(&self) -> bool { self.len == self.cfg.max_ctx }
}

// ── Prefix cache ──────────────────────────────────────────────────────────

/// A single cached prefix entry.
pub struct PrefixEntry {
    /// Hash of the token sequence that produced this cache entry.
    pub token_hash: u64,
    /// Number of tokens in this prefix.
    pub n_tokens:   usize,
    /// KV state for each layer — GPU buffers.
    // TODO: actual GPU allocation handles per layer
    pub layers:     Vec<()>,
}

pub struct PrefixCache {
    cfg:     CacheConfig,
    entries: Vec<PrefixEntry>,
    capacity: usize,
}

impl PrefixCache {
    pub fn new(cfg: CacheConfig, capacity: usize) -> Self {
        Self { cfg, entries: Vec::new(), capacity }
    }

    /// Look up a cached prefix by token sequence hash.
    /// Returns the number of prefix tokens that can be skipped, or 0.
    pub fn lookup(&self, tokens: &[u32]) -> usize {
        let hash = hash_tokens(tokens);
        self.entries
            .iter()
            .find(|e| e.token_hash == hash && e.n_tokens <= tokens.len())
            .map_or(0, |e| e.n_tokens)
    }

    /// Store the current KV state as a prefix cache entry.
    pub fn store(&mut self, tokens: &[u32], _cache: &RingCache) -> Result<(), CacheError> {
        if self.entries.len() >= self.capacity {
            // Evict least-recently-used entry.
            // TODO: proper LRU tracking
            self.entries.remove(0);
        }
        self.entries.push(PrefixEntry {
            token_hash: hash_tokens(tokens),
            n_tokens:   tokens.len(),
            layers:     vec![(); self.cfg.n_layers],
        });
        Ok(())
    }
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    // FNV-1a over the token bytes — fast, good enough for cache keys
    let mut h = 0xcbf29ce484222325u64;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_quant_parse_symmetric() {
        let cfg = KvQuantConfig::from_str("q8").unwrap();
        assert_eq!(cfg.key, KvQuant::Q8);
        assert_eq!(cfg.val, KvQuant::Q8);
        assert!(!cfg.dynamic);
    }

    #[test]
    fn kv_quant_parse_asymmetric() {
        let cfg = KvQuantConfig::from_str("q8,q4").unwrap();
        assert_eq!(cfg.key, KvQuant::Q8);
        assert_eq!(cfg.val, KvQuant::Q4);
    }

    #[test]
    fn kv_quant_parse_auto() {
        let cfg = KvQuantConfig::from_str("auto").unwrap();
        assert!(cfg.dynamic);
    }

    #[test]
    fn kv_quant_parse_f16_is_default() {
        let cfg = KvQuantConfig::from_str("f16").unwrap();
        assert!(cfg.is_default());
    }

    #[test]
    fn kv_quant_degrade_ladder() {
        assert_eq!(KvQuant::F16.degrade(), Some(KvQuant::Q8));
        assert_eq!(KvQuant::Q8.degrade(),  Some(KvQuant::Q4));
        assert_eq!(KvQuant::Q4.degrade(),  None);
    }

    #[test]
    fn kv_quant_bytes_per_elem() {
        assert_eq!(KvQuant::F16.bytes_per_elem(), 2.0);
        assert_eq!(KvQuant::Q8.bytes_per_elem(),  1.0);
        assert_eq!(KvQuant::Q4.bytes_per_elem(),  0.5);
    }

    #[test]
    fn ring_cache_sizes_with_quant() {
        // 4 layers, 128 ctx, 8 heads, 64 head_dim
        let base_cfg = CacheConfig {
            n_layers: 4, n_kv_heads: 8, head_dim: 64, max_ctx: 128,
            strategy: CacheStrategy::Ring,
            quant: KvQuantConfig::default(),
        };
        let f16_cache = RingCache::new(base_cfg.clone()).unwrap();
        // 4 * 128 * 8 * 64 * 2 (f16) * 2 (K+V) = 1_048_576 bytes
        drop(f16_cache);

        let q8_cfg = CacheConfig {
            quant: KvQuantConfig::from_str("q8").unwrap(),
            ..base_cfg.clone()
        };
        let _q8_cache = RingCache::new(q8_cfg).unwrap();
        // K+V both at q8: 4 * 128 * 8 * 64 * 1.0 * 2 = 524_288 bytes (50% smaller)

        let asym_cfg = CacheConfig {
            quant: KvQuantConfig::from_str("q8,q4").unwrap(),
            ..base_cfg
        };
        let _asym_cache = RingCache::new(asym_cfg).unwrap();
        // K at q8 (1 bpe) + V at q4 (0.5 bpe): 4 * 128 * 8 * 64 * 1.5 = 393_216 bytes
    }
}
