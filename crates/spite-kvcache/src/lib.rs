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

pub mod paged;
pub mod persist;
pub mod quant;
pub mod vbr;

pub use quant::{dequantize, packed_bytes, quantize};
pub use vbr::{VbrPolicy, VbrRows};

use spite_abi::{SpiteKvCache, SpiteTensor};
use thiserror::Error;

// ── KV cache quantization ─────────────────────────────────────────────────

/// Precision level for one side (K or V) of the KV cache.
///
/// Classic GGML quant types only — no Turbo FWHT/PolarQuant/TCQ kernels.
/// `F16` is the default; move down the ladder when VRAM is tight.
///
/// Degradation ladder (high → low quality):
///   F32 (4.0) → F16 (2.0 bpe) → Q8 (1.06 bpe) → Q5_1 (0.75 bpe) → Q4 (0.56 bpe)
///
/// The engine degrades automatically as the sequence grows — you just set the
/// tier you want to start at. See [`vbr`](crate::vbr) for the trigger policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KvQuant {
    /// Full precision — reference / debug only.  (4 bytes/element)
    F32,
    /// 16-bit float — default, no quality cost.  (2 bytes/element)
    F16,
    /// 8-bit uniform (q8_0) — imperceptible quality loss.  (~1.06 bytes/element)
    Q8,
    /// 5-bit with scale+bias (q5_1) — good quality/size balance.  (0.75 bytes/element)
    Q5_1,
    /// 4-bit uniform (q4_0) — mild loss at long context.  (~0.56 bytes/element)
    Q4,
}

impl KvQuant {
    /// Storage bytes per element (GGML block sizes, 32-element blocks).
    pub fn bytes_per_elem(self) -> f32 {
        match self {
            // Exact GGML block footprints:
            // q8_0:  { f16 delta, i8[32] }    = 34 bytes / 32 elems = 1.0625
            // q5_1:  { f16 d, f16 m, u32 qh, u8[16] } = 24 bytes / 32 elems = 0.75
            // q4_0:  { f16 delta, u8[16] }    = 18 bytes / 32 elems = 0.5625
            Self::F32 => 4.0,
            Self::F16 => 2.0,
            Self::Q8 => 1.0625,
            Self::Q5_1 => 0.75,
            Self::Q4 => 0.5625,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8 => "q8_0",
            Self::Q5_1 => "q5_1",
            Self::Q4 => "q4_0",
        }
    }

    /// Next tier down the degradation ladder (`None` if already at floor).
    pub fn degrade(self) -> Option<Self> {
        match self {
            Self::F32 => Some(Self::F16),
            Self::F16 => Some(Self::Q8),
            Self::Q8 => Some(Self::Q5_1),
            Self::Q5_1 => Some(Self::Q4),
            Self::Q4 => None,
        }
    }

    /// Next tier up the ladder (`None` if already at the top).
    ///
    /// Used when a kernel cannot read the requested tier: the start tier is
    /// walked up until one the kernel accepts is found.
    pub fn upgrade(self) -> Option<Self> {
        match self {
            Self::F32 => None,
            Self::F16 => Some(Self::F32),
            Self::Q8 => Some(Self::F16),
            Self::Q5_1 => Some(Self::Q8),
            Self::Q4 => Some(Self::Q5_1),
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
            "f32" => Ok(Self::F32),
            "f16" | "fp16" => Ok(Self::F16),
            "q8" | "q8_0" => Ok(Self::Q8),
            "q5_1" | "q5" => Ok(Self::Q5_1),
            "q4" | "q4_0" => Ok(Self::Q4),
            other => Err(format!(
                "unknown KV quant '{other}'; use f16, q8, q5_1, or q4"
            )),
        }
    }
}

/// Quantization policy for the KV cache.
///
/// K and V are independently settable (K is more attention-sensitive than V,
/// so a common configuration is K=q8, V=q5_1 or K=q8, V=q4).
///
/// The engine degrades both sides automatically as the sequence grows (at 1/4,
/// 1/2 and 3/4 of the context window). You set the preferred starting tier; no
/// separate "auto" flag is needed.
///
/// # How to choose
///
/// | Scenario                         | Setting              |
/// |----------------------------------|----------------------|
/// | Plenty of VRAM (default)         | `KvQuantConfig::default()` — f16 |
/// | Tight VRAM, quality first        | key=Q8, val=Q8       |
/// | Tight VRAM, balanced             | key=Q8, val=Q5_1     |
/// | Very tight VRAM                  | key=Q8, val=Q4       |
#[derive(Debug, Clone)]
pub struct KvQuantConfig {
    /// Quantization for key tensors.  Default: F16.
    pub key: KvQuant,
    /// Quantization for value tensors.  Default: F16.
    pub val: KvQuant,
}

impl Default for KvQuantConfig {
    fn default() -> Self {
        Self {
            key: KvQuant::F16,
            val: KvQuant::F16,
        }
    }
}

impl std::str::FromStr for KvQuantConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.trim().splitn(2, ',').collect();
        let key: KvQuant = parts[0].parse()?;
        let val: KvQuant = parts.get(1).map(|p| p.parse()).transpose()?.unwrap_or(key);
        Ok(Self { key, val })
    }
}

impl KvQuantConfig {
    /// Estimated VRAM bytes for the KV cache given model dimensions.
    pub fn vram_bytes(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
    ) -> u64 {
        let elems = (n_layers * max_ctx * n_kv_heads * head_dim) as f64;
        let k_bytes = elems * self.key.bytes_per_elem() as f64;
        let v_bytes = elems * self.val.bytes_per_elem() as f64;
        (k_bytes + v_bytes) as u64
    }

    pub fn is_default(&self) -> bool {
        self.key == KvQuant::F16 && self.val == KvQuant::F16
    }

    /// Full-precision *starting* tier (no quantization at short context).
    ///
    /// The CPU model paths default to this so their outputs stay exact for the
    /// short prompts used by reference tests. VBR still applies from here: at
    /// long context the cache degrades down the ladder from f32. The executor
    /// overrides the start tier via [`KvQuantConfig::default`] before inference.
    pub fn full_precision() -> Self {
        Self {
            key: KvQuant::F32,
            val: KvQuant::F32,
        }
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
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_ctx: usize,
    pub strategy: CacheStrategy,
    /// KV quantization policy.  Default: both K and V at f16.
    pub quant: KvQuantConfig,
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
    cfg: CacheConfig,
    /// Flat GPU buffer: [n_layers, max_ctx, n_kv_heads, head_dim] × 2 (K+V).
    /// Layout is interleaved by layer so each layer's forward pass touches
    /// a contiguous region.
    // TODO: replace with an actual GPU allocation handle
    _buf: Vec<u8>,
    /// Current write head. Wraps at max_ctx.
    head: usize,
    /// How many token positions are actually populated.
    len: usize,
}

impl RingCache {
    pub fn new(cfg: CacheConfig) -> Result<Self, CacheError> {
        let elems = cfg.n_layers * cfg.max_ctx * cfg.n_kv_heads * cfg.head_dim;
        let k_bytes = (elems as f64 * cfg.quant.key.bytes_per_elem() as f64) as usize;
        let v_bytes = (elems as f64 * cfg.quant.val.bytes_per_elem() as f64) as usize;
        let bytes = k_bytes + v_bytes;

        // TODO: allocate on GPU (cudaMalloc / hipMalloc / Metal buffer)
        let _buf = vec![0u8; bytes];

        Ok(Self {
            cfg,
            _buf,
            head: 0,
            len: 0,
        })
    }

    /// Returns the SpiteKvCache view for `layer` at the current write head.
    /// The kernel writes K/V into the returned tensors.
    pub fn layer_view(&mut self, layer: usize) -> Result<SpiteKvCache, CacheError> {
        if layer >= self.cfg.n_layers {
            return Err(CacheError::LayerOutOfRange(layer));
        }
        // TODO: compute pointer offset into _buf, wrap SpiteTensor around it
        Ok(SpiteKvCache {
            k: SpiteTensor::null(),
            v: SpiteTensor::null(),
            layer: layer as i32,
        })
    }

    /// Advance the write head after a token is committed.
    pub fn commit(&mut self) {
        self.head = (self.head + 1) % self.cfg.max_ctx;
        self.len = (self.len + 1).min(self.cfg.max_ctx);
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn is_full(&self) -> bool {
        self.len == self.cfg.max_ctx
    }
}

// ── Prefix cache ──────────────────────────────────────────────────────────

/// A single cached prefix entry.
pub struct PrefixEntry {
    /// Hash of the token sequence that produced this cache entry.
    pub token_hash: u64,
    /// Number of tokens in this prefix.
    pub n_tokens: usize,
    /// KV state for each layer — GPU buffers.
    // TODO: actual GPU allocation handles per layer
    pub layers: Vec<()>,
}

pub struct PrefixCache {
    cfg: CacheConfig,
    entries: Vec<PrefixEntry>,
    capacity: usize,
}

impl PrefixCache {
    pub fn new(cfg: CacheConfig, capacity: usize) -> Self {
        Self {
            cfg,
            entries: Vec::new(),
            capacity,
        }
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
            n_tokens: tokens.len(),
            layers: vec![(); self.cfg.n_layers],
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
    use std::str::FromStr;

    #[test]
    fn kv_quant_parse_symmetric() {
        let cfg = KvQuantConfig::from_str("q8").unwrap();
        assert_eq!(cfg.key, KvQuant::Q8);
        assert_eq!(cfg.val, KvQuant::Q8);
    }

    #[test]
    fn kv_quant_parse_asymmetric() {
        let cfg = KvQuantConfig::from_str("q8,q4").unwrap();
        assert_eq!(cfg.key, KvQuant::Q8);
        assert_eq!(cfg.val, KvQuant::Q4);
    }

    #[test]
    fn kv_quant_parse_q5_1() {
        let cfg = KvQuantConfig::from_str("q5_1").unwrap();
        assert_eq!(cfg.key, KvQuant::Q5_1);
        assert_eq!(cfg.val, KvQuant::Q5_1);
    }

    #[test]
    fn kv_quant_parse_asymmetric_with_q5() {
        let cfg = KvQuantConfig::from_str("q8,q5_1").unwrap();
        assert_eq!(cfg.key, KvQuant::Q8);
        assert_eq!(cfg.val, KvQuant::Q5_1);
    }

    #[test]
    fn kv_quant_parse_f16_is_default() {
        let cfg = KvQuantConfig::from_str("f16").unwrap();
        assert!(cfg.is_default());
    }

    #[test]
    fn kv_quant_degrade_ladder() {
        assert_eq!(KvQuant::F16.degrade(), Some(KvQuant::Q8));
        assert_eq!(KvQuant::Q8.degrade(), Some(KvQuant::Q5_1));
        assert_eq!(KvQuant::Q5_1.degrade(), Some(KvQuant::Q4));
        assert_eq!(KvQuant::Q4.degrade(), None);
    }

    #[test]
    fn kv_quant_bytes_per_elem() {
        assert_eq!(KvQuant::F16.bytes_per_elem(), 2.0);
        assert_eq!(KvQuant::Q8.bytes_per_elem(), 1.0625);
        assert_eq!(KvQuant::Q5_1.bytes_per_elem(), 0.75);
        assert_eq!(KvQuant::Q4.bytes_per_elem(), 0.5625);
    }

    #[test]
    fn ring_cache_sizes_with_quant() {
        // 4 layers, 128 ctx, 8 heads, 64 head_dim
        let base_cfg = CacheConfig {
            n_layers: 4,
            n_kv_heads: 8,
            head_dim: 64,
            max_ctx: 128,
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
            quant: KvQuantConfig::from_str("q8,q5_1").unwrap(),
            ..base_cfg
        };
        let _asym_cache = RingCache::new(asym_cfg).unwrap();
        // K at q8 (1.0625 bpe) + V at q5_1 (0.75 bpe)
    }
}
