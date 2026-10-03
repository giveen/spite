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

use spite_abi::{SpiteKvCache, SpiteTensor};
use thiserror::Error;

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
        let elem_size = 2; // fp16
        let bytes = cfg.n_layers
            * cfg.max_ctx
            * cfg.n_kv_heads
            * cfg.head_dim
            * elem_size
            * 2; // K and V

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
