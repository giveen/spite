//! Forward pass executor.
//!
//! Wires together: loaded model weights → dispatch kernels → KV cache →
//! sampling → output tokens. The key feature over a plain model call is
//! the **hybrid CPU+GPU layer split**: the first `n_gpu_layers` transformer
//! layers run through the dispatch table (GPU kernels), the remainder on
//! the CPU fallback. Both paths use the same `ModelArch` interface.
//!
//! Typical life cycle:
//!   let mut exec = Executor::new(cfg, model, dispatch, kvcache);
//!   let logits = exec.prefill(&prompt_tokens, &ctx)?;   // populate KV cache
//!   loop {
//!       let token = sampler.sample(&logits);
//!       if token == eos { break; }
//!       output.push(token);
//!       let logits = exec.decode_step(token, &ctx)?;    // single-token step
//!   }

use spite_abi::{SpiteCtx, ShardStrategy};
use spite_plugin::{PluginKey, Registry};
use spite_sampling::{Sampler, DefaultSampler};
use spite_tokenizer::Tokenize;
use spite_kvcache::{Cache, KvQuantConfig};
use spite_offload::OffloadConfig;
use thiserror::Error;

// ── Per-request overrides ─────────────────────────────────────────────────

/// Per-inference-call override bundle.
///
/// Every field is optional: `None` means "use the engine's registered
/// default for this request's `PluginKey`".  Populate only the fields
/// you actually want to replace — nothing else is affected.
///
/// # Example
///
/// ```text
/// // Use a custom greedy sampler for this one request:
/// let overrides = InferenceOverrides {
///     key:     PluginKey::for_task("summarize"),
///     sampler: Some(Box::new(GreedySampler)),
///     ..Default::default()
/// };
/// executor.prefill(&tokens, &ctx, &overrides)?;
/// ```
#[derive(Default)]
pub struct InferenceOverrides {
    /// Resolution key used when looking up registry defaults.
    /// Set `model_arch`, `gpu_arch`, or `task` to narrow the match.
    pub key: PluginKey,

    /// Replace the entire sampling pipeline for this call.
    /// Overrides any registry registration for `key`.
    pub sampler: Option<Box<dyn Sampler>>,

    /// Replace the tokenizer for this call.
    /// Useful for domain-specific vocabularies or custom special-token handling.
    pub tokenizer: Option<Box<dyn Tokenize>>,

    /// Replace the KV cache backend for this call.
    /// Useful for memory-constrained scenarios or custom eviction policies.
    pub cache: Option<Box<dyn Cache>>,
}

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("model forward failed: {0}")]
    Model(String),
    #[error("gpu error: {0}")]
    Gpu(String),
    #[error("executor not yet initialized — call prefill first")]
    NotInitialized,
    #[error("context length exceeded: {used} > {max}")]
    ContextOverflow { used: usize, max: usize },
}

// Re-export so callers only need one import.
pub use spite_abi::ShardStrategy as LayerSplit;

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub layer_split:  ShardStrategy,
    pub n_threads:    usize,
    pub ctx_len:      usize,
    pub batch_size:   usize,
    /// Lock model weights in physical memory to prevent swapping.
    pub mlock:        bool,
    /// Advise the OS that mmap'd weights will be accessed sequentially.
    pub madvise_seq:  bool,
    /// Weight offload policy (VRAM → RAM → disk).
    /// `None` means keep everything in VRAM; overflow panics if VRAM is insufficient.
    pub offload:      Option<OffloadConfig>,
    /// KV cache quantization.  Default: both K and V at f16.
    pub kv_quant:     KvQuantConfig,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            layer_split: ShardStrategy::None,
            n_threads:   4,
            ctx_len:     4096,
            batch_size:  512,
            mlock:       false,
            madvise_seq: true,
            offload:     None,
            kv_quant:    KvQuantConfig::default(),
        }
    }
}

/// Owns the model, dispatch table, KV cache, and GPU activation buffers.
///
/// One `Executor` per loaded model; share it across requests using
/// `spite-scheduler` for concurrency.
pub struct Executor {
    pub cfg: ExecutorConfig,
    // TODO: model:    Box<dyn spite_models::ModelArch>
    // TODO: dispatch: spite_dispatch::DispatchTable
    // TODO: kvcache:  spite_kvcache::RingCache  (or PrefixCache for server mode)
    // TODO: gpu_bufs: Vec<spite_gpu::DeviceBuffer>  (one activation buffer per layer)
    n_ctx_used: usize,
}

impl Executor {
    pub fn new(cfg: ExecutorConfig) -> Self {
        Self { cfg, n_ctx_used: 0 }
    }

    /// Prefill: run a batched forward pass over `tokens`, populating the KV cache.
    ///
    /// Returns `[vocab_size]` logits for the last token position.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        ctx:    &SpiteCtx,
    ) -> Result<Vec<f32>, ExecutorError> {
        let max = self.cfg.ctx_len;
        if tokens.len() > max {
            return Err(ExecutorError::ContextOverflow { used: tokens.len(), max });
        }
        self.n_ctx_used = tokens.len();
        let _ = ctx;
        // TODO:
        // 1. Upload token embeddings to GPU activation buffer
        // 2. For each layer 0..n_gpu_layers: call dispatch kernel
        // 3. For each layer n_gpu_layers..n_layers: call CPU fallback
        // 4. Apply output norm, project to logits
        // 5. Download last-position logits to CPU
        Err(ExecutorError::NotInitialized)
    }

    /// Decode step: run a single-token forward pass (KV cache already populated).
    ///
    /// Returns `[vocab_size]` logits.
    pub fn decode_step(
        &mut self,
        token: u32,
        ctx:   &SpiteCtx,
    ) -> Result<Vec<f32>, ExecutorError> {
        let max = self.cfg.ctx_len;
        if self.n_ctx_used >= max {
            return Err(ExecutorError::ContextOverflow { used: self.n_ctx_used + 1, max });
        }
        self.n_ctx_used += 1;
        let _ = (token, ctx);
        // TODO: same as prefill but seq_len=1, reuse populated KV cache
        Err(ExecutorError::NotInitialized)
    }

    /// Reset the KV cache (start a new conversation without re-loading weights).
    pub fn reset(&mut self) {
        self.n_ctx_used = 0;
        // TODO: kvcache.reset()
    }

    pub fn n_ctx_used(&self) -> usize { self.n_ctx_used }
}

// ── Engine-level plugin registries ────────────────────────────────────────

/// All pluggable subsystem registries, held in one place.
///
/// Users fill these via `EngineBuilder`; the engine resolves the best match
/// for each request using the same priority-chain semantics as the kernel
/// dispatcher.
pub struct EngineRegistries {
    pub samplers:   Registry<dyn Sampler>,
    pub tokenizers: Registry<dyn Tokenize>,
    pub caches:     Registry<dyn Cache>,
}

impl EngineRegistries {
    fn with_defaults() -> Self {
        let mut r = Self {
            samplers:   Registry::new(),
            tokenizers: Registry::new(),
            caches:     Registry::new(),
        };
        // Ship a working sampler out of the box; tokenizer/cache are
        // model- and hardware-specific so they start empty.
        r.samplers.set_default(Box::new(DefaultSampler::new(0)));
        r
    }
}

// ── EngineBuilder ─────────────────────────────────────────────────────────

/// Builder for an [`Engine`] with custom subsystem overrides.
///
/// Start with `EngineBuilder::new()` (pre-populated with engine defaults),
/// chain `with_*` calls for your model/card/task, then call `build`.
///
/// ```rust,ignore
/// let engine = EngineBuilder::new()
///     .with_sampler(PluginKey::for_model("llama3"), Box::new(MyGreedySampler))
///     .with_cache(PluginKey::default(),             Box::new(PagedKvCache::new(vram)))
///     .build(ExecutorConfig::default());
/// ```
///
/// No fork required — register what you need and the engine resolves the
/// best match for every request.
pub struct EngineBuilder {
    registries: EngineRegistries,
}

impl Default for EngineBuilder {
    fn default() -> Self { Self::new() }
}

impl EngineBuilder {
    /// Create a builder pre-populated with built-in defaults.
    pub fn new() -> Self {
        Self { registries: EngineRegistries::with_defaults() }
    }

    /// Register a custom sampler for `key`.
    pub fn with_sampler(mut self, key: PluginKey, s: Box<dyn Sampler>) -> Self {
        self.registries.samplers.register(key, s);
        self
    }

    /// Register a custom tokenizer for `key`.
    pub fn with_tokenizer(mut self, key: PluginKey, t: Box<dyn Tokenize>) -> Self {
        self.registries.tokenizers.register(key, t);
        self
    }

    /// Register a custom KV cache for `key`.
    pub fn with_cache(mut self, key: PluginKey, c: Box<dyn Cache>) -> Self {
        self.registries.caches.register(key, c);
        self
    }

    /// Consume the builder and produce a ready-to-use [`Engine`].
    pub fn build(self, cfg: ExecutorConfig) -> Engine {
        Engine { executor: Executor::new(cfg), registries: self.registries }
    }
}

// ── Engine ────────────────────────────────────────────────────────────────

/// Top-level engine handle: forward-pass executor + all pluggable registries.
///
/// Resolve subsystems by calling the typed helpers with a [`PluginKey`]
/// describing the current request's model arch, GPU arch, and task.
/// The registry walks from most-specific to least-specific and returns the
/// first match — the same priority chain as the kernel dispatcher.
pub struct Engine {
    pub executor:   Executor,
    pub registries: EngineRegistries,
}

impl Engine {
    /// Resolve the sampler that best matches `key`.
    pub fn sampler(&self, key: &PluginKey) -> Option<&dyn Sampler> {
        self.registries.samplers.resolve(key)
    }

    /// Resolve the tokenizer that best matches `key`.
    pub fn tokenizer(&self, key: &PluginKey) -> Option<&dyn Tokenize> {
        self.registries.tokenizers.resolve(key)
    }

    /// Resolve the KV cache that best matches `key`.
    pub fn cache(&self, key: &PluginKey) -> Option<&dyn Cache> {
        self.registries.caches.resolve(key)
    }

}
