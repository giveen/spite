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

use std::ffi::c_int;

use spite_abi::{ShardStrategy, SpiteCtx, SpiteTensor};
use spite_kvcache::{Cache, KvQuantConfig};
use spite_offload::OffloadConfig;
use spite_plugin::{PluginKey, Registry};
use spite_sampling::{DefaultSampler, Sampler};
use spite_tokenizer::Tokenize;
use thiserror::Error;

// ── Contiguity guard ──────────────────────────────────────────────────────

/// Panic with a diagnostic message if `t` is not contiguous.
///
/// Call this at each executor dispatch site before invoking an external kernel
/// `.so`. All kernels compiled against ABI v4 assume contiguous inputs; a
/// non-contiguous tensor here means a KV-cache view or activation slice was
/// passed without materialising it first.
#[track_caller]
pub fn require_contiguous(t: &SpiteTensor, name: &str) {
    assert!(
        t.is_contiguous(),
        "tensor '{name}' passed to kernel dispatch is not contiguous \
         (ne={:?} nb={:?} kind={:?}). \
         Materialise strided views before dispatch.",
        t.ne,
        t.nb,
        t.kind,
    );
}

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
    #[error("sampling failed: {0}")]
    Sampling(String),
}

// Re-export so callers only need one import.
pub use spite_abi::ShardStrategy as LayerSplit;

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub layer_split: ShardStrategy,
    pub n_threads: usize,
    pub ctx_len: usize,
    pub batch_size: usize,
    /// Lock model weights in physical memory to prevent swapping.
    pub mlock: bool,
    /// Advise the OS that mmap'd weights will be accessed sequentially.
    pub madvise_seq: bool,
    /// Weight offload policy (VRAM → RAM → disk).
    /// `None` means keep everything in VRAM; overflow panics if VRAM is insufficient.
    pub offload: Option<OffloadConfig>,
    /// KV cache quantization.  Default: both K and V at f16.
    pub kv_quant: KvQuantConfig,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            layer_split: ShardStrategy::None,
            n_threads: 4,
            ctx_len: 4096,
            batch_size: 512,
            mlock: false,
            madvise_seq: true,
            offload: None,
            kv_quant: KvQuantConfig::default(),
        }
    }
}

/// Owns the model, dispatch table, KV cache, and GPU activation buffers.
///
/// One `Executor` per loaded model; share it across requests using
/// `spite-scheduler` for concurrency.
pub struct Executor {
    pub cfg: ExecutorConfig,
    model: Option<Box<dyn spite_models::ModelArch>>,
    n_ctx_used: usize,
}

impl Executor {
    pub fn new(cfg: ExecutorConfig) -> Self {
        Self {
            cfg,
            model: None,
            n_ctx_used: 0,
        }
    }

    /// Load model weights. Resets KV state; call once before `prefill`.
    pub fn load_model(&mut self, model: Box<dyn spite_models::ModelArch>) {
        // Hand the KV quantization policy to the model before any token is
        // cached; `set_kv_quant` also clears state, so the tier applies from
        // position zero.
        model.set_kv_quant(self.cfg.kv_quant.clone());
        model.reset_cache();
        self.model = Some(model);
        self.n_ctx_used = 0;
    }

    /// Prefill: run a batched forward pass over `tokens`, populating the KV cache.
    ///
    /// Returns `[vocab_size]` logits for the last token position.
    pub fn prefill(&mut self, tokens: &[u32], ctx: &SpiteCtx) -> Result<Vec<f32>, ExecutorError> {
        let max = self.cfg.ctx_len;
        if tokens.len() > max {
            return Err(ExecutorError::ContextOverflow {
                used: tokens.len(),
                max,
            });
        }
        let Some(model) = &self.model else {
            return Err(ExecutorError::NotInitialized);
        };
        model.reset_cache();
        let vocab = model.config().vocab_size;
        let mut all = vec![0f32; tokens.len() * vocab];
        // ponytail: caller ctx only overrides threading/batch; pos runs 0..len.
        let mut fwd = *ctx;
        fwd.pos = 0;
        model
            .forward(tokens, &mut all, &fwd)
            .map_err(|e| ExecutorError::Model(e.to_string()))?;
        self.n_ctx_used = tokens.len();
        Ok(all[(tokens.len() - 1) * vocab..].to_vec())
    }

    /// Decode step: run a single-token forward pass (KV cache already populated).
    ///
    /// Returns `[vocab_size]` logits.
    pub fn decode_step(&mut self, token: u32, ctx: &SpiteCtx) -> Result<Vec<f32>, ExecutorError> {
        let max = self.cfg.ctx_len;
        if self.n_ctx_used >= max {
            return Err(ExecutorError::ContextOverflow {
                used: self.n_ctx_used + 1,
                max,
            });
        }
        let Some(model) = &self.model else {
            return Err(ExecutorError::NotInitialized);
        };
        let vocab = model.config().vocab_size;
        let mut out = vec![0f32; vocab];
        let mut step = *ctx;
        step.pos = self.n_ctx_used as c_int;
        model
            .forward(&[token], &mut out, &step)
            .map_err(|e| ExecutorError::Model(e.to_string()))?;
        self.n_ctx_used += 1;
        Ok(out)
    }

    /// Reset the KV cache (start a new conversation without re-loading weights).
    pub fn reset(&mut self) {
        if let Some(model) = &self.model {
            model.reset_cache();
        }
        self.n_ctx_used = 0;
    }

    pub fn n_ctx_used(&self) -> usize {
        self.n_ctx_used
    }

    /// Roll back the context position by `n` steps (e.g. on rejected speculative draft tokens).
    pub fn rollback(&mut self, n: usize) {
        self.n_ctx_used = self.n_ctx_used.saturating_sub(n);
    }

    /// Full generate loop: encode is done by the caller; this runs
    /// prefill → sample/decode until `max_tokens` or EOS.
    ///
    /// Returns per-token `(id, decoded piece)` pairs; text is their concat.
    pub fn generate(
        &mut self,
        tokenizer: &dyn Tokenize,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Vec<(u32, String)>, ExecutorError> {
        use spite_sampling::{SamplerConfig, sample};

        let eos = tokenizer.eos_id();
        let mut ids = prompt_ids.to_vec();
        let ctx = SpiteCtx {
            n_ctx: self.cfg.ctx_len as c_int,
            n_batch: self.cfg.batch_size as c_int,
            n_threads: self.cfg.n_threads as c_int,
            pos: 0,
            n_heads: 0,
            n_kv_heads: 0,
            gpu_stream: std::ptr::null_mut(),
            scratchpad: std::ptr::null_mut(),
            scratchpad_bytes: 0,
        };
        let sampler_cfg = SamplerConfig {
            temperature,
            // Base 8B models fall into verbatim repetition loops under plain
            // top-p sampling; a mild penalty keeps generations moving without
            // distorting the distribution. Callers wanting the raw library
            // default can still override it.
            repetition_penalty: 1.1,
            ..Default::default()
        };
        let mut rng = seed;
        let mut out = Vec::new();

        let mut logits = self.prefill(&ids, &ctx)?;
        for _ in 0..max_tokens {
            let tok = sample(&mut logits, &ids, &sampler_cfg, &mut rng)
                .map_err(|e| ExecutorError::Sampling(e.to_string()))?;
            ids.push(tok);
            if tok == eos {
                break;
            }
            out.push((tok, tokenizer.decode_one(tok).into_owned()));
            logits = self.decode_step(tok, &ctx)?;
        }
        Ok(out)
    }
}

// ── Engine-level plugin registries ────────────────────────────────────────

/// All pluggable subsystem registries, held in one place.
///
/// Users fill these via `EngineBuilder`; the engine resolves the best match
/// for each request using the same priority-chain semantics as the kernel
/// dispatcher.
pub struct EngineRegistries {
    pub samplers: Registry<dyn Sampler>,
    pub tokenizers: Registry<dyn Tokenize>,
    pub caches: Registry<dyn Cache>,
}

impl EngineRegistries {
    fn with_defaults() -> Self {
        let mut r = Self {
            samplers: Registry::new(),
            tokenizers: Registry::new(),
            caches: Registry::new(),
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
///     .with_sampler(PluginKey::for_model("llama4"), Box::new(MyGreedySampler))
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
    fn default() -> Self {
        Self::new()
    }
}

impl EngineBuilder {
    /// Create a builder pre-populated with built-in defaults.
    pub fn new() -> Self {
        Self {
            registries: EngineRegistries::with_defaults(),
        }
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
        Engine {
            executor: Executor::new(cfg),
            registries: self.registries,
        }
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
    pub executor: Executor,
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
