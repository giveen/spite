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

use spite_abi::SpiteCtx;
use spite_plugin::PluginKey;
use spite_sampling::Sampler;
use spite_kvcache::Cache;
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

/// How many transformer layers run on the GPU; the rest use the CPU fallback.
#[derive(Debug, Clone, Copy)]
pub enum LayerSplit {
    /// All layers on GPU (default when VRAM is sufficient).
    All,
    /// All layers on CPU (no GPU required).
    None,
    /// First `n` layers on GPU, remaining on CPU.
    Gpu(usize),
}

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub layer_split:  LayerSplit,
    pub n_threads:    usize,
    pub ctx_len:      usize,
    pub batch_size:   usize,
    /// Lock model weights in physical memory to prevent swapping.
    pub mlock:        bool,
    /// Advise the OS that mmap'd weights will be accessed sequentially.
    pub madvise_seq:  bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            layer_split: LayerSplit::All,
            n_threads:   4,
            ctx_len:     4096,
            batch_size:  512,
            mlock:       false,
            madvise_seq: true,
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
