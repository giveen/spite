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

/// Default sampler settings for [`Executor::generate`].
///
/// A mild repetition penalty keeps small models from looping under top-p, but
/// it must be **off** for greedy decoding (`temperature == 0`): greedy is
/// defined as the raw argmax of the logits, and a penalty scaled per occurrence
/// can promote a different token, so "greedy" output stops matching
/// llama.cpp's greedy decode.
pub fn default_sampler_config(temperature: f32) -> spite_sampling::SamplerConfig {
    spite_sampling::SamplerConfig {
        temperature,
        // 1.0 = disabled.
        repetition_penalty: if temperature > 0.0 { 1.1 } else { 1.0 },
        ..Default::default()
    }
}

/// Draft/accept counters from a speculative decode, for benchmarks.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecStats {
    /// MTP draft tokens proposed.
    pub drafted: usize,
    /// Draft tokens that the trunk accepted.
    pub accepted: usize,
}

/// Sampling parameters for [`Executor::generate_speculative`].
#[derive(Debug, Clone, Copy)]
pub struct SpecDecodeConfig {
    pub temperature: f32,
    pub seed: u64,
    /// MTP tokens proposed per step (>= 1).
    pub n_draft: usize,
}

impl Default for SpecDecodeConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            seed: 0,
            n_draft: 3,
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

    /// True when the loaded model exposes an in-weights NextN/MTP draft head.
    pub fn has_mtp(&self) -> bool {
        self.model.as_ref().is_some_and(|m| m.has_mtp())
    }

    /// Run one MTP draft step through the loaded model: consume the hidden
    /// state left by the latest forward, draft from `token` at RoPE position
    /// `pos`, and write the draft logits. See
    /// [`spite_models::hybrid::HybridDecoder::mtp_step`].
    pub fn mtp_step(
        &self,
        token: u32,
        pos: usize,
        logits_out: &mut [f32],
    ) -> Result<(), ExecutorError> {
        let Some(model) = &self.model else {
            return Err(ExecutorError::NotInitialized);
        };
        model
            .mtp_step(token, pos, logits_out)
            .map_err(|e| ExecutorError::Model(e.to_string()))
    }

    /// Speculative decode: prefill `prompt_ids`, then emit up to `max_tokens`
    /// with the MTP head drafting and the trunk verifying.
    ///
    /// Falls back to plain [`Self::generate`] when the model has no MTP head.
    pub fn generate_speculative(
        &mut self,
        tokenizer: &dyn Tokenize,
        prompt_ids: &[u32],
        max_tokens: usize,
        spec: SpecDecodeConfig,
    ) -> Result<Vec<(u32, String)>, ExecutorError> {
        Ok(self
            .generate_speculative_with_stats(tokenizer, prompt_ids, max_tokens, spec)?
            .0)
    }

    /// Like [`Self::generate_speculative`], also returning the draft/accept
    /// counters (used by `spite-bench` and the correctness tests).
    pub fn generate_speculative_with_stats(
        &mut self,
        tokenizer: &dyn Tokenize,
        prompt_ids: &[u32],
        max_tokens: usize,
        spec: SpecDecodeConfig,
    ) -> Result<(Vec<(u32, String)>, SpecStats), ExecutorError> {
        if !self.has_mtp() || spec.n_draft == 0 {
            let out = self.generate(
                tokenizer,
                prompt_ids,
                max_tokens,
                spec.temperature,
                spec.seed,
            )?;
            return Ok((out, SpecStats::default()));
        }
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
        let logits = self.prefill(prompt_ids, &ctx)?;
        self.generate_speculative_from_logits(tokenizer, prompt_ids, logits, max_tokens, spec)
    }

    /// Speculative decode from an already-prefilled state: `logits` is the
    /// trunk distribution for the next token. Public so a benchmark can time
    /// prefill and decode separately from the same entry point.
    ///
    /// The MTP head proposes up to `n_draft` tokens per step; the trunk then
    /// accepts the longest prefix that matches. The result is exact for greedy
    /// decoding and, through residual sampling, preserves the target
    /// distribution at `temperature > 0`.
    ///
    /// Returns the emitted `(id, piece)` pairs and the draft/accept counts.
    pub fn generate_speculative_from_logits(
        &mut self,
        tokenizer: &dyn Tokenize,
        prompt_ids: &[u32],
        mut logits: Vec<f32>,
        max_tokens: usize,
        spec: SpecDecodeConfig,
    ) -> Result<(Vec<(u32, String)>, SpecStats), ExecutorError> {
        let SpecDecodeConfig {
            temperature,
            seed,
            n_draft,
        } = spec;
        use spite_sampling::{apply_processors, sample, sample_probs, softmax, uniform};
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
        let cfg = default_sampler_config(temperature);
        let greedy = temperature <= 0.0;
        let mut rng = seed;
        let mut ids = prompt_ids.to_vec();
        let mut out = Vec::new();
        let mut stats = SpecStats::default();
        let mut produced = 0usize;

        while produced < max_tokens {
            // Verified token from the trunk distribution.
            let tok = sample(&mut logits, &ids, &cfg, &mut rng)
                .map_err(|e| ExecutorError::Sampling(e.to_string()))?;
            ids.push(tok);
            produced += 1;
            if tokenizer.is_eog(tok) {
                break;
            }
            out.push((tok, tokenizer.decode_one(tok).into_owned()));
            if produced >= max_tokens {
                break;
            }

            // The MTP head consumes the hidden state of the last trunk token,
            // which sits at `n_ctx_used - 1` (the verified token is not
            // decoded into the trunk until below).
            let base_pos = self.n_ctx_used.saturating_sub(1);
            let base_len = ids.len();
            let mut drafts: Vec<(u32, Vec<f32>)> = Vec::with_capacity(n_draft);
            for k in 0..n_draft {
                let in_tok = if k == 0 { tok } else { drafts[k - 1].0 };
                let mut q = vec![0f32; logits.len()];
                if self.mtp_step(in_tok, base_pos + k, &mut q).is_err() {
                    break;
                }
                // Sample the draft under the same context the draft model
                // would see, including its own earlier proposals.
                let d = sample(&mut q.clone(), &ids, &cfg, &mut rng)
                    .map_err(|e| ExecutorError::Sampling(e.to_string()))?;
                drafts.push((d, q));
                ids.push(d);
            }
            if drafts.is_empty() {
                logits = self.decode_step(tok, &ctx)?;
                continue;
            }
            stats.drafted += drafts.len();

            // Advance the trunk with the verified token; `logits` now predicts
            // the position the first draft targets.
            logits = self.decode_step(tok, &ctx)?;
            for (k, (d, mut q)) in drafts.into_iter().enumerate() {
                // Verify d_k against the context of its own position:
                // [prompt, tok, accepted drafts].
                ids.truncate(base_len + k);
                if greedy {
                    let mut p = logits.clone();
                    let expected = sample(&mut p, &ids, &cfg, &mut rng)
                        .map_err(|e| ExecutorError::Sampling(e.to_string()))?;
                    if expected != d {
                        break;
                    }
                } else {
                    let mut p = logits.clone();
                    apply_processors(&mut p, &ids, &cfg);
                    apply_processors(&mut q, &ids, &cfg);
                    let pp = softmax(&p);
                    let qp = softmax(&q);
                    let qx = qp.get(d as usize).copied().unwrap_or(0.0);
                    let px = pp.get(d as usize).copied().unwrap_or(0.0);
                    if qx <= 0.0 || uniform(&mut rng) >= (px / qx).min(1.0) {
                        // Rejected: draw the correction from (p - q)_+.
                        let residual: Vec<f32> =
                            pp.iter().zip(&qp).map(|(a, b)| (a - b).max(0.0)).collect();
                        let y = sample_probs(&residual, &mut rng)
                            .map_err(|e| ExecutorError::Sampling(e.to_string()))?;
                        ids.push(y);
                        produced += 1;
                        if tokenizer.is_eog(y) {
                            return Ok((out, stats));
                        }
                        out.push((y, tokenizer.decode_one(y).into_owned()));
                        logits = self.decode_step(y, &ctx)?;
                        break;
                    }
                }
                // Accepted.
                stats.accepted += 1;
                ids.push(d);
                produced += 1;
                if tokenizer.is_eog(d) {
                    return Ok((out, stats));
                }
                out.push((d, tokenizer.decode_one(d).into_owned()));
                logits = self.decode_step(d, &ctx)?;
                if produced >= max_tokens {
                    break;
                }
            }
        }
        Ok((out, stats))
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
        use spite_sampling::sample;

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
        let sampler_cfg = default_sampler_config(temperature);
        let mut rng = seed;
        let mut out = Vec::new();

        let mut logits = self.prefill(&ids, &ctx)?;
        for _ in 0..max_tokens {
            let tok = sample(&mut logits, &ids, &sampler_cfg, &mut rng)
                .map_err(|e| ExecutorError::Sampling(e.to_string()))?;
            ids.push(tok);
            if tokenizer.is_eog(tok) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use spite_sampling::sample;

    #[test]
    fn greedy_disables_the_repetition_penalty() {
        assert_eq!(default_sampler_config(0.0).repetition_penalty, 1.0);
        assert_eq!(default_sampler_config(1.0).repetition_penalty, 1.1);
    }

    /// Regression: a per-occurrence penalty used to run even at temperature 0,
    /// so a repeated token (2.0) could lose to a lower one (1.95) — i.e. greedy
    /// was not the argmax.
    #[test]
    fn greedy_is_the_raw_argmax() {
        let mut rng = 0x1234_5678;
        let logits = [0.0f32, 2.0, 1.95];

        let mut a = logits;
        assert_eq!(
            sample(&mut a, &[1], &default_sampler_config(0.0), &mut rng).unwrap(),
            1
        );

        let mut penalized = default_sampler_config(0.0);
        penalized.repetition_penalty = 1.1;
        let mut b = logits;
        assert_eq!(sample(&mut b, &[1], &penalized, &mut rng).unwrap(), 2);
    }
}
