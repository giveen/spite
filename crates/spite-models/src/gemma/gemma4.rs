//! Google Gemma 4 — GGUF arch `gemma4`.
//!
//! Variant: Gemma 4 (2025). Separate instruct variant uses arch `gemma4-assistant`.
//!
//! Inherits the Gemma architecture lineage:
//! - Local + global alternating attention (5:1 ratio)
//! - GeGLU FFN activation
//! - Pre- and post-norm around both sublayers
//! - Logit soft-capping (`final_logit_softcapping`)
//! - SigLIP vision encoder integration (multimodal)
//!
//! Notable features:
//! - Extended context (>32K)
//! - Updated KV head ratio (GQA n_kv_heads)
//! - Improved RoPE scaling
//!
//! Update this stub once Gemma 4 architecture details are public.

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;

use crate::dense::{self, Activation, DenseOptions, DenseWeights, KvStore};
use crate::{ModelArch, ModelConfig, ModelError};
use spite_kvcache::KvQuantConfig;

pub struct Gemma4 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    kv: RwLock<KvStore>,
}

impl Gemma4 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv: RwLock::new(KvStore::default()),
        }
    }
}

impl ModelArch for Gemma4 {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load_weights(&mut self, model: &GgufModel) -> Result<(), ModelError> {
        self.weights = Some(DenseWeights::load(model)?);
        Ok(())
    }

    fn reset_cache(&self) {
        if let Ok(mut kv) = self.kv.write() {
            kv.reset();
        }
    }

    fn set_kv_quant(&self, cfg: KvQuantConfig) {
        if let Ok(mut kv) = self.kv.write() {
            kv.set_quant(&cfg);
        }
    }

    fn forward(
        &self,
        tokens: &[u32],
        logits_out: &mut [f32],
        ctx: &SpiteCtx,
    ) -> Result<(), ModelError> {
        let Some(w) = &self.weights else {
            return Err(ModelError::Forward("load_weights not called".into()));
        };
        // Gemma alternates local/global layers; without per-layer types in
        // GGUF metadata every layer uses the sliding window when set.
        // Logit soft-capping is a sampling concern, applied downstream.
        let opts = DenseOptions {
            activation: Activation::GeGlu,
            sliding_window: self.config.sliding_window,
            rope_stride: 1,
            apply_qk_norm: false,
        };
        dense::forward_with(
            &self.config,
            w,
            &self.kv,
            tokens,
            ctx.pos as usize,
            logits_out,
            &opts,
        )
    }
}
