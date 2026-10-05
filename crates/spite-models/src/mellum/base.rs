//! JetBrains Mellum — GGUF arch `mellum`.
//!
//! JetBrains' Fill-in-the-Middle (FIM) code completion model (2025).
//! Optimised for IDE inline completion (prefix + suffix → middle).
//!
//! # FIM format
//!
//! Uses sentinel tokens for FIM:
//! - `<|fim_prefix|>` — precedes the code before the cursor
//! - `<|fim_suffix|>` — precedes the code after the cursor
//! - `<|fim_middle|>` — model generates the completion here
//!
//! At inference: assemble prompt as:
//!   `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>`
//! then sample until `<|fim_pad|>` or `<|endoftext|>`.
//!
//! # Architecture
//!
//! - Dense decoder (Llama-style): GQA, RoPE, RMSNorm, SwiGLU
//! - Sizes: 1B, 9B (2025)
//! - Trained on multi-repo code (JetBrains internal crawl + permissive OSS)

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;

use crate::dense::{self, DenseWeights, KvStore};
use crate::{ModelArch, ModelConfig, ModelError};
use spite_kvcache::KvQuantConfig;

pub struct Mellum {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    kv: RwLock<KvStore>,
}

impl Mellum {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv: RwLock::new(KvStore::default()),
        }
    }
}

impl ModelArch for Mellum {
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
        dense::forward(
            &self.config,
            w,
            &self.kv,
            tokens,
            ctx.pos as usize,
            logits_out,
        )
    }
}
