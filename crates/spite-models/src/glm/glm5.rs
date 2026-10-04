//! Zhipu AI GLM-5 Next — GGUF arch `glm5-next`.
//!
//! Next-generation GLM after the previous generation.
//! Architecture details TBD once released publicly.
//!
//! Expected to retain:
//! - Causal decoder (full, not prefix-LM)
//! - RoPE, GQA
//! - Likely expands vocabulary and context window from earlier GLM
//! - May incorporate lessons from GLM-DSA sparse attention
//!
//! This stub is registered so GGUF files load without error when weights ship.

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;

use crate::dense::{self, DenseWeights, KvStore};
use crate::{ModelArch, ModelConfig, ModelError};

pub struct Glm5 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    kv: RwLock<KvStore>,
}

impl Glm5 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv: RwLock::new(KvStore::default()),
        }
    }
}

impl ModelArch for Glm5 {
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
