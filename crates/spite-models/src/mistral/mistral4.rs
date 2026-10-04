//! Mistral 4 / Magistral — GGUF arch `mistral4`.
//!
//! Dense GQA + SwiGLU, no sliding window: uses the shared dense forward.

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;

use crate::dense::{self, DenseWeights, KvStore};
use crate::{ModelArch, ModelConfig, ModelError};

pub struct Mistral4 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    kv: RwLock<KvStore>,
}

impl Mistral4 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv: RwLock::new(KvStore::default()),
        }
    }
}

impl ModelArch for Mistral4 {
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
