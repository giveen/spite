//! MiniMax M3 — GGUF arch `minimax-m3`.
//!
//! Third-generation MiniMax model (2025).
//! Stub registered for forward compatibility.

use std::sync::RwLock;

use spite_abi::SpiteCtx;
use spite_loader::GgufModel;

use crate::dense::{self, DenseWeights, KvStore};
use crate::{ModelArch, ModelConfig, ModelError};
use spite_kvcache::KvQuantConfig;

pub struct MinimaxM3 {
    config: ModelConfig,
    weights: Option<DenseWeights>,
    // ponytail: RwLock, uncontended single-threaded use; sharded locks if parallel decode matters.
    kv: RwLock<KvStore>,
}

impl MinimaxM3 {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            weights: None,
            kv: RwLock::new(KvStore::default()),
        }
    }
}

impl ModelArch for MinimaxM3 {
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
