//! Perplexity and KL divergence evaluation for spite kernels.
//!
//! Two tests:
//!
//! ## Perplexity (PPL)
//! Standard language model quality metric.
//!   PPL = exp( -1/N * Σ log P(token_i | prefix) )
//! Lower is better. Run against a text corpus (WikiText-2 is standard).
//! A correct kernel should produce PPL within ~0.1% of the reference.
//!
//! ## KL Divergence (KLD)
//! Compares the full token probability distribution produced by two
//! kernels — your optimized kernel vs the reference (generic fallback).
//!   KLD(P‖Q) = Σ P(x) * log(P(x) / Q(x))
//!
//! This is the **primary quality gate for contributed kernels**:
//!   mean KLD < 0.001  →  pass  (effectively identical distributions)
//!   mean KLD > 0.01   →  fail  (something is wrong with your kernel)
//!
//! PPL alone can look fine while distributions diverge on rare tokens.
//! KLD catches those cases.

pub mod dataset;
pub mod kld;
pub mod ppl;
pub mod report;

pub use kld::{KldConfig, KldResult};
pub use ppl::{PplConfig, PplResult};
pub use report::Report;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("model load: {0}")]
    Load(#[from] spite_loader::LoadError),
    #[error("dispatch: {0}")]
    Dispatch(#[from] spite_dispatch::DispatchError),
    #[error("dataset: {0}")]
    Dataset(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
