//! Dataset loading for PPL / KLD evaluation.
//!
//! Supports two input modes:
//!   1. A plain text file — tokenize it, use it directly.
//!   2. A pre-tokenized file — one token ID per line (for reproducibility).
//!
//! The standard corpus for LLM perplexity is WikiText-2 (test split).
//! Download: https://huggingface.co/datasets/wikitext
//! Use the raw text variant and pass it as --corpus.

use crate::EvalError;
use std::path::Path;

pub struct Dataset {
    pub tokens: Vec<u32>,
    pub source: String,
}

impl Dataset {
    /// Load from a pre-tokenized file (one u32 per line).
    pub fn from_token_file(path: &Path) -> Result<Self, EvalError> {
        let text = std::fs::read_to_string(path)?;
        let tokens: Vec<u32> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| EvalError::Dataset(format!("bad token id: {e}")))?;

        if tokens.is_empty() {
            return Err(EvalError::Dataset("token file is empty".into()));
        }

        Ok(Self {
            tokens,
            source: path.display().to_string(),
        })
    }

    /// Load from a raw text file.
    /// Tokenization is deferred — caller provides the tokenize closure.
    pub fn from_text_file<F>(path: &Path, mut tokenize: F) -> Result<Self, EvalError>
    where
        F: FnMut(&str) -> Vec<u32>,
    {
        let text = std::fs::read_to_string(path)?;
        let tokens = tokenize(&text);

        if tokens.is_empty() {
            return Err(EvalError::Dataset("text produced zero tokens".into()));
        }

        Ok(Self {
            tokens,
            source: path.display().to_string(),
        })
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }
}
