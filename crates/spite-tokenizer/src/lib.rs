//! Tokenizer — reads the vocabulary embedded in GGUF metadata.
//!
//! GGUF files contain a complete tokenizer in their key-value store:
//!   tokenizer.ggml.model      — "llama", "gpt2", "bert", ...
//!   tokenizer.ggml.tokens     — array of token strings
//!   tokenizer.ggml.scores     — array of token scores (BPE merge priority)
//!   tokenizer.ggml.token_type — array of token type flags
//!
//! No separate tokenizer download or config file needed.

pub mod bpe;
pub mod sentencepiece;

use thiserror::Error;

// ── Tokenize trait ────────────────────────────────────────────────────────

/// The pluggable tokenizer interface.
///
/// Implement this to replace the tokenizer for a specific model or task
/// without touching any other part of the engine.
///
/// Register implementations in a `Registry<dyn Tokenize>` so overrides
/// apply only where they're needed.
pub trait Tokenize: Send + Sync {
    /// Encode `text` to token ids. Prepends BOS if `add_bos` is true.
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokenizerError>;

    /// Decode token ids to UTF-8 text.
    fn decode(&self, ids: &[u32], skip_special: bool) -> String;

    /// Decode a single token — used for streaming output.
    fn decode_one(&self, id: u32) -> &str;

    fn vocab_size(&self) -> usize;
    fn bos_id(&self) -> u32;
    fn eos_id(&self) -> u32;
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("GGUF has no tokenizer metadata")]
    NoTokenizer,
    #[error("unsupported tokenizer model: {0}")]
    UnsupportedModel(String),
    #[error("vocab size mismatch: tokens={tokens} scores={scores}")]
    VocabMismatch { tokens: usize, scores: usize },
    #[error("encode failed: {0}")]
    Encode(String),
}

// ── Token type flags (matches GGUF spec) ──────────────────────────────────

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal      = 1,
    Unknown     = 2,
    Control     = 3,
    UserDefined = 4,
    Unused      = 5,
    Byte        = 6,
}

// ── Vocabulary ────────────────────────────────────────────────────────────

pub struct Vocab {
    /// Token strings, indexed by token id.
    pub tokens: Vec<String>,
    /// BPE merge scores / SentencePiece log-probabilities.
    pub scores: Vec<f32>,
    /// Token type flags.
    pub token_types: Vec<TokenType>,
    /// Special token ids.
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: Option<u32>,
    pub unk_id: Option<u32>,
}

impl Vocab {
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_to_str(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    pub fn is_special(&self, id: u32) -> bool {
        matches!(
            self.token_types.get(id as usize),
            Some(TokenType::Control | TokenType::Unknown)
        )
    }
}

// ── Tokenizer ─────────────────────────────────────────────────────────────

pub enum TokenizerKind {
    /// BPE tokenizer (GPT-2 style). Most GGUF models.
    Bpe,
    /// SentencePiece Unigram (LLaMA, Mistral).
    SentencePiece,
    /// WordPiece (BERT). Rare in GGUF inference models.
    WordPiece,
}

pub struct Tokenizer {
    pub vocab: Vocab,
    pub kind:  TokenizerKind,
    // TODO: BPE merge table or SP model trie
}

impl Tokenize for Tokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokenizerError> {
        self.encode(text, add_bos)
    }
    fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.decode(ids, skip_special)
    }
    fn decode_one(&self, id: u32) -> &str {
        self.decode_one(id)
    }
    fn vocab_size(&self) -> usize { self.vocab.vocab_size() }
    fn bos_id(&self) -> u32      { self.vocab.bos_id }
    fn eos_id(&self) -> u32      { self.vocab.eos_id }
}

impl Tokenizer {
    /// Build a Tokenizer from a loaded GGUF model's metadata.
    pub fn from_gguf(_model: &spite_loader::GgufModel) -> Result<Self, TokenizerError> {
        // TODO:
        // 1. Read tokenizer.ggml.model → determine kind
        // 2. Read tokenizer.ggml.tokens / scores / token_type arrays
        // 3. Read tokenizer.ggml.bos_token_id / eos_token_id
        // 4. Build merge table (BPE) or trie (SP)
        Err(TokenizerError::NoTokenizer)
    }

    /// Encode text to token ids. Prepends BOS if `add_bos` is true.
    pub fn encode(&self, _text: &str, _add_bos: bool) -> Result<Vec<u32>, TokenizerError> {
        // TODO: BPE or SP encode
        Err(TokenizerError::Encode("not yet implemented".into()))
    }

    /// Decode token ids to a UTF-8 string.
    /// Handles byte-fallback tokens (U+2581 → space, <0xNN> → raw byte).
    pub fn decode(&self, _ids: &[u32], _skip_special: bool) -> String {
        // TODO: concatenate token strings, apply byte fallback
        String::new()
    }

    /// Decode a single token id — used for streaming output.
    pub fn decode_one(&self, id: u32) -> &str {
        self.vocab.token_to_str(id).unwrap_or("")
    }
}
