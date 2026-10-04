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

use std::borrow::Cow;
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
    ///
    /// Returns `Borrowed` for normal tokens (zero-copy from vocab table).
    /// Returns `Owned` for byte-fallback tokens that need reconstruction.
    fn decode_one(&self, id: u32) -> Cow<'_, str>;

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
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
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
    pub kind: TokenizerKind,
    merges: Option<bpe::BpeMergeTable>,
    unigram: Option<sentencepiece::UnigramModel>,
    /// token string → id, for BPE byte lookup and merge resolution.
    vocab_map: std::collections::HashMap<String, u32>,
}

impl Tokenize for Tokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokenizerError> {
        self.encode(text, add_bos)
    }
    fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.decode(ids, skip_special)
    }
    fn decode_one(&self, id: u32) -> Cow<'_, str> {
        self.decode_one(id)
    }
    fn vocab_size(&self) -> usize {
        self.vocab.vocab_size()
    }
    fn bos_id(&self) -> u32 {
        self.vocab.bos_id
    }
    fn eos_id(&self) -> u32 {
        self.vocab.eos_id
    }
}

impl Tokenizer {
    /// Build a Tokenizer from a loaded GGUF model's metadata.
    pub fn from_gguf(model: &spite_loader::GgufModel) -> Result<Self, TokenizerError> {
        let tokens = model
            .get_array("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::NoTokenizer)?;
        let mut vocab_tokens = Vec::with_capacity(tokens.len());
        for t in tokens {
            match t {
                spite_loader::MetaValue::Str(s) => vocab_tokens.push(s.clone()),
                _ => return Err(TokenizerError::NoTokenizer),
            }
        }
        let scores = model
            .get_array("tokenizer.ggml.scores")
            .map(|a| {
                a.iter()
                    .map(|v| match v {
                        spite_loader::MetaValue::F32(f) => *f,
                        _ => 0.0,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![0.0; vocab_tokens.len()]);
        if scores.len() != vocab_tokens.len() {
            return Err(TokenizerError::VocabMismatch {
                tokens: vocab_tokens.len(),
                scores: scores.len(),
            });
        }
        let token_types = model
            .get_array("tokenizer.ggml.token_type")
            .map(|a| {
                a.iter()
                    .map(|v| match v {
                        spite_loader::MetaValue::I32(i) => *i as u32,
                        spite_loader::MetaValue::U32(u) => *u,
                        _ => TokenType::Normal as u32,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![TokenType::Normal as u32; vocab_tokens.len()]);
        let token_types = token_types
            .into_iter()
            .map(|t| match t {
                1 => TokenType::Normal,
                2 => TokenType::Unknown,
                3 => TokenType::Control,
                4 => TokenType::UserDefined,
                5 => TokenType::Unused,
                6 => TokenType::Byte,
                _ => TokenType::Normal,
            })
            .collect();

        let kind_str = model.get_str("tokenizer.ggml.model").unwrap_or("gpt2");
        let kind = match kind_str {
            "llama" | "unigram" => TokenizerKind::SentencePiece,
            "bert" | "wordpiece" => TokenizerKind::WordPiece,
            _ => TokenizerKind::Bpe,
        };

        let vocab_map: std::collections::HashMap<String, u32> = vocab_tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let merges = match kind {
            TokenizerKind::Bpe => model
                .get_array("tokenizer.ggml.merges")
                .map(|a| {
                    a.iter()
                        .filter_map(|v| match v {
                            spite_loader::MetaValue::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .map(|raw| bpe::BpeMergeTable::from_merges(&raw, &vocab_map))
                .transpose()?,
            _ => None,
        };
        let unigram = match kind {
            TokenizerKind::SentencePiece => {
                Some(sentencepiece::UnigramModel::new(&vocab_tokens, &scores)?)
            }
            _ => None,
        };

        let u32_id = |key: &str, default: u32| model.get_u32(key).unwrap_or(default);
        Ok(Self {
            vocab: Vocab {
                bos_id: u32_id("tokenizer.ggml.bos_token_id", 0),
                eos_id: u32_id("tokenizer.ggml.eos_token_id", 1),
                unk_id: model.get_u32("tokenizer.ggml.unknown_token_id"),
                pad_id: model.get_u32("tokenizer.ggml.padding_token_id"),
                tokens: vocab_tokens,
                scores,
                token_types,
            },
            kind,
            merges,
            unigram,
            vocab_map,
        })
    }

    /// Encode text to token ids. Prepends BOS if `add_bos` is true.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.vocab.bos_id);
        }
        match self.kind {
            TokenizerKind::Bpe => {
                let table = self
                    .merges
                    .as_ref()
                    .ok_or_else(|| TokenizerError::Encode("no BPE merge table".into()))?;
                for word in bpe::pretokenize_gpt2(text) {
                    // Map each byte to its single-byte token id.
                    let mut bytes = Vec::with_capacity(word.len());
                    for &b in word.as_bytes() {
                        let s = (b as char).to_string();
                        match self.vocab_map.get(&s) {
                            Some(&id) => bytes.push(id),
                            None => {
                                if let Some(unk) = self.vocab.unk_id {
                                    bytes.push(unk);
                                } else {
                                    return Err(TokenizerError::Encode(format!(
                                        "no byte token for 0x{b:02x}"
                                    )));
                                }
                            }
                        }
                    }
                    ids.extend(table.encode_word(&bytes));
                }
            }
            TokenizerKind::SentencePiece => {
                let model = self
                    .unigram
                    .as_ref()
                    .ok_or_else(|| TokenizerError::Encode("no unigram model".into()))?;
                // SentencePiece treats the input as-is (no pre-tokenization);
                // a leading ▁ is handled by the model itself.
                ids.extend(model.encode(text));
            }
            TokenizerKind::WordPiece => {
                return Err(TokenizerError::Encode(
                    "wordpiece encode not yet ported".into(),
                ));
            }
        }
        Ok(ids)
    }

    /// Decode token ids to a UTF-8 string.
    /// Handles byte-fallback tokens (▁ → space, <0xNN> → raw byte).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special && self.vocab.is_special(id) {
                continue;
            }
            let Some(tok) = self.vocab.token_to_str(id) else {
                continue;
            };
            if let Some(hex) = tok.strip_prefix("<0x").and_then(|s| s.strip_suffix('>'))
                && let Ok(b) = u8::from_str_radix(hex, 16)
            {
                bytes.push(b);
                continue;
            }
            bytes.extend_from_slice(tok.replace('▁', " ").as_bytes());
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single token id — used for streaming output.
    pub fn decode_one(&self, id: u32) -> Cow<'_, str> {
        if let Some(tok) = self.vocab.token_to_str(id) {
            if let Some(hex) = tok.strip_prefix("<0x").and_then(|s| s.strip_suffix('>'))
                && let Ok(b) = u8::from_str_radix(hex, 16)
            {
                return Cow::Owned((b as char).to_string());
            }
            return Cow::Borrowed(tok);
        }
        Cow::Borrowed("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spite_loader::GgufModel;

    fn test_tokenizer() -> Tokenizer {
        let tmp = spite_testkit::FakeGguf::default()
            .write_to_tempfile()
            .unwrap();
        let model = GgufModel::open(tmp.path()).unwrap();
        Tokenizer::from_gguf(&model).unwrap()
    }

    #[test]
    fn bpe_roundtrip_with_merge() {
        let tok = test_tokenizer();
        // "AB" merges to one id via the "A B" merge rule.
        let ids = tok.encode("AB", false).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(tok.decode(&ids, true), "AB");
        // BOS prepends when asked.
        let with_bos = tok.encode("AB", true).unwrap();
        assert_eq!(with_bos[0], tok.vocab.bos_id);
        assert_eq!(with_bos.len(), 2);
    }
}
