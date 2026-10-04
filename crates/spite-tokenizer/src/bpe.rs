//! Byte-Pair Encoding (BPE) tokenizer (GPT-2 style).
//!
//! BPE merges byte-pairs iteratively during training. At inference, we
//! reverse the process: apply pre-tokenization (regex split), then
//! greedily apply the stored merge table to each word piece.
//!
//! # Merge table
//!
//! Stored in GGUF as `tokenizer.ggml.merges` — a string array where
//! each entry is "a b" (two token strings separated by a space), ordered
//! by merge priority (lower index = higher priority).
//!
//! # Pre-tokenization
//!
//! GPT-2 uses a regex that splits on whitespace-prefix word boundaries.
//! LLaMA-3 uses tiktoken with a similar but slightly different pattern.
//! The pattern is embedded in GGUF as `tokenizer.ggml.pre` (optional).

use crate::TokenizerError;
use std::collections::HashMap;

/// A compiled BPE merge table: (token_a, token_b) → merged_id.
pub struct BpeMergeTable {
    /// Maps (a_id, b_id) → merged_id. Priority is implicit in insertion order.
    merges: Vec<(u32, u32, u32)>, // (a, b, result)
    merge_map: HashMap<(u32, u32), u32>,
}

impl BpeMergeTable {
    /// Build from a GGUF merges array.
    ///
    /// `merges_raw`: array of "token_a token_b" strings, in priority order.
    /// `vocab`:      maps token string → id (needed to resolve merge string pairs).
    pub fn from_merges(
        merges_raw: &[String],
        vocab_map: &HashMap<String, u32>,
    ) -> Result<Self, TokenizerError> {
        let mut merges = Vec::with_capacity(merges_raw.len());
        let mut merge_map = HashMap::with_capacity(merges_raw.len());

        for raw in merges_raw {
            let (a, b) = raw
                .split_once(' ')
                .ok_or_else(|| TokenizerError::Encode(format!("bad merge entry: {raw}")))?;
            let a_id = vocab_map
                .get(a)
                .copied()
                .ok_or_else(|| TokenizerError::Encode(format!("merge token not in vocab: {a}")))?;
            let b_id = vocab_map
                .get(b)
                .copied()
                .ok_or_else(|| TokenizerError::Encode(format!("merge token not in vocab: {b}")))?;
            let merged = vocab_map
                .get(&format!("{a}{b}"))
                .copied()
                .unwrap_or(u32::MAX);
            merges.push((a_id, b_id, merged));
            merge_map.insert((a_id, b_id), merged);
        }
        Ok(Self { merges, merge_map })
    }

    /// Encode a pre-tokenized word (as a list of byte token ids) using BPE.
    ///
    /// Iteratively applies the highest-priority merge until no more apply.
    pub fn encode_word(&self, bytes: &[u32]) -> Vec<u32> {
        let mut tokens: Vec<u32> = bytes.to_vec();
        loop {
            // Find the highest-priority (lowest merge index) applicable pair.
            let best = tokens
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| {
                    self.merge_map
                        .get(&(pair[0], pair[1]))
                        .map(|&merged| (i, merged, self.priority(pair[0], pair[1])))
                })
                .min_by_key(|&(_, _, priority)| priority);

            let Some((pos, merged, _)) = best else {
                break;
            };
            tokens[pos] = merged;
            tokens.remove(pos + 1);
        }
        tokens
    }

    fn priority(&self, a: u32, b: u32) -> usize {
        self.merges
            .iter()
            .position(|&(x, y, _)| x == a && y == b)
            .unwrap_or(usize::MAX)
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphabetic()
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

/// GPT-2 / Qwen2-style pre-tokenization.
///
/// This is a hand-rolled transcription of the canonical pattern
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
/// The critical property is that a **single leading space is attached to the
/// following word**, so `"The capital"` splits as `["The", " capital"]` and
/// tokenizes to `["The", " capital"]`. Splitting on whitespace instead (as a
/// naive `split(' ')` does) yields `["The", " ", "capital"]`, a token sequence
/// that never occurs in training data — which makes the model generate
/// space-less continuations.
pub fn pretokenize_gpt2(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let start = i;
        let next = text[i..].chars().next().expect("in-bounds char");

        // Contractions: 's 't 're 've 'm 'll 'd
        if next == '\'' && i + 1 < bytes.len() {
            let rest = &text[i + 1..];
            let contraction = ["s", "t", "re", "ve", "m", "ll", "d"]
                .into_iter()
                .find(|c| rest.starts_with(c));
            if let Some(c) = contraction {
                i += 1 + c.len();
                out.push(&text[start..i]);
                continue;
            }
        }

        // Optional single leading space, but only when a non-space follows:
        // this is what merges " capital" into one piece.
        let mut j = i;
        if next == ' '
            && let Some(after) = text[i + 1..].chars().next()
            && !is_space(after)
        {
            j += 1;
        }
        let head = text[j..].chars().next().unwrap_or(next);

        if is_word_char(head) {
            for c in text[j..].chars() {
                if !is_word_char(c) {
                    break;
                }
                j += c.len_utf8();
            }
        } else if head.is_ascii_digit() {
            for c in text[j..].chars() {
                if !c.is_ascii_digit() {
                    break;
                }
                j += c.len_utf8();
            }
        } else if !is_space(head) {
            for c in text[j..].chars() {
                if is_space(c) || is_word_char(c) || c.is_ascii_digit() {
                    break;
                }
                j += c.len_utf8();
            }
        } else {
            // A whitespace run; a trailing run before end-of-text is kept whole.
            for c in text[j..].chars() {
                if !is_space(c) {
                    break;
                }
                j += c.len_utf8();
            }
        }
        if j == i {
            // Defensive: never stall on an unclassified character.
            j = i + text[i..].chars().next().map_or(1, char::len_utf8);
        }
        i = j;
        out.push(&text[start..i]);
    }
    out
}

// ── Byte-level alphabet ─────────────────────────────────────────────────────
//
// GPT-2-style byte-level BPE never stores raw bytes in the vocabulary: each of
// the 256 byte values is represented by a printable unicode character, so that
// a token string is valid UTF-8. The mapping is fixed across every model that
// uses this scheme (Qwen2/Qwen3, GPT-2, CodeGen, ...), and it is *not* the
// identity: 0x20 (space) is `Ġ` (U+0120), and control bytes are shifted into
// U+0100.. in byte order.
//
// Vocabularies built this way contain no `<0xNN>` byte-fallback tokens and no
// literal space, so encoding has to go through this table.

/// `byte_level_char[b]` is the unicode char a byte-level vocabulary uses for
/// byte `b` (e.g. `byte_level_char[0x20] == 'Ġ'`, `[0x41] == 'A'`).
pub fn byte_level_char(b: u8) -> char {
    static TABLE: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        // Bytes GPT-2 keeps as-is: printable ASCII plus two Latin-1 ranges.
        let mut direct = [false; 256];
        for b in b'!'..=b'~' {
            direct[b as usize] = true;
        }
        for b in 0xA1u8..=0xAC {
            direct[b as usize] = true;
        }
        for b in 0xAEu8..=0xFF {
            direct[b as usize] = true;
        }
        let mut out = ['\0'; 256];
        let mut n = 0u32;
        for b in 0usize..256 {
            let cp = if direct[b] {
                b as u32
            } else {
                let cp = 256 + n;
                n += 1;
                cp
            };
            out[b] = char::from_u32(cp).expect("byte-level codepoint is valid");
        }
        out
    });
    table[b as usize]
}

/// Inverse of [`byte_level_char`]: the byte a byte-level vocabulary char
/// stands for, or `None` for chars outside the alphabet (e.g. `<|im_start|>`).
pub fn byte_level_byte(c: char) -> Option<u8> {
    (0..256u16)
        .map(|b| b as u8)
        .find(|&b| byte_level_char(b) == c)
}
