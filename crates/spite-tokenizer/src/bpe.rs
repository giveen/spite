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

use std::collections::HashMap;
use crate::TokenizerError;

/// A compiled BPE merge table: (token_a, token_b) → merged_id.
pub struct BpeMergeTable {
    /// Maps (a_id, b_id) → merged_id. Priority is implicit in insertion order.
    merges:    Vec<(u32, u32, u32)>,  // (a, b, result)
    merge_map: HashMap<(u32, u32), u32>,
}

impl BpeMergeTable {
    /// Build from a GGUF merges array.
    ///
    /// `merges_raw`: array of "token_a token_b" strings, in priority order.
    /// `vocab`:      maps token string → id (needed to resolve merge string pairs).
    pub fn from_merges(
        merges_raw: &[String],
        vocab_map:  &HashMap<String, u32>,
    ) -> Result<Self, TokenizerError> {
        let mut merges    = Vec::with_capacity(merges_raw.len());
        let mut merge_map = HashMap::with_capacity(merges_raw.len());

        for raw in merges_raw {
            let (a, b) = raw.split_once(' ')
                .ok_or_else(|| TokenizerError::Encode(format!("bad merge entry: {raw}")))?;
            let a_id = vocab_map.get(a).copied()
                .ok_or_else(|| TokenizerError::Encode(format!("merge token not in vocab: {a}")))?;
            let b_id = vocab_map.get(b).copied()
                .ok_or_else(|| TokenizerError::Encode(format!("merge token not in vocab: {b}")))?;
            let merged = vocab_map.get(&format!("{a}{b}")).copied()
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
            let best = tokens.windows(2)
                .enumerate()
                .filter_map(|(i, pair)| {
                    self.merge_map.get(&(pair[0], pair[1]))
                        .map(|&merged| (i, merged, self.priority(pair[0], pair[1])))
                })
                .min_by_key(|&(_, _, priority)| priority);

            let Some((pos, merged, _)) = best else { break; };
            tokens[pos] = merged;
            tokens.remove(pos + 1);
        }
        tokens
    }

    fn priority(&self, a: u32, b: u32) -> usize {
        self.merges.iter().position(|&(x, y, _)| x == a && y == b)
            .unwrap_or(usize::MAX)
    }
}

/// Simple GPT-2-style pre-tokenization: split on whitespace, keeping the
/// leading space attached to each word (LLaMA convention).
pub fn pretokenize_gpt2(text: &str) -> Vec<&str> {
    // TODO: full GPT-2 regex: r"'s|'t|'re|...|[a-z]+|[A-Z]+|[0-9]+|[^\s]+"
    // For now: split into whitespace-separated chunks, prepend space marker.
    text.split_inclusive(' ').collect()
}
