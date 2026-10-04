//! SentencePiece Unigram tokenizer (LLaMA / Mistral style).
//!
//! Unigram is a probabilistic model: each token has a log-probability score.
//! Encoding selects the token sequence that maximizes the sum of scores
//! (Viterbi decoding over a trie of all vocab entries).
//!
//! # Special byte tokens
//!
//! Rare bytes that don't appear in the vocab as UTF-8 sequences are
//! represented as `<0xNN>` hex-escape tokens (token_type = Byte).
//! When decoding, `Ġ` (U+0120) → space, `▁` (U+2581) → space.
//!
//! # GGUF keys read
//!
//! - `tokenizer.ggml.tokens`     — token strings
//! - `tokenizer.ggml.scores`     — log-probabilities
//! - `tokenizer.ggml.token_type` — flags (Normal=1, Byte=6, ...)

use crate::TokenizerError;

/// A trie node used for Viterbi forward pass.
struct TrieNode {
    children: Vec<(u8, usize)>, // (byte, child_node_idx), sorted
    token_id: Option<u32>,
    score: f32,
}

/// SentencePiece Unigram model.
pub struct UnigramModel {
    nodes: Vec<TrieNode>,
    n_vocab: usize,
}

impl UnigramModel {
    /// Build trie from vocab tokens and their scores.
    pub fn new(tokens: &[String], scores: &[f32]) -> Result<Self, TokenizerError> {
        if tokens.len() != scores.len() {
            return Err(TokenizerError::VocabMismatch {
                tokens: tokens.len(),
                scores: scores.len(),
            });
        }
        let n_vocab = tokens.len();
        // Root node
        let mut nodes = vec![TrieNode {
            children: vec![],
            token_id: None,
            score: 0.0,
        }];

        for (id, (tok, &score)) in tokens.iter().zip(scores.iter()).enumerate() {
            let bytes = tok.as_bytes();
            if bytes.is_empty() {
                continue;
            }
            let mut node = 0usize;
            for &b in bytes {
                let next = nodes[node]
                    .children
                    .iter()
                    .find(|&&(c, _)| c == b)
                    .map(|&(_, n)| n);
                if let Some(n) = next {
                    node = n;
                } else {
                    let new_node = nodes.len();
                    nodes[node].children.push((b, new_node));
                    nodes.push(TrieNode {
                        children: vec![],
                        token_id: None,
                        score: 0.0,
                    });
                    node = new_node;
                }
            }
            nodes[node].token_id = Some(id as u32);
            nodes[node].score = score;
        }

        Ok(Self { nodes, n_vocab })
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Encode text to token ids via Viterbi decoding.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        // best[i] = (score, prev_pos, token_id) for position i
        let neg_inf = f32::NEG_INFINITY;
        let mut best_score: Vec<f32> = vec![neg_inf; n + 1];
        let mut best_prev: Vec<usize> = vec![0; n + 1];
        let mut best_tok: Vec<u32> = vec![0; n + 1];
        best_score[0] = 0.0;

        for start in 0..n {
            if best_score[start] == neg_inf {
                continue;
            }
            // Walk trie from 'start'
            let mut node = 0usize;
            for end in start..n {
                let b = bytes[end];
                let next = self.nodes[node]
                    .children
                    .iter()
                    .find(|&&(c, _)| c == b)
                    .map(|&(_, n)| n);
                let Some(nxt) = next else {
                    break;
                };
                node = nxt;
                if let Some(tok_id) = self.nodes[node].token_id {
                    let s = best_score[start] + self.nodes[node].score;
                    if s > best_score[end + 1] {
                        best_score[end + 1] = s;
                        best_prev[end + 1] = start;
                        best_tok[end + 1] = tok_id;
                    }
                }
            }
        }

        // Backtrack
        let mut out = Vec::new();
        let mut pos = n;
        while pos > 0 {
            out.push(best_tok[pos]);
            pos = best_prev[pos];
        }
        out.reverse();
        out
    }
}
