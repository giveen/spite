//! Structured output: GBNF grammar and JSON schema enforcement.
//!
//! During sampling, `constrain_logits` masks the logit distribution so
//! only tokens that can advance a valid parse remain selectable.
//! This guarantees the model outputs valid JSON, SQL, or any GBNF grammar.
//!
//! GBNF is the grammar format used by llama.cpp — a simple BNF dialect
//! where rules look like:
//!   root   ::= object
//!   object ::= "{" ws members ws "}"
//!   value  ::= object | array | string | number | "true" | "false" | "null"

pub mod gbnf;
pub mod json_schema;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GrammarError {
    #[error("parse error at position {pos}: {msg}")]
    ParseError { pos: usize, msg: String },
    #[error("invalid grammar: {0}")]
    InvalidGrammar(String),
    #[error("no valid token advances the grammar — dead end")]
    DeadEnd,
    #[error("json schema error: {0}")]
    SchemaError(String),
}

/// A compiled grammar ready for logit masking.
pub struct Grammar {
    // TODO: NFA/DFA built from parsed GBNF rules
    _rules: Vec<()>,
}

impl Grammar {
    /// Parse and compile a GBNF grammar string.
    pub fn from_gbnf(src: &str) -> Result<Self, GrammarError> {
        gbnf::parse(src)?;
        Ok(Self { _rules: vec![] })
    }

    /// Build a grammar that enforces a JSON schema (draft-07 subset).
    pub fn from_json_schema(schema: &serde_json::Value) -> Result<Self, GrammarError> {
        let _gbnf = json_schema::schema_to_gbnf(schema)?;
        Ok(Self { _rules: vec![] })
    }

    /// Set `logits[i] = f32::NEG_INFINITY` for every token that cannot legally
    /// follow the current grammar state. Call before sampling each token.
    pub fn constrain_logits(&self, _logits: &mut [f32], _vocab_size: usize) {
        // TODO: query FSM for the valid-next-token bitset, mask everything else
    }

    /// Advance the grammar FSM after token `token_id` was accepted.
    pub fn advance(&mut self, _token_id: u32) -> Result<(), GrammarError> {
        // TODO: step the FSM; return DeadEnd if no valid next state
        Ok(())
    }

    /// Returns `true` when the current FSM state is a valid end-of-sequence.
    pub fn is_complete(&self) -> bool { false }
}
