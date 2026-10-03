//! GBNF (GGML BNF) grammar parser.
//!
//! Tokenizes and parses GBNF rules into an intermediate representation
//! that the compiler can turn into an NFA for incremental parsing.
//!
//! GBNF syntax:
//!   rule-name ::= alternative1 | alternative2
//!   alternative ::= element+
//!   element ::= rule-ref | literal | char-class | element? | element* | element+

use crate::GrammarError;

/// A parsed GBNF rule before compilation to an NFA.
pub struct Rule {
    pub name: String,
    pub alts: Vec<Alt>,
}

/// One alternative (a sequence of elements).
pub struct Alt {
    pub elements: Vec<Element>,
}

pub enum Element {
    RuleRef(String),
    Literal(String),
    CharClass(String), // e.g. [a-zA-Z0-9]
    Optional(Box<Element>),
    ZeroOrMore(Box<Element>),
    OneOrMore(Box<Element>),
}

/// Parse a GBNF source string into a list of rules.
pub fn parse(_src: &str) -> Result<Vec<Rule>, GrammarError> {
    // TODO: tokenize → lex → parse rule definitions
    //       validate that a "root" rule exists
    Ok(vec![])
}
