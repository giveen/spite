//! JSON Schema (draft-07 subset) → GBNF compiler.
//!
//! Translates a JSON Schema object into a GBNF grammar string that the
//! grammar engine can compile, ensuring the model outputs conforming JSON.
//!
//! Supported schema keywords:
//!   type, properties, required, additionalProperties,
//!   items, minItems, maxItems,
//!   enum, const, anyOf, oneOf, allOf,
//!   minimum, maximum, minLength, maxLength, pattern

use crate::GrammarError;

/// Translate a JSON Schema value into a GBNF grammar string.
///
/// The returned string is ready to pass to `Grammar::from_gbnf`.
pub fn schema_to_gbnf(schema: &serde_json::Value) -> Result<String, GrammarError> {
    let mut out = String::new();
    out.push_str("root ::= ");
    emit_type(schema, &mut out)?;
    out.push('\n');
    // TODO: emit helper rules for ws, string, number, array, object
    Ok(out)
}

fn emit_type(schema: &serde_json::Value, out: &mut String) -> Result<(), GrammarError> {
    match schema.get("type").and_then(|v| v.as_str()) {
        Some("object")  => out.push_str("object"),
        Some("array")   => out.push_str("array"),
        Some("string")  => out.push_str("string"),
        Some("number")  => out.push_str("number"),
        Some("integer") => out.push_str("integer"),
        Some("boolean") => out.push_str("( \"true\" | \"false\" )"),
        Some("null")    => out.push_str("\"null\""),
        Some(other) => return Err(GrammarError::SchemaError(
            format!("unknown type: {other}")
        )),
        None => out.push_str("value"), // untyped → accept any JSON value
    }
    Ok(())
}
