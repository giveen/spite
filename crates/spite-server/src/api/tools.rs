//! Tool / function calling (OpenAI tool_use format).
//!
//! Strategy:
//!   1. Inject tool definitions into the system prompt in the model's
//!      expected format (each architecture uses a slightly different schema).
//!   2. Generate normally with `finish_reason = "stop"` or `"tool_calls"`.
//!   3. After generation, detect a tool call in the output by looking for
//!      the model-specific delimiter (e.g. `<tool_call>` for LLaMA-4-Instruct).
//!   4. Parse the JSON arguments and return a `ToolCall` in the response.
//!
//! Structured output via `spite-grammar` can constrain the argument JSON
//! to match the tool's parameter schema exactly.

use serde::{Deserialize, Serialize};

/// A tool definition as sent by the client in the chat request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String, // always "function"
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

/// A tool call produced by the model.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,           // random id, e.g. "call_abc123"
    pub r#type: &'static str, // "function"
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String, // JSON-encoded argument object
}

/// Inject tool definitions into the system prompt for a given architecture.
///
/// LLaMA-4-Instruct expects a `<|python_tag|>`-style preamble;
/// Mistral 4 uses `[TOOL_CALLS]` markers; generic fallback uses a plain
/// "Available functions:" block.
pub fn inject_tools(arch: &str, system: &mut String, tools: &[ToolDefinition]) {
    if tools.is_empty() {
        return;
    }
    match arch {
        "llama4" => inject_llama4(system, tools),
        "mistral4" | "magistral" => inject_mistral(system, tools),
        _ => inject_generic(system, tools),
    }
}

fn inject_llama4(system: &mut String, tools: &[ToolDefinition]) {
    // TODO: format tools as the Llama-4-Instruct tool preamble
    let _ = (system, tools);
}

fn inject_mistral(system: &mut String, tools: &[ToolDefinition]) {
    // TODO: format tools as Mistral 4 function-calling preamble
    let _ = (system, tools);
}

fn inject_generic(system: &mut String, tools: &[ToolDefinition]) {
    system.push_str("\n\nAvailable functions (call as JSON):\n");
    for t in tools {
        let sig = serde_json::to_string(&t.function.parameters).unwrap_or_default();
        system.push_str(&format!(
            "  {} — {} — params: {}\n",
            t.function.name, t.function.description, sig
        ));
    }
}

/// Detect and parse a tool call from model output.
///
/// Returns `Some(ToolCall)` when the model chose to call a tool,
/// `None` for plain text responses.
pub fn parse_tool_call(output: &str, tools: &[ToolDefinition]) -> Option<ToolCall> {
    // TODO: parse <tool_call>{"name":…,"arguments":…}</tool_call> or
    //       [TOOL_CALLS] [{"name":…}] depending on arch
    let _ = (output, tools);
    None
}

/// Generate a random tool call id.
pub fn new_call_id() -> String {
    // TODO: use a real CSPRNG; this is a placeholder
    format!("call_{:016x}", 0xdeadbeef_u64)
}
