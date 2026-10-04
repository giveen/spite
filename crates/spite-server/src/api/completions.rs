//! POST /v1/completions  (legacy text completion endpoint)

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.7
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: &'static str,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub async fn create_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let created = super::chat::unix_now();

    let mut exec = state.executor.lock().unwrap();
    let ids = match state.tokenizer.encode(&req.prompt, true) {
        Ok(ids) => ids,
        Err(e) => {
            return error_response(&req.model, created, &e.to_string());
        }
    };
    let prompt_tokens = ids.len();
    let pieces = match exec.generate(&state.tokenizer, &ids, req.max_tokens, req.temperature, 0) {
        Ok(pieces) => pieces,
        Err(e) => {
            return error_response(&req.model, created, &e.to_string());
        }
    };
    let completion_tokens = pieces.len();
    let text: String = pieces.into_iter().map(|(_, s)| s).collect();

    Json(CompletionResponse {
        id: format!("cmpl-{created:x}"),
        object: "text_completion",
        created,
        model: req.model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

fn error_response(model: &str, created: u64, err: &str) -> Json<CompletionResponse> {
    Json(CompletionResponse {
        id: format!("cmpl-{created:x}"),
        object: "text_completion",
        created,
        model: model.to_owned(),
        choices: vec![CompletionChoice {
            text: format!("[spite error: {err}]"),
            index: 0,
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    })
}
