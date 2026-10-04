//! POST /v1/chat/completions
//!
//! OpenAI chat completions API — streaming and non-streaming.
//! Wire format matches the OpenAI spec so drop-in clients work.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::AppState;

// ── Request ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.95
}

// ── Response types ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: &'static str,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chunk — one token.
#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatCompletionChunk {
    pub fn token(id: &str, model: &str, token: &str) -> Self {
        Self {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created: unix_now(),
            model: model.to_owned(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(token.to_owned()),
                },
                finish_reason: None,
            }],
        }
    }

    pub fn stop(id: &str, model: &str) -> Self {
        Self {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created: unix_now(),
            model: model.to_owned(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop"),
            }],
        }
    }
}

// ── Handler ────────────────────────────────────────────────────────────────

pub async fn create_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let state = Arc::clone(&state);

    if req.stream {
        stream_response(state, req).await.into_response()
    } else {
        blocking_response(state, req).await.into_response()
    }
}

async fn blocking_response(_state: Arc<AppState>, req: ChatRequest) -> Json<ChatCompletion> {
    // TODO: run tokenizer → inference loop → detokenize
    let reply = format!(
        "[spite inference not yet implemented — model={}, messages={}]",
        req.model,
        req.messages.len()
    );

    Json(ChatCompletion {
        id: new_id(),
        object: "chat.completion",
        created: unix_now(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: reply,
            },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    })
}

async fn stream_response(
    _state: Arc<AppState>,
    req: ChatRequest,
) -> Sse<
    impl tokio_stream::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    // TODO: replace with real token stream from inference loop
    let id = new_id();
    let model = req.model.clone();

    let placeholder = vec![
        "[spite".to_owned(),
        " streaming".to_owned(),
        " not".to_owned(),
        " yet".to_owned(),
        " implemented]".to_owned(),
    ];

    let token_stream = tokio_stream::iter(placeholder);
    let sse_stream = crate::sse::token_stream(token_stream, id, model);

    Sse::new(sse_stream).keep_alive(axum::response::sse::KeepAlive::default())
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn new_id() -> String {
    format!("chatcmpl-{:x}", unix_now())
}
