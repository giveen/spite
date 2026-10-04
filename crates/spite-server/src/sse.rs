//! Server-Sent Events helpers for streaming token output.
//!
//! The wire format matches OpenAI's streaming chat completions so that
//! existing clients (Open WebUI, SillyTavern, etc.) work without changes.
//!
//! Each token becomes one `data: <json>\n\n` frame.
//! The stream ends with `data: [DONE]\n\n`.

use axum::response::sse::Event;
use serde::Serialize;
use tokio_stream::Stream;

use crate::api::chat::ChatCompletionChunk;

/// Convert an async stream of token strings into SSE events.
pub fn token_stream(
    tokens: impl Stream<Item = String> + Send + 'static,
    completion_id: String,
    model: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    use tokio_stream::StreamExt as _;

    let id = completion_id;

    tokens
        .map(move |token| {
            let chunk = ChatCompletionChunk::token(&id, &model, &token);
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(json))
        })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))))
}

/// A single SSE data payload for non-streaming responses.
pub fn single_event<T: Serialize>(value: &T) -> Event {
    Event::default().data(serde_json::to_string(value).unwrap_or_default())
}
