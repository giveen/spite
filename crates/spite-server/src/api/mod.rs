use std::sync::Arc;

use axum::{Router, routing::{get, post}};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::AppState;

pub mod chat;
pub mod completions;
pub mod models;
pub mod embeddings;
pub mod tools;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/models",                    get(models::list_models))
        .route("/v1/completions",               post(completions::create_completion))
        .route("/v1/chat/completions",          post(chat::create_chat_completion))
        // Health
        .route("/health",                       get(health))
        // Spite-specific: dump which kernels are active
        .route("/spite/dispatch",               get(dispatch_info))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health() -> &'static str { "ok" }

async fn dispatch_info(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "gpu_arch": state.gpu_arch,
        "model_arch": state.model.arch(),
    }))
}
