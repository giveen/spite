//! GET /v1/models

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::AppState;

#[derive(Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: state.model_arch.clone(),
            object: "model",
            created,
            owned_by: "spite",
        }],
    })
}
