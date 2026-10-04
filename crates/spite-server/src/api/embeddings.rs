//! POST /v1/embeddings — OpenAI-compatible embeddings endpoint.
//!
//! Runs the model in pooled-embedding mode: forward pass over the input,
//! then mean-pool the last hidden states across the sequence dimension to
//! produce a fixed-size vector per input string.
//!
//! The model must have `can_embed = true` in its `SpiteModelCaps`
//! (a future extension — not all generation models support embedding mode).

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    /// Single string or a batch of strings.
    pub input: EmbedInput,
    #[serde(default)]
    pub encoding_format: EmbedFormat,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbedInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbedFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub object: &'static str, // "list"
    pub data: Vec<EmbedObject>,
    pub model: String,
    pub usage: EmbedUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbedObject {
    pub object: &'static str, // "embedding"
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbedUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// Handler (wired in api/mod.rs once Executor is integrated):
//
// pub async fn create_embeddings(
//     State(state): State<Arc<AppState>>,
//     Json(req):    Json<EmbedRequest>,
// ) -> Result<Json<EmbedResponse>, StatusCode> {
//     let inputs = match req.input {
//         EmbedInput::Single(s) => vec![s],
//         EmbedInput::Batch(v)  => v,
//     };
//     let mut data = Vec::with_capacity(inputs.len());
//     let mut total_tokens = 0usize;
//     for (idx, text) in inputs.iter().enumerate() {
//         let tokens = state.tokenizer.encode(text)?;
//         total_tokens += tokens.len();
//         let hidden = state.executor.prefill(&tokens, &ctx)?;  // [seq, d_model]
//         let embedding = mean_pool(&hidden, d_model);           // [d_model]
//         data.push(EmbedObject { object: "embedding", embedding, index: idx });
//     }
//     Ok(Json(EmbedResponse { object: "list", data, model: req.model, usage: EmbedUsage {
//         prompt_tokens: total_tokens, total_tokens,
//     }}))
// }
//
// fn mean_pool(hidden: &[f32], d_model: usize) -> Vec<f32> {
//     let seq_len = hidden.len() / d_model;
//     let mut out = vec![0f32; d_model];
//     for t in 0..seq_len {
//         for d in 0..d_model { out[d] += hidden[t * d_model + d]; }
//     }
//     out.iter_mut().for_each(|x| *x /= seq_len as f32);
//     out
// }
