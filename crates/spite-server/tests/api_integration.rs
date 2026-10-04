//! Integration tests for the HTTP API.
//!
//! Starts a real axum server bound to a random port, sends HTTP requests
//! using reqwest, and checks responses. No GPU needed — the fake GGUF model
//! satisfies the loader and runs real CPU inference.

use std::sync::Arc;

use axum::Router;
use spite_testkit::FakeGguf;
use tokio::net::TcpListener;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Spawn the server on a random OS-assigned port; return the base URL.
async fn start_server() -> String {
    let fake = FakeGguf {
        arch: "mistral4".to_string(),
        ..Default::default()
    };
    let tmp = fake.write_to_tempfile().unwrap();
    let state = Arc::new(
        spite_server::AppState::load(
            tmp.path(),
            std::path::Path::new("build/kernels"), // non-existent is fine — no kernels needed
            "generic",
            4,
        )
        .expect("AppState::load with fake model"),
    );

    // Keep tempfile alive for the server's lifetime
    std::mem::forget(tmp);

    let app: Router = spite_server::api::router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    format!("http://{addr}")
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok() {
    let base = start_server().await;
    let resp = reqwest::get(format!("{base}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn models_list_returns_model_card() {
    let base = start_server().await;
    let resp = reqwest::get(format!("{base}/v1/models")).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert!(
        !data.is_empty(),
        "models list should contain at least one entry"
    );
    assert_eq!(data[0]["object"], "model");
    // model id should be the arch from the fake GGUF
    assert_eq!(data[0]["id"], "mistral4");
}

#[tokio::test]
async fn dispatch_info_contains_arch_fields() {
    let base = start_server().await;
    let resp = reqwest::get(format!("{base}/spite/dispatch"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["model_arch"].is_string());
    assert!(body["gpu_arch"].is_string());
}

#[tokio::test]
async fn chat_completion_non_stream_returns_400_or_stub() {
    let base = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "mistral4",
            "messages": [{"role": "user", "content": "AB"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Server is wired; executor stubs return NotInitialized.
    // We accept either a well-formed stub response (200) or a
    // 500/stub while the executor isn't implemented — what matters
    // is that the server doesn't panic and returns valid JSON or text.
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.is_empty() && !content.contains("not yet implemented"),
        "expected generated text, got: {content}"
    );
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0);
}

#[tokio::test]
async fn unknown_route_returns_404() {
    let base = start_server().await;
    let resp = reqwest::get(format!("{base}/v1/does-not-exist"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}
