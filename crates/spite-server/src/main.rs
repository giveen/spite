use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use spite_server::{AppState, api};

#[derive(Parser)]
#[command(
    name = "spite-server",
    about = "OpenAI-compatible LLM inference server"
)]
struct Cli {
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "kernels")]
    kernels_dir: PathBuf,

    #[arg(long, env = "SPITE_GPU_ARCH")]
    gpu_arch: Option<String>,

    #[arg(long, default_value = "0.0.0.0:8080")]
    host: SocketAddr,

    /// Maximum concurrent requests. Defaults to 1 (single-GPU safe default).
    #[arg(long, default_value_t = 1)]
    max_concurrent: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("spite_server=info".parse()?))
        .init();

    let cli = Cli::parse();

    let gpu_arch = cli.gpu_arch.unwrap_or_else(spite_dispatch::detect_gpu_arch);

    let state = Arc::new(AppState::load(
        &cli.model,
        &cli.kernels_dir,
        &gpu_arch,
        cli.max_concurrent,
    )?);

    info!(
        model = %cli.model.display(),
        gpu   = %gpu_arch,
        host  = %cli.host,
        "spite-server ready"
    );

    let app = api::router(state);
    let listener = tokio::net::TcpListener::bind(cli.host).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
