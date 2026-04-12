use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

use swe_ml_embed_server::load_gguf;
use swe_ml_embed_server::build_embedding_router;

/// swe-ml-embed — HTTP daemon for embedding models.
///
/// Serves an OpenAI-compatible /v1/embeddings endpoint.
#[derive(Parser)]
#[command(name = "swe-ml-embed", version, about)]
struct Cli {
    /// Path to a GGUF embedding model file.
    gguf_path: PathBuf,

    /// Address to bind (default: 127.0.0.1).
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on (default: 8091).
    #[arg(long, default_value_t = 8091)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let embedding_state = load_gguf(&cli.gguf_path)?;
    let model_id = embedding_state.model_id.clone();

    let state = Arc::new(embedding_state);
    let app = build_embedding_router(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    log::info!("swe-ml-embed serving '{}' on http://{}", model_id, addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
