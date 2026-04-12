use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, bail};

use swe_ml_embed_server::{
    AppConfig, build_embedding_router, load_config, load_gguf,
};

/// swe-ml-embed — HTTP daemon for embedding models.
///
/// Serves an OpenAI-compatible /v1/embeddings endpoint. Config-driven
/// via `llmserv/main/config/application.toml` with optional XDG
/// overlays. No CLI flags — set XDG_CONFIG_HOME to override.
#[tokio::main]
async fn main() -> Result<()> {
    let loaded = load_config()?;
    apply_logging_filter(&loaded.app);
    env_logger::init();

    log::info!("Config sources ({}):", loaded.sources.len());
    for src in &loaded.sources {
        log::info!("  - {}", src);
    }

    let gguf_path = resolve_model_path(&loaded.app)?;
    let embedding_state = load_gguf(&gguf_path)?;
    let model_id = embedding_state.model_id.clone();

    let state = Arc::new(embedding_state);
    let app = build_embedding_router(state);

    let addr: SocketAddr = format!(
        "{}:{}",
        loaded.app.embedding.server.host, loaded.app.embedding.server.port
    )
    .parse()?;
    log::info!("swe-ml-embed serving '{}' on http://{}", model_id, addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn apply_logging_filter(cfg: &AppConfig) {
    // SAFETY: runs single-threaded before any tokio/rayon threads spawn.
    if std::env::var_os("RUST_LOG").is_none() {
        unsafe {
            std::env::set_var("RUST_LOG", &cfg.logging.level);
        }
    }
}

fn resolve_model_path(cfg: &AppConfig) -> Result<PathBuf> {
    let path = &cfg.embedding.model.gguf_path;
    if path.is_empty() {
        bail!(
            "[embedding.model].gguf_path is not set. Edit application.toml or provide \
             an XDG override with the path to your embedding GGUF file."
        );
    }
    Ok(PathBuf::from(path))
}
