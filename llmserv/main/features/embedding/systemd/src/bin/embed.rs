//! `embed` — embedding HTTP daemon.
//!
//! Hosts the OpenAI-compatible `POST /v1/embeddings` endpoint over a
//! GGUF embedding model (nomic-bert at the moment). Config is file-
//! driven; there are no per-flag overrides on the CLI.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use swe_embedding_systemd::{
    AppConfig, build_embedding_router, load_config, load_gguf,
};

/// `embed` — embedding HTTP daemon.
///
/// The daemon loads a GGUF embedding model (e.g. nomic-embed-text) and
/// serves OpenAI-compatible embedding requests.
///
/// Config sources (later overrides earlier):
///   1. Bundled default (compiled into the binary)
///   2. $XDG_CONFIG_DIRS/llmserv/application.toml
///   3. $XDG_CONFIG_HOME/llmserv/application.toml
///      (on Windows: %APPDATA%\llmserv\application.toml)
///
/// EXAMPLES:
///
///   # Start with the default config (requires [embedding.model].gguf_path set):
///   embed serve
///
///   # Override the config via XDG:
///   mkdir -p ~/.config/llmserv
///   cat > ~/.config/llmserv/application.toml <<EOF
///   [embedding.model]
///   gguf_path = "/path/to/nomic-embed-text.Q8_0.gguf"
///   [embedding.server]
///   host = "127.0.0.1"
///   port = 8081
///   EOF
///   embed serve
///
///   # Hit the running daemon:
///   curl -s http://127.0.0.1:8081/health
///   curl -s -X POST http://127.0.0.1:8081/v1/embeddings \
///     -H 'Content-Type: application/json' \
///     -d '{"input":"hello world","model":"nomic"}'
#[derive(Parser)]
#[command(
    name = "embed",
    version,
    about = "Embedding HTTP daemon — OpenAI-compatible /v1/embeddings",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the embedding HTTP daemon.
    ///
    /// Loads the GGUF model referenced by `[embedding.model].gguf_path`
    /// in the merged config and binds to `[embedding.server].host:port`.
    /// Runs in the foreground; send SIGINT/Ctrl-C to stop.
    Serve,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve => run_serve(),
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn run_serve() -> Result<()> {
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
    log::info!("embed: serving '{}' on http://{}", model_id, addr);

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
