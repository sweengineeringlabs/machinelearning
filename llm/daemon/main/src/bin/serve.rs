use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, bail};
use clap::Parser;

use swellmd::{AppState, build_router, load_gguf, load_safetensors};
use rustml_nlp::{ConfigOps, OptProfile};

/// swellmd — HTTP daemon for RustML LLM inference.
///
/// Serves an OpenAI-compatible /v1/chat/completions endpoint.
#[derive(Parser)]
#[command(name = "swellmd", version, about)]
struct Cli {
    /// Path to a GGUF model file.
    #[arg(conflicts_with = "safetensors")]
    gguf_path: Option<PathBuf>,

    /// HuggingFace model ID to load via SafeTensors (e.g. openai-community/gpt2).
    #[arg(long)]
    safetensors: Option<String>,

    /// Address to bind (default: 127.0.0.1).
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on (default: 8080).
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Optimization profile: optimized (default), baseline, aggressive.
    #[arg(long, default_value = "optimized")]
    opt_profile: String,
}

fn parse_opt_profile(s: &str) -> Result<OptProfile> {
    match s {
        "optimized" => Ok(OptProfile::Optimized),
        "baseline" => Ok(OptProfile::Baseline),
        "aggressive" => Ok(OptProfile::Aggressive),
        other => bail!("Unknown --opt-profile '{}' (expected: optimized, baseline, aggressive)", other),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    if cli.gguf_path.is_none() && cli.safetensors.is_none() {
        bail!("Provide a GGUF model path or --safetensors <MODEL_ID>");
    }

    let profile = parse_opt_profile(&cli.opt_profile)?;
    profile
        .runtime_config()
        .apply()
        .map_err(|e| anyhow::anyhow!("Failed to apply runtime config: {}", e))?;

    let bundle = if let Some(ref model_id) = cli.safetensors {
        load_safetensors(model_id, profile)?
    } else {
        let path = cli.gguf_path.as_ref().unwrap();
        load_gguf(path, profile)?
    };

    let model_id = bundle.model_id.clone();
    let state = Arc::new(AppState { bundle });
    let app = build_router(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    log::info!("swellmd serving model '{}' on http://{}", model_id, addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
