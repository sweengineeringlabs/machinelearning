//! `infer` — inference HTTP daemon.
//!
//! Serves OpenAI-compatible chat + embedding endpoints over a loaded
//! LLM. Config is file-driven; all runtime behavior (backend, model,
//! throttling, quantization, timeouts, logging) lives in
//! `application.toml`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use clap::{Parser, Subcommand};

use swe_inference_systemd::{
    AppConfig, AppState, Model, ModelBackend, ModelBackendLoader, NativeRustBackendLoader,
    SemaphoreThrottle, Throttle, build_router, load_config,
};
use swe_llmmodel_model::OptProfile;
use swe_inference_thread_config::ThreadConfig;

/// `infer` — inference HTTP daemon.
///
/// Loads an LLM (safetensors via HuggingFace Hub, or a local GGUF file)
/// and serves OpenAI-compatible completions and embedding requests.
///
/// Config sources (later overrides earlier):
///   1. Bundled default (compiled into the binary)
///   2. $XDG_CONFIG_DIRS/llmserv/application.toml
///   3. $XDG_CONFIG_HOME/llmserv/application.toml
///      (on Windows: %APPDATA%\llmserv\application.toml)
///
/// EXAMPLES:
///
///   # Start with the default config:
///   infer serve
///
///   # Override the config via XDG:
///   mkdir -p ~/.config/llmserv
///   cat > ~/.config/llmserv/application.toml <<EOF
///   [server]
///   host = "127.0.0.1"
///   port = 8080
///   [model]
///   backend = "native_rust"
///   source  = "safetensors"
///   id      = "openai-community/gpt2"
///   EOF
///   infer serve
///
///   # Hit the running daemon:
///   curl -s http://127.0.0.1:8080/health
///   curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
///     -H 'Content-Type: application/json' \
///     -d '{"model":"gpt2","messages":[{"role":"user","content":"Hello"}]}'
#[derive(Parser)]
#[command(
    name = "infer",
    version,
    about = "Inference HTTP daemon — OpenAI-compatible /v1/chat/completions + /v1/embeddings",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the inference HTTP daemon.
    ///
    /// Resolves the backend (`native_rust` or `llama_cpp` when built with
    /// `--features backend-llama-cpp`), loads the model per `[model]`, and
    /// binds to `[server].host:port`. Runs in the foreground; send
    /// SIGINT/Ctrl-C to stop.
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

    let profile = parse_opt_profile(&loaded.app.runtime.opt_profile)?;
    profile
        .apply()
        .map_err(|e| anyhow::anyhow!("Failed to apply runtime config: {}", e))?;

    let thread_config = swe_inference_thread_config::AutoThreadConfig::new();
    log::info!(
        "Thread config: {} threads ({})",
        thread_config.num_threads(),
        thread_config.describe()
    );

    let registry = build_backend_registry();
    let model = load_model(&registry, &loaded.app, profile, &loaded.merged_toml)?;
    let model_id = model.model_id().to_string();

    let throttle = build_throttle(&loaded.app)?;
    log::info!(
        "Admission control: provider={}, capacity={}",
        loaded.app.throttle.provider,
        throttle.capacity()
    );

    let request_timeout = match loaded.app.generation.request_timeout_secs {
        0 => None,
        secs => Some(Duration::from_secs(secs)),
    };
    log::info!(
        "Generation timeout: {}",
        request_timeout
            .map(|d| format!("{}s", d.as_secs()))
            .unwrap_or_else(|| "disabled".into())
    );

    let state = Arc::new(AppState {
        model,
        throttle,
        request_timeout,
    });
    let app = build_router(state);

    let addr: SocketAddr = format!("{}:{}", loaded.app.server.host, loaded.app.server.port).parse()?;
    log::info!("infer: serving model '{}' on http://{}", model_id, addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn parse_opt_profile(s: &str) -> Result<OptProfile> {
    match s {
        "optimized" => Ok(OptProfile::Optimized),
        "baseline" => Ok(OptProfile::Baseline),
        "aggressive" => Ok(OptProfile::Aggressive),
        other => bail!(
            "Unknown runtime.opt_profile '{}' (expected: optimized, baseline, aggressive)",
            other
        ),
    }
}

fn apply_logging_filter(cfg: &AppConfig) {
    // env_logger reads RUST_LOG; set it from config if not already set.
    // SAFETY: runs single-threaded before any tokio/rayon threads spawn.
    if std::env::var_os("RUST_LOG").is_none() {
        unsafe {
            std::env::set_var("RUST_LOG", &cfg.logging.level);
        }
    }
}

type BackendRegistry = HashMap<ModelBackend, Box<dyn ModelBackendLoader>>;

/// SPI registration point — every supported backend registers a loader here.
/// The native-Rust loader is always present; `LlamaCpp` is included only
/// when the daemon is built with `--features backend-llama-cpp`.
fn build_backend_registry() -> BackendRegistry {
    let mut reg: BackendRegistry = HashMap::new();
    reg.insert(ModelBackend::NativeRust, Box::new(NativeRustBackendLoader));

    #[cfg(feature = "backend-llama-cpp")]
    reg.insert(
        ModelBackend::LlamaCpp,
        Box::new(swe_inference_backend_llama_cpp::LlamaCppBackendLoader),
    );

    reg
}

fn load_model(
    registry: &BackendRegistry,
    cfg: &AppConfig,
    profile: OptProfile,
    merged_toml: &str,
) -> Result<Box<dyn Model>> {
    let loader = registry.get(&cfg.model.backend).ok_or_else(|| {
        anyhow!(
            "[model].backend = {:?} is not registered — rebuild with the \
             appropriate feature flag (e.g. --features backend-llama-cpp)",
            cfg.model.backend
        )
    })?;
    log::info!("Model backend: {}", loader.name());
    loader.load(&cfg.model, profile, merged_toml)
}

/// DI point: construct the throttle implementation selected by config.
fn build_throttle(cfg: &AppConfig) -> Result<Box<dyn Throttle>> {
    match cfg.throttle.provider.as_str() {
        "semaphore" => Ok(Box::new(SemaphoreThrottle::new(
            cfg.throttle.semaphore.max_concurrent,
        ))),
        other => bail!(
            "Unknown [throttle].provider '{}' (expected: semaphore)",
            other
        ),
    }
}
