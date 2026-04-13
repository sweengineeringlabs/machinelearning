use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};

use swellmd::{
    AppConfig, AppState, Model, ModelBackend, ModelBackendLoader, NativeRustBackendLoader,
    SemaphoreThrottle, Throttle, build_router, load_config,
};
use rustml_model::OptProfile;
use rustml_thread_config::ThreadConfig;

/// swellmd — HTTP daemon for RustML LLM inference.
///
/// Serves an OpenAI-compatible /v1/chat/completions endpoint. All behavior
/// is driven by `llmserv/main/config/application.toml` plus optional XDG
/// overlays. No CLI flags — set `XDG_CONFIG_HOME` to point at an override
/// directory if you need a different configuration for one invocation.
#[tokio::main]
async fn main() -> Result<()> {
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

    let thread_config = rustml_thread_config::AutoThreadConfig::new();
    log::info!("Thread config: {} threads ({})", thread_config.num_threads(), thread_config.describe());

    let registry = build_backend_registry();
    let model = load_model(&registry, &loaded.app, profile, &loaded.merged_toml)?;
    let model_id = model.model_id().to_string();

    let throttle = build_throttle(&loaded.app)?;
    log::info!(
        "Admission control: provider={}, capacity={}",
        loaded.app.throttle.provider,
        throttle.capacity()
    );

    let state = Arc::new(AppState { model, throttle });
    let app = build_router(state);

    let addr: SocketAddr = format!("{}:{}", loaded.app.server.host, loaded.app.server.port).parse()?;
    log::info!("swellmd serving model '{}' on http://{}", model_id, addr);

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
        Box::new(rustml_backend_llama_cpp::LlamaCppBackendLoader),
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
