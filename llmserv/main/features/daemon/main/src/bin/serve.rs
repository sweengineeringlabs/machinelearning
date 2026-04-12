use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, bail};

use swellmd::{
    AppConfig, AppState, Model, ModelSource, SemaphoreThrottle, Throttle,
    build_router, load_config, load_gguf, load_safetensors,
};
use rustml_model::OptProfile;
use rustml_thread_config::ThreadConfig;
use rustml_compute::ComputeBackend;

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

    let compute = rustml_compute::CpuBackend;
    log::info!("Compute backend: {}", compute.name());

    let model = load_model(&loaded.app, profile, &loaded.merged_toml)?;
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

fn load_model(cfg: &AppConfig, profile: OptProfile, merged_toml: &str) -> Result<Box<dyn Model>> {
    match cfg.model.source {
        ModelSource::Safetensors => {
            let id = cfg.model.id.as_deref().ok_or_else(|| {
                anyhow::anyhow!("[model].source = \"safetensors\" requires [model].id")
            })?;
            Ok(Box::new(load_safetensors(id, profile, merged_toml)?))
        }
        ModelSource::Gguf => {
            let path_str = cfg.model.path.as_deref().ok_or_else(|| {
                anyhow::anyhow!("[model].source = \"gguf\" requires [model].path")
            })?;
            Ok(Box::new(load_gguf(&PathBuf::from(path_str), profile)?))
        }
    }
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
