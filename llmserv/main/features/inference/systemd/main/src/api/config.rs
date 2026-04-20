//! Typed application-config schema for the daemon.
//!
//! The schema mirrors the sections of `llmserv/main/config/application.toml`.
//! Only sections the daemon consumes are defined here; unknown sections are
//! tolerated and ignored. Backend-selection types (`ModelSpec`,
//! `ModelBackend`, `ModelSource`) live in `swe_inference_backend_api` so backend crates
//! can consume them without pulling in the daemon's HTTP stack.

use swe_inference_backend_api::ModelSpec;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub model: ModelSpec,
    #[serde(default)]
    pub runtime: RuntimeSpec,
    #[serde(default)]
    pub throttle: ThrottleSpec,
    #[serde(default)]
    pub generation: GenerationSpec,
    #[serde(default)]
    pub logging: LoggingSpec,
}

/// Per-request generation limits applied uniformly by the daemon.
#[derive(Debug, Deserialize)]
pub struct GenerationSpec {
    /// Wall-clock cap on a single completion. `0` disables the deadline,
    /// preserving prior unbounded behavior. Any positive value is wired
    /// into `CompletionParams.deadline` so a stuck forward pass cannot
    /// hold a throttle permit indefinitely.
    #[serde(default)]
    pub request_timeout_secs: u64,
}

impl Default for GenerationSpec {
    fn default() -> Self {
        Self {
            request_timeout_secs: 0,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

fn default_host() -> String {
    "127.0.0.1".into()
}

fn default_port() -> u16 {
    8080
}

#[derive(Debug, Deserialize)]
pub struct RuntimeSpec {
    #[serde(default = "default_opt_profile")]
    pub opt_profile: String,
    #[serde(default)]
    pub threads: usize,
}

impl Default for RuntimeSpec {
    fn default() -> Self {
        Self {
            opt_profile: default_opt_profile(),
            threads: 0,
        }
    }
}

fn default_opt_profile() -> String {
    "optimized".into()
}

/// Admission-control provider selection (DI).
#[derive(Debug, Deserialize)]
pub struct ThrottleSpec {
    #[serde(default = "default_throttle_provider")]
    pub provider: String,
    #[serde(default)]
    pub semaphore: SemaphoreThrottleSpec,
}

impl Default for ThrottleSpec {
    fn default() -> Self {
        Self {
            provider: default_throttle_provider(),
            semaphore: SemaphoreThrottleSpec::default(),
        }
    }
}

fn default_throttle_provider() -> String {
    "semaphore".into()
}

#[derive(Debug, Deserialize)]
pub struct SemaphoreThrottleSpec {
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
}

impl Default for SemaphoreThrottleSpec {
    fn default() -> Self {
        Self {
            max_concurrent: default_max_concurrent(),
        }
    }
}

fn default_max_concurrent() -> usize {
    2
}

#[derive(Debug, Deserialize)]
pub struct LoggingSpec {
    #[serde(default = "default_log_level")]
    pub level: String,
}

impl Default for LoggingSpec {
    fn default() -> Self {
        Self {
            level: default_log_level(),
        }
    }
}

fn default_log_level() -> String {
    "info".into()
}
