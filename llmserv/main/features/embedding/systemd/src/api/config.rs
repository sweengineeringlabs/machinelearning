//! Typed application-config schema for the embedding server.
//!
//! Mirrors the sections of `llmserv/main/config/application.toml` that
//! this binary consumes.

use serde::Deserialize;
use swe_systemd::DaemonConfig;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub embedding: EmbeddingSection,
    #[serde(default)]
    pub logging: LoggingSpec,
}

impl DaemonConfig for AppConfig {
    fn logging_level(&self) -> &str {
        &self.logging.level
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct EmbeddingSection {
    #[serde(default)]
    pub server: EmbeddingServerConfig,
    #[serde(default)]
    pub model: EmbeddingModelSpec,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for EmbeddingServerConfig {
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
    8081
}

#[derive(Debug, Deserialize, Default)]
pub struct EmbeddingModelSpec {
    /// Path to a GGUF embedding model. Required — empty string means unset.
    #[serde(default)]
    pub gguf_path: String,
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
