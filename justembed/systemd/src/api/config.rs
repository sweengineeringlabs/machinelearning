//! Typed application-config schema for the embedding server.
//!
//! Mirrors the sections of `llminference/main/config/application.toml` that
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
    #[serde(default)]
    pub grpc: EmbeddingGrpcConfig,
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

/// Bind + security config for the gRPC `EmbedService` server that runs
/// alongside the OpenAI-compatible REST endpoint.
///
/// Defaults assume a development workstation: bind to loopback, plaintext,
/// `allow_unauthenticated = true`.  Production deployments MUST set TLS
/// paths and (when an auth interceptor is wired) flip
/// `allow_unauthenticated` to `false`.
#[derive(Debug, Deserialize)]
pub struct EmbeddingGrpcConfig {
    /// Bind host.  Defaults to `127.0.0.1` — never bind 0.0.0.0 by default.
    #[serde(default = "default_grpc_host")]
    pub host: String,

    /// Bind port.  Defaults to `8181` (REST is `8081`).
    #[serde(default = "default_grpc_port")]
    pub port: u16,

    /// Max single gRPC message size in bytes.  Defaults to 4 MiB.
    #[serde(default = "default_max_message_bytes")]
    pub max_message_bytes: usize,

    /// When `false`, the server refuses to start unless an auth handler is
    /// wired up.  Defaults to `true` for compatibility with the initial
    /// rollout — flip to `false` once an auth interceptor is registered.
    #[serde(default = "default_allow_unauthenticated")]
    pub allow_unauthenticated: bool,

    /// Optional TLS / mTLS configuration.  Absent → plaintext (loopback only).
    #[serde(default)]
    pub tls: Option<EmbeddingGrpcTlsConfig>,
}

impl Default for EmbeddingGrpcConfig {
    fn default() -> Self {
        Self {
            host:                  default_grpc_host(),
            port:                  default_grpc_port(),
            max_message_bytes:     default_max_message_bytes(),
            allow_unauthenticated: default_allow_unauthenticated(),
            tls:                   None,
        }
    }
}

/// TLS material paths for the gRPC server.
#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingGrpcTlsConfig {
    /// Path to a PEM file with the server certificate chain (leaf first).
    pub cert_pem_path: String,
    /// Path to a PEM file with the server's private key.
    pub key_pem_path:  String,
    /// Optional path to a client-CA PEM file — when set, mTLS is enforced.
    #[serde(default)]
    pub client_ca_pem_path: Option<String>,
}

fn default_grpc_host() -> String { "127.0.0.1".into() }
fn default_grpc_port() -> u16 { 8181 }
fn default_max_message_bytes() -> usize { 4 * 1024 * 1024 }
fn default_allow_unauthenticated() -> bool { true }

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
