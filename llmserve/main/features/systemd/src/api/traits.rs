use serde::de::DeserializeOwned;

/// Marker for types that can be loaded via [`crate::load_config`].
///
/// The typed config must deserialize from TOML and must expose at least
/// one piece of daemon-lifecycle metadata: the logging level. That's the
/// minimum a daemon needs to wire up `env_logger` before its subsystems
/// start logging.
pub trait DaemonConfig: DeserializeOwned {
    /// Logging level (e.g. `"info"`, `"debug"`) that `apply_logging_filter`
    /// installs into `RUST_LOG` if the env var is not already set.
    fn logging_level(&self) -> &str;
}
