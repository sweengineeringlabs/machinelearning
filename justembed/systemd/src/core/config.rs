//! Config loader: delegates to `swe_systemd::load_config` with the
//! embedding daemon's bundled default and `AppConfig` shape.

use swe_systemd::{LoadConfigResult, LoadedConfig, load_config};

use crate::api::config::AppConfig;

/// Bundled default config compiled into the binary. Lives at
/// `llminference/main/config/application.toml` relative to the repo root.
const BUNDLED_DEFAULT: &str = include_str!("../../../../llminference/main/config/application.toml");

/// Load + merge application.toml from XDG locations.
pub fn load() -> LoadConfigResult<LoadedConfig<AppConfig>> {
    load_config::<AppConfig>("llminference", BUNDLED_DEFAULT)
}
