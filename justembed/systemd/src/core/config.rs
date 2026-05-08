//! Config loader: delegates to `swe_systemd::load_config` with the
//! embedding daemon's bundled default and `AppConfig` shape.

use swe_systemd::{LoadConfigResult, LoadedConfig, load_config};

use crate::api::config::AppConfig;

const BUNDLED_DEFAULT: &str = include_str!("../../../../llmserve/main/config/application.toml");

pub fn load() -> LoadConfigResult<LoadedConfig<AppConfig>> {
    load_config::<AppConfig>("llminference", BUNDLED_DEFAULT)
}
