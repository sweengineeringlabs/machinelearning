//! Config loader — mirror of the daemon's, scoped to the embedding server.
//!
//! Load order (later wins via deep-merge):
//!   1. Bundled default (include_str!-ed at compile time)
//!   2. $XDG_CONFIG_DIRS/llmserv/application.toml (each)
//!   3. $XDG_CONFIG_HOME/llmserv/application.toml

use std::path::PathBuf;

use anyhow::{Context, Result};
use toml::Value;

use crate::api::config::AppConfig;

const BUNDLED_DEFAULT: &str = include_str!("../../../../../config/application.toml");

pub struct LoadedConfig {
    pub app: AppConfig,
    pub sources: Vec<String>,
}

pub fn load() -> Result<LoadedConfig> {
    let mut merged: Value = toml::from_str(BUNDLED_DEFAULT)
        .context("Failed to parse bundled default config")?;
    let mut sources = vec!["<bundled default>".to_string()];

    for system_dir in xdg_system_config_dirs() {
        let candidate = system_dir.join("llmserv").join("application.toml");
        if candidate.is_file() {
            overlay_file(&mut merged, &candidate, &mut sources)?;
        }
    }

    if let Some(user_dir) = dirs::config_dir() {
        let candidate = user_dir.join("llmserv").join("application.toml");
        if candidate.is_file() {
            overlay_file(&mut merged, &candidate, &mut sources)?;
        }
    }

    let app: AppConfig = merged
        .try_into()
        .context("Merged config does not match AppConfig schema")?;

    Ok(LoadedConfig { app, sources })
}

fn overlay_file(merged: &mut Value, path: &std::path::Path, sources: &mut Vec<String>) -> Result<()> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let overlay: Value = toml::from_str(&text)
        .with_context(|| format!("Failed to parse {}", path.display()))?;
    deep_merge(merged, overlay);
    sources.push(path.display().to_string());
    Ok(())
}

fn xdg_system_config_dirs() -> Vec<PathBuf> {
    if let Ok(raw) = std::env::var("XDG_CONFIG_DIRS") {
        let sep = if cfg!(windows) { ';' } else { ':' };
        raw.split(sep)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect()
    } else if cfg!(unix) {
        vec![PathBuf::from("/etc/xdg")]
    } else {
        Vec::new()
    }
}

fn deep_merge(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_tbl), Value::Table(overlay_tbl)) => {
            for (key, overlay_val) in overlay_tbl {
                match base_tbl.get_mut(&key) {
                    Some(base_val) => deep_merge(base_val, overlay_val),
                    None => {
                        base_tbl.insert(key, overlay_val);
                    }
                }
            }
        }
        (base_slot, overlay_val) => {
            *base_slot = overlay_val;
        }
    }
}
