//! Generic XDG config loader with deep-merge.
//!
//! Load order (later overrides earlier):
//!   1. Bundled default (caller passes `include_str!`-ed TOML text)
//!   2. `$XDG_CONFIG_DIRS/<app>/application.toml` (each, in order)
//!   3. `$XDG_CONFIG_HOME/<app>/application.toml`
//!
//! Scope (strict):
//!   - Find config files via XDG
//!   - Deep-merge TOML tables (override wins at leaf level)
//!   - Deserialize into caller's `T: DaemonConfig`
//!   - Expose the merged raw TOML string for subsystems that still
//!     want to parse their own section
//!
//! Nothing else: no validation beyond type checks, no templating,
//! no hot reload, no env-var injection.

use std::path::PathBuf;

use toml::Value;

use crate::api::error::{LoadConfigError, LoadConfigResult};
use crate::api::traits::DaemonConfig;
use crate::api::types::LoadedConfig;

/// Load + merge an `application.toml` for a daemon.
///
/// * `app_dir_name` — the subdirectory name under XDG config dirs
///   (e.g. `"llmserv"` → `$XDG_CONFIG_HOME/llmserv/application.toml`).
/// * `bundled_default` — TOML text compiled into the binary, usually
///   via `include_str!`.
pub fn load_config<T: DaemonConfig>(
    app_dir_name: &str,
    bundled_default: &str,
) -> LoadConfigResult<LoadedConfig<T>> {
    let mut merged: Value = toml::from_str(bundled_default)
        .map_err(|e| LoadConfigError::ParseDefault(e.to_string()))?;
    let mut sources = vec!["<bundled default>".to_string()];

    // XDG system dirs (lower precedence than user dir).
    for system_dir in xdg_system_config_dirs() {
        let candidate = system_dir.join(app_dir_name).join("application.toml");
        if candidate.is_file() {
            overlay_file(&mut merged, &candidate, &mut sources)?;
        }
    }

    // XDG user dir (highest precedence).
    if let Some(user_dir) = dirs::config_dir() {
        let candidate = user_dir.join(app_dir_name).join("application.toml");
        if candidate.is_file() {
            overlay_file(&mut merged, &candidate, &mut sources)?;
        }
    }

    let merged_toml = toml::to_string(&merged)
        .map_err(|e| LoadConfigError::Serialize(e.to_string()))?;

    let app: T = merged
        .try_into()
        .map_err(|e: toml::de::Error| LoadConfigError::Schema(e.to_string()))?;

    Ok(LoadedConfig {
        app,
        merged_toml,
        sources,
    })
}

fn overlay_file(
    merged: &mut Value,
    path: &std::path::Path,
    sources: &mut Vec<String>,
) -> LoadConfigResult<()> {
    let text = std::fs::read_to_string(path).map_err(|e| LoadConfigError::ReadFile {
        path: path.display().to_string(),
        source: e,
    })?;
    let overlay: Value = toml::from_str(&text).map_err(|e| LoadConfigError::ParseFile {
        path: path.display().to_string(),
        source: e,
    })?;
    deep_merge(merged, overlay);
    sources.push(path.display().to_string());
    Ok(())
}

/// `$XDG_CONFIG_DIRS` (colon-separated on Unix, semicolon on Windows).
/// Returns in-order; first entry has highest precedence among system dirs.
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

/// Recursively merge `overlay` into `base`. Tables are merged key-by-key;
/// leaves (non-table values) are replaced wholesale by the overlay.
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

#[cfg(test)]
mod tests {
    use super::*;
    use toml::toml;

    #[test]
    fn test_deep_merge_overwrites_leaf() {
        let mut base: Value = toml! { x = 1 y = 2 }.into();
        let overlay: Value = toml! { x = 99 }.into();
        deep_merge(&mut base, overlay);
        let tbl = base.as_table().unwrap();
        assert_eq!(tbl.get("x").unwrap().as_integer(), Some(99));
        assert_eq!(tbl.get("y").unwrap().as_integer(), Some(2));
    }

    #[test]
    fn test_deep_merge_recurses_into_tables() {
        let mut base: Value = toml! {
            [server]
            host = "127.0.0.1"
            port = 8080
        }
        .into();
        let overlay: Value = toml! {
            [server]
            port = 9090
        }
        .into();
        deep_merge(&mut base, overlay);
        let server = base.get("server").unwrap().as_table().unwrap();
        assert_eq!(server.get("host").unwrap().as_str(), Some("127.0.0.1"));
        assert_eq!(server.get("port").unwrap().as_integer(), Some(9090));
    }

    #[test]
    fn test_deep_merge_adds_new_keys() {
        let mut base: Value = toml! { x = 1 }.into();
        let overlay: Value = toml! { y = 2 }.into();
        deep_merge(&mut base, overlay);
        let tbl = base.as_table().unwrap();
        assert_eq!(tbl.get("x").unwrap().as_integer(), Some(1));
        assert_eq!(tbl.get("y").unwrap().as_integer(), Some(2));
    }
}
