//! Config loader: XDG search path + deep-merge + typed deserialization.
//!
//! Load order (later entries override earlier):
//!   1. Bundled default (embedded at compile time — always present)
//!   2. $XDG_CONFIG_DIRS/llmserv/application.toml (system, each dir)
//!   3. $XDG_CONFIG_HOME/llmserv/application.toml (user)
//!
//! Scope of this module (strict):
//!   - Find config files via XDG
//!   - Deep-merge TOML tables (override wins at leaf level)
//!   - Deserialize into AppConfig
//!   - Expose the merged raw TOML string for subsystems that still want
//!     to parse it themselves (the quantizer)
//!
//! Nothing else: no validation beyond type checks, no templating, no
//! hot reload, no env-var injection.

use std::path::PathBuf;

use anyhow::{Context, Result};
use toml::Value;

use crate::api::config::AppConfig;

/// Bundled default config compiled into the binary. Lives at
/// `llmserv/main/config/application.toml` relative to the repo root.
const BUNDLED_DEFAULT: &str = include_str!("../../../../../../config/application.toml");

/// Result of loading and merging the application config.
pub struct LoadedConfig {
    /// Typed view of the merged config.
    pub app: AppConfig,
    /// Merged TOML serialized back to a string. Subsystems that need
    /// their own section (e.g., the quantizer for `[quantization]`) can
    /// parse this without re-reading files.
    pub merged_toml: String,
    /// Paths that contributed to the merge, in load order. Logged at
    /// startup so operators can see what's active.
    pub sources: Vec<String>,
}

/// Load + merge application.toml from XDG locations.
pub fn load() -> Result<LoadedConfig> {
    let mut merged: Value = toml::from_str(BUNDLED_DEFAULT)
        .context("Failed to parse bundled default config")?;
    let mut sources = vec!["<bundled default>".to_string()];

    // XDG system dirs (lowest precedence of the file-based overlays).
    for system_dir in xdg_system_config_dirs() {
        let candidate = system_dir.join("llmserv").join("application.toml");
        if candidate.is_file() {
            overlay_file(&mut merged, &candidate, &mut sources)?;
        }
    }

    // XDG user dir (highest precedence).
    if let Some(user_dir) = dirs::config_dir() {
        let candidate = user_dir.join("llmserv").join("application.toml");
        if candidate.is_file() {
            overlay_file(&mut merged, &candidate, &mut sources)?;
        }
    }

    // Serialize merged back to string for subsystems that parse their own
    // section (the quantizer wants the raw [quantization] table text).
    let merged_toml = toml::to_string(&merged)
        .context("Failed to serialize merged config")?;

    // Typed deserialization.
    let app: AppConfig = merged
        .try_into()
        .context("Merged config does not match AppConfig schema")?;

    Ok(LoadedConfig {
        app,
        merged_toml,
        sources,
    })
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

/// `$XDG_CONFIG_DIRS` (colon-separated on Unix, semicolon on Windows).
/// Returns in-order; first entry has highest precedence among system dirs.
fn xdg_system_config_dirs() -> Vec<PathBuf> {
    if let Ok(raw) = std::env::var("XDG_CONFIG_DIRS") {
        let sep = if cfg!(windows) { ';' } else { ':' };
        raw.split(sep)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect()
    } else {
        // Unix default; no analogous Windows system dir via `dirs`, so empty there.
        if cfg!(unix) {
            vec![PathBuf::from("/etc/xdg")]
        } else {
            Vec::new()
        }
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
