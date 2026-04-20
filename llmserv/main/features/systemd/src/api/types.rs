/// Result of loading + merging a daemon config.
///
/// Generic over the typed config `T: DaemonConfig`.
pub struct LoadedConfig<T> {
    /// Typed view of the merged config.
    pub app: T,
    /// Merged TOML serialized back to a string. Subsystems that parse
    /// their own sections (e.g. a quantizer reading `[quantization]`)
    /// can use this without re-reading files from disk.
    pub merged_toml: String,
    /// Config source paths that contributed to the merge, in load
    /// order. Log these at startup so operators can see what's active.
    pub sources: Vec<String>,
}
