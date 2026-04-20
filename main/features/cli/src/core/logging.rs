/// Install `RUST_LOG=<level>` into the process env if it isn't already
/// set. Call this BEFORE [`init_env_logger`] — after init, the
/// env_logger reader has already snapshotted the env.
///
/// # Safety
///
/// `set_var` is `unsafe` because the env is process-global and shared
/// between threads. This helper must run single-threaded during startup,
/// before any tokio/rayon threads spawn.
pub fn apply_logging_filter(level: &str) {
    if std::env::var_os("RUST_LOG").is_none() {
        // SAFETY: caller promises single-threaded startup invocation.
        unsafe {
            std::env::set_var("RUST_LOG", level);
        }
    }
}

/// Initialize env_logger with the current `RUST_LOG`. Idempotent-ish:
/// if another crate has already called `env_logger::init`, this is a
/// no-op (env_logger internally panics on second init — we swallow
/// that into a silent fallback).
pub fn init_env_logger() {
    // `try_init` returns Err if a global logger is already set; that's
    // fine for a helper meant to be called from `main()`.
    let _ = env_logger::try_init();
}
