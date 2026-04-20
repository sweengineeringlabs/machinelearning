/// Install `RUST_LOG=<level>` into the process env if it isn't already
/// set. Call this BEFORE `env_logger::init()` — after init, the env_logger
/// reader has already snapshotted the env.
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
