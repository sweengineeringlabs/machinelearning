/// Install a panic hook that logs the panic at `error` level before
/// the default hook runs. Calling this after `env_logger` init means
/// any panic — in tokio tasks, rayon threads, or the main thread —
/// ends up in the log stream with the same formatting as other error
/// messages, instead of only on stderr.
///
/// Idempotent: calling more than once chains hooks; the latest takes
/// effect first, and the default hook still runs last.
pub fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Extract payload (the string passed to `panic!`).
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string payload>".into());

        match info.location() {
            Some(loc) => log::error!(
                "PANIC at {}:{}:{}: {}",
                loc.file(),
                loc.line(),
                loc.column(),
                payload
            ),
            None => log::error!("PANIC (unknown location): {}", payload),
        }

        // Chain to the previously-installed hook (e.g. the default
        // stderr formatter, or a pretty-panic handler).
        prev(info);
    }));
}
