use anyhow::Result;
use clap::Parser;

/// Entry-point trait for a CLI binary.
///
/// The binary's `Cli` struct derives `clap::Parser` and implements
/// [`Cli::dispatch`] to route its subcommand enum. `main()` becomes one
/// line: `<MyCli as Cli>::run()`.
///
/// Logging setup is intentionally left to the implementor's
/// [`dispatch`](Self::dispatch) body, because config-driven daemons
/// need to load config before the log level is known, while static-
/// level CLIs want the default immediately. Both patterns can reuse
/// [`crate::apply_logging_filter`] + [`crate::init_env_logger`] from
/// the same crate.
pub trait Cli: Parser {
    /// Route the parsed subcommand enum.
    fn dispatch(self) -> Result<()>;

    /// Default entry point. Parses argv and dispatches. Override only
    /// if your binary needs a non-standard startup (e.g. panic hook
    /// installation before parse).
    fn run() -> Result<()> {
        Self::parse().dispatch()
    }
}
