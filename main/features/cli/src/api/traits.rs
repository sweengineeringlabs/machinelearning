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
/// the same crate. For stacked `-v`/`-q` flags, flatten
/// [`crate::VerbosityArgs`] into your `Cli` and call its `resolve()`.
///
/// # Exit codes
///
/// The default [`run`](Self::run) returns `anyhow::Result<()>`, so
/// `main()` — when used via `<MyCli as Cli>::run()` — follows the
/// Unix convention:
///
/// | exit | meaning                                                    |
/// |------|------------------------------------------------------------|
/// | `0`  | success (`Ok(())`)                                         |
/// | `1`  | runtime error (`Err(e)` from `dispatch`, bubbled by main)  |
/// | `2`  | usage error (clap parse failure, e.g. unknown subcommand)  |
///
/// Binaries that need other codes (e.g. `3` for "model not cached")
/// should handle that at the end of `dispatch()` with
/// `std::process::exit`.
///
/// # Color
///
/// clap emits colored help by default on TTY (feature `color` is on);
/// env_logger auto-detects TTY for colored log output. No extra wiring
/// needed. Users can disable with `NO_COLOR=1` (env_logger and clap
/// both honor it).
pub trait Cli: Parser {
    /// Route the parsed subcommand enum.
    fn dispatch(self) -> Result<()>;

    /// Default entry point. Parses argv and dispatches. Override only
    /// if your binary needs a non-standard startup (e.g. installing a
    /// panic hook before parse).
    fn run() -> Result<()> {
        Self::parse().dispatch()
    }
}
