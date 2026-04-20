use clap::{ArgAction, Args};

/// Global verbosity flags: `-v` increments, `-q` decrements.
///
/// Flatten into a binary's `Cli` struct with `#[command(flatten)]`:
///
/// ```ignore
/// #[derive(clap::Parser)]
/// struct Cli {
///     #[command(flatten)]
///     pub verbosity: swe_cli::VerbosityArgs,
///     // ... other flags, subcommand, etc.
/// }
/// ```
///
/// Then resolve the effective log level in `dispatch()`:
///
/// ```ignore
/// let level = self.verbosity.resolve("info");
/// swe_cli::apply_logging_filter(&level);
/// swe_cli::init_env_logger();
/// ```
///
/// Both flags are `global = true` so users can pass them anywhere:
/// `mycli -vv subcmd` and `mycli subcmd -vv` behave the same.
#[derive(Args, Debug, Clone, Default)]
pub struct VerbosityArgs {
    /// Increase verbosity. `-v` = debug, `-vv` = trace. Stacks with `-q`.
    #[arg(short = 'v', long = "verbose", global = true, action = ArgAction::Count)]
    pub verbose: u8,

    /// Decrease verbosity. `-q` = warn, `-qq` = error, `-qqq` = off. Stacks with `-v`.
    #[arg(short = 'q', long = "quiet", global = true, action = ArgAction::Count)]
    pub quiet: u8,
}

impl VerbosityArgs {
    /// Resolve the effective log level for `RUST_LOG`, given a default.
    ///
    /// Net verbosity (`verbose - quiet`) shifts the level:
    /// - `+2` or more → `"trace"`
    /// - `+1`         → `"debug"`
    /// - `0`          → `default`
    /// - `-1`         → `"warn"`
    /// - `-2`         → `"error"`
    /// - `-3` or less → `"off"`
    pub fn resolve(&self, default: &str) -> String {
        let net = self.verbose as i32 - self.quiet as i32;
        match net {
            2.. => "trace".into(),
            1 => "debug".into(),
            0 => default.into(),
            -1 => "warn".into(),
            -2 => "error".into(),
            _ => "off".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_zero_returns_default() {
        let v = VerbosityArgs::default();
        assert_eq!(v.resolve("info"), "info");
    }

    #[test]
    fn resolve_positive_steps_up() {
        let v = VerbosityArgs { verbose: 1, quiet: 0 };
        assert_eq!(v.resolve("info"), "debug");
        let v = VerbosityArgs { verbose: 2, quiet: 0 };
        assert_eq!(v.resolve("info"), "trace");
        let v = VerbosityArgs { verbose: 5, quiet: 0 };
        assert_eq!(v.resolve("info"), "trace");
    }

    #[test]
    fn resolve_negative_steps_down() {
        let v = VerbosityArgs { verbose: 0, quiet: 1 };
        assert_eq!(v.resolve("info"), "warn");
        let v = VerbosityArgs { verbose: 0, quiet: 2 };
        assert_eq!(v.resolve("info"), "error");
        let v = VerbosityArgs { verbose: 0, quiet: 5 };
        assert_eq!(v.resolve("info"), "off");
    }

    #[test]
    fn resolve_cancels_out() {
        let v = VerbosityArgs { verbose: 2, quiet: 2 };
        assert_eq!(v.resolve("info"), "info");
    }
}
