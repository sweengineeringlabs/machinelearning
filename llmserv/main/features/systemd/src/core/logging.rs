//! Re-export of [`swe_cli::apply_logging_filter`]. Canonical
//! implementation lives in `swe-cli` so CLIs and daemons share one
//! logging-filter helper.

pub use swe_cli::apply_logging_filter;
