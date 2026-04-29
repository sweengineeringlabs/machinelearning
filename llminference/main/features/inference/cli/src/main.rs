mod cmd;

use anyhow::Result;
use clap::Parser;
use swe_cli::{
    Cli as SweCli, VerbosityArgs, apply_logging_filter, init_env_logger, install_panic_hook,
};

/// llmc — developer CLI for llminference.
///
/// Subcommands: infer, gguf, tokenizer, load. Model download lives in the
/// standalone `llmmodel` binary.
#[derive(Parser)]
#[command(name = "llmc", version, about)]
struct Cli {
    #[command(flatten)]
    verbosity: VerbosityArgs,

    #[command(subcommand)]
    command: cmd::Command,
}

impl SweCli for Cli {
    fn dispatch(self) -> Result<()> {
        apply_logging_filter(&self.verbosity.resolve("info"));
        init_env_logger();
        install_panic_hook();
        cmd::run(self.command)
    }
}

fn main() -> Result<()> {
    <Cli as SweCli>::run()
}
