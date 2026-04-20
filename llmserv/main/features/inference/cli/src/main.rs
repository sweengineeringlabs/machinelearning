mod cmd;

use anyhow::Result;
use clap::Parser;
use swe_cli::{Cli as SweCli, apply_logging_filter, init_env_logger};

/// llmc — developer CLI for llmserv.
///
/// Subcommands: infer, gguf, tokenizer, load. Model download lives in the
/// standalone `llmmodel` binary.
#[derive(Parser)]
#[command(name = "llmc", version, about)]
struct Cli {
    #[command(subcommand)]
    command: cmd::Command,
}

impl SweCli for Cli {
    fn dispatch(self) -> Result<()> {
        apply_logging_filter("info");
        init_env_logger();
        cmd::run(self.command)
    }
}

fn main() -> Result<()> {
    <Cli as SweCli>::run()
}
