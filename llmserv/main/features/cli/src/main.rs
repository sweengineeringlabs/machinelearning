mod cmd;

use anyhow::Result;
use clap::Parser;

/// llmc — developer CLI for llmserv.
///
/// Subcommands: infer, hub, gguf, tokenizer, load.
#[derive(Parser)]
#[command(name = "llmc", version, about)]
struct Cli {
    #[command(subcommand)]
    command: cmd::Command,
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    cmd::run(cli.command)
}
