pub mod gguf;
pub mod hub;
pub mod infer;
pub mod load;
pub mod tokenizer;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Command {
    /// Inspect GGUF model files (info, metadata, tensors, verify).
    #[command(subcommand)]
    Gguf(gguf::GgufCommand),

    /// Encode, decode, and inspect tokenizer vocabularies.
    Tokenizer(tokenizer::TokenizerArgs),

    /// Download, cache, and inspect HuggingFace models.
    Hub(hub::HubArgs),

    /// Run text generation on a GGUF model.
    Infer(infer::InferArgs),

    /// HTTP load test an endpoint — fires N requests at C concurrency,
    /// reports latency percentiles and status distribution.
    Load(load::LoadArgs),
}

pub fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Gguf(cmd) => gguf::run(cmd),
        Command::Tokenizer(args) => tokenizer::run(args),
        Command::Hub(args) => hub::run(args),
        Command::Infer(args) => infer::run(args),
        Command::Load(args) => load::run(args),
    }
}
