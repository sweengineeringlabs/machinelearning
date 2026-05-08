use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use swe_cli::{
    Cli as SweCli, VerbosityArgs, apply_logging_filter, init_env_logger, install_panic_hook,
};
use swe_llmmodel_download::{Download, HuggingFaceDownload};

/// llmmodel — download, cache, and inspect HuggingFace models.
#[derive(Parser)]
#[command(name = "llmmodel", version, about)]
struct Cli {
    /// Override the default cache directory.
    #[arg(long, global = true)]
    cache_dir: Option<PathBuf>,

    /// HuggingFace API token for private models.
    #[arg(long, global = true)]
    token: Option<String>,

    #[command(flatten)]
    verbosity: VerbosityArgs,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download a model from HuggingFace Hub.
    Download {
        /// Model identifier (e.g. "openai-community/gpt2").
        model_id: String,

        /// Download a GGUF file instead of SafeTensors.
        /// Provide the filename within the repo (e.g. "model-Q4_0.gguf").
        #[arg(long)]
        gguf: Option<String>,
    },

    /// List cached models in the local cache directory.
    List,

    /// Show config.json for a cached model.
    Info {
        /// Model identifier (e.g. "openai-community/gpt2").
        model_id: String,
    },
}

fn build_downloader(cli: &Cli) -> HuggingFaceDownload {
    let mut dl = match &cli.cache_dir {
        Some(dir) => HuggingFaceDownload::with_cache_dir(dir),
        None => HuggingFaceDownload::new(),
    };
    if let Some(ref token) = cli.token {
        dl = dl.with_token(token);
    }
    dl
}

fn main() -> Result<()> {
    <Cli as SweCli>::run()
}

impl SweCli for Cli {
    fn dispatch(self) -> Result<()> {
        apply_logging_filter(&self.verbosity.resolve("info"));
        init_env_logger();
        install_panic_hook();

        let dl = build_downloader(&self);

        match &self.command {
        Command::Download { model_id, gguf } => {
            if let Some(filename) = gguf {
                eprintln!("Downloading GGUF {model_id} / {filename} ...");
                let bundle = dl
                    .download_gguf(model_id, filename)
                    .with_context(|| format!("Failed to download GGUF: {model_id}/{filename}"))?;
                println!("{}", bundle.gguf_path.display());
            } else {
                eprintln!("Downloading model {model_id} ...");
                let bundle = dl
                    .download_model(model_id)
                    .with_context(|| format!("Failed to download model: {model_id}"))?;
                println!("{}", bundle.model_dir.display());
            }
        }

        Command::List => {
            let cache = dl.cache_dir();
            if !cache.exists() {
                eprintln!("Cache directory does not exist: {}", cache.display());
                return Ok(());
            }
            let mut found = false;
            let entries = std::fs::read_dir(cache)
                .with_context(|| format!("Failed to read cache dir: {}", cache.display()))?;
            for entry in entries {
                let entry = entry?;
                if entry.path().is_dir() {
                    let dir_name = entry.file_name();
                    let dir_str = dir_name.to_string_lossy();
                    let model_id = dir_str.replacen("--", "/", 1);
                    println!("{model_id}");
                    found = true;
                }
            }
            if !found {
                eprintln!("No cached models found.");
            }
        }

        Command::Info { model_id } => {
            let bundle = dl
                .get_cached(model_id)
                .with_context(|| format!("Model not cached: {model_id}"))?;
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(bundle.config_path())
                    .with_context(|| format!("Failed to read config for {model_id}"))?,
            )
            .with_context(|| format!("Failed to parse config for {model_id}"))?;
            println!("{}", serde_json::to_string_pretty(&config)?);
        }
    }

        Ok(())
    }
}
