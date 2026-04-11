use std::path::Path;

use anyhow::{Context, Result};

use rustml_gguf::GGUFFile;
use rustml_model::{
    LlmModel, ModelConfig, gguf_config_to_model_config, convert_tensors,
};
use rustml_tokenizer::{GgufTokenizer, Tokenizer};

use super::state::EmbeddingState;

/// Load an embedding model from a GGUF file.
pub fn load_gguf(path: &Path) -> Result<EmbeddingState> {
    log::info!("Loading GGUF: {}", path.display());
    let gguf = GGUFFile::parse_header(path)
        .with_context(|| format!("Failed to parse GGUF: {}", path.display()))?;
    log::info!("  GGUF v{}, {} tensors", gguf.version, gguf.tensor_infos.len());

    let gguf_config = gguf
        .to_model_config()
        .with_context(|| "Failed to extract model config from GGUF")?;
    let config = gguf_config_to_model_config(&gguf_config)
        .with_context(|| "Failed to convert GGUF config to model config")?;
    log::info!(
        "  arch={}, dim={}, layers={}, heads={}, vocab={}",
        gguf_config.architecture,
        config.dim, config.n_layers, config.n_heads, config.vocab_size
    );

    let tokenizer: Box<dyn Tokenizer + Send + Sync> = Box::new(
        GgufTokenizer::from_gguf(&gguf).with_context(|| "Failed to build tokenizer from GGUF")?,
    );

    let arch = gguf_config.architecture.as_str();
    let loaded_tensors = match arch {
        "nomic-bert" => gguf
            .load_and_remap_nomic_bert(path, config.n_layers)
            .with_context(|| "Failed to load/remap nomic-bert tensors")?,
        _ => gguf
            .load_and_remap(path, config.n_layers)
            .with_context(|| "Failed to load/remap tensors")?,
    };
    let tensors = convert_tensors(loaded_tensors);

    let model = match arch {
        "nomic-bert" => LlmModel::from_pretrained_nomic_bert(&config, tensors)
            .with_context(|| "Failed to build nomic-bert model")?,
        _ => LlmModel::from_pretrained_bert(&config, tensors)
            .with_context(|| "Failed to build model")?,
    };

    let (total_params, _) = model.parameter_count();
    log::info!("  Model ready: {:.1}M params", total_params as f64 / 1e6);

    let model_id = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "embedding-model".into());

    Ok(EmbeddingState {
        model,
        tokenizer,
        model_id,
    })
}
