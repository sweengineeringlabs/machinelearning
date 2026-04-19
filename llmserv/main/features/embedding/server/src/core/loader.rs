use std::path::Path;

use anyhow::{Context, Result};

use swe_llmmodel_gguf::GGUFFile;
use swe_llmmodel_model::{
    ModelBuilderRegistry, gguf_config_to_model_config,
};
use rustml_tokenizer::{GgufTokenizer, Tokenizer};

use super::state::EmbeddingState;

/// Create the model registry with embedding-relevant architectures.
fn create_registry() -> ModelBuilderRegistry {
    let mut reg = ModelBuilderRegistry::new();
    reg.register("nomic-bert", Box::new(rustml_arch_nomic_bert::NomicBertBuilder));
    // Add more embedding architectures here as needed:
    // reg.register("bert", Box::new(rustml_arch_bert::BertBuilder));
    reg
}

/// Load an embedding model from a GGUF file.
pub fn load_gguf(path: &Path) -> Result<EmbeddingState> {
    log::info!("Loading GGUF: {}", path.display());
    let gguf = GGUFFile::parse_header(path)
        .with_context(|| format!("Failed to parse GGUF: {}", path.display()))?;
    log::info!("  GGUF v{}, {} tensors", gguf.version, gguf.tensor_infos.len());

    let gguf_config = gguf
        .to_model_config()
        .with_context(|| "Failed to extract model config from GGUF")?;
    let mut config = gguf_config_to_model_config(&gguf_config)
        .with_context(|| "Failed to convert GGUF config to model config")?;

    // Set architecture from GGUF metadata for registry dispatch
    config.architecture = gguf_config.architecture.clone();

    log::info!(
        "  arch={}, dim={}, layers={}, heads={}, vocab={}",
        config.architecture,
        config.dim, config.n_layers, config.n_heads, config.vocab_size
    );

    let tokenizer: Box<dyn Tokenizer + Send + Sync> = Box::new(
        GgufTokenizer::from_gguf(&gguf).with_context(|| "Failed to build tokenizer from GGUF")?,
    );

    let arch = gguf_config.architecture.as_str();
    let tensors = match arch {
        "nomic-bert" => gguf
            .load_and_remap_nomic_bert_mmap(path, config.n_layers)
            .with_context(|| "Failed to mmap/remap nomic-bert tensors")?,
        _ => gguf
            .load_and_remap_mmap(path, config.n_layers)
            .with_context(|| "Failed to mmap/remap tensors")?,
    };

    // Build model via registry — config.architecture drives builder selection
    let registry = create_registry();
    let model = registry.build_model(&config, tensors)
        .with_context(|| format!("Failed to build {} model", config.architecture))?;

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
