use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};

use rustml_gguf::GGUFFile;
use rustml_hub::HubApi;
use rustml_model::{
    LlmModel, ModelConfig, ModelRegistry, OptProfile, convert_tensors,
    gguf_config_to_model_config,
};
use rustml_tokenizer::{BpeTokenizer, GgufTokenizer, HFTokenizer, Tokenizer};
use rustml_quantizer::Quantizer;

use super::state::ModelBundle;

/// Create the model registry with all supported architectures.
fn create_registry() -> ModelRegistry {
    let mut reg = ModelRegistry::new();
    reg.register("llama", Box::new(rustml_arch_llama::LlamaBuilder));
    reg.register("mistral", Box::new(rustml_arch_llama::LlamaBuilder));
    reg.register("qwen2", Box::new(rustml_arch_llama::LlamaBuilder));
    reg.register("phi3", Box::new(rustml_arch_llama::LlamaBuilder));
    reg.register("gpt2", Box::new(rustml_arch_gpt2::Gpt2Builder));
    reg.register("", Box::new(rustml_arch_gpt2::Gpt2Builder)); // default
    reg.register("falcon", Box::new(rustml_arch_falcon::FalconBuilder));
    reg.register("mixtral", Box::new(rustml_arch_mixtral::MixtralBuilder));
    reg.register("gemma2", Box::new(rustml_arch_llama::LlamaBuilder)); // Gemma2 uses llama loader
    reg.register("gemma3", Box::new(rustml_arch_gemma3::Gemma3Builder));
    reg.register("gemma3_text", Box::new(rustml_arch_gemma3::Gemma3Builder));
    reg.register("gemma4", Box::new(rustml_arch_gemma4::Gemma4Builder));
    reg.register("gemma4_text", Box::new(rustml_arch_gemma4::Gemma4Builder));
    reg.register("bert", Box::new(rustml_arch_bert::BertBuilder));
    reg.register("nomic-bert", Box::new(rustml_arch_nomic_bert::NomicBertBuilder));
    reg
}

/// Load a model from a GGUF file on disk.
pub fn load_gguf(path: &Path, profile: OptProfile) -> Result<ModelBundle> {
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
        config.dim,
        config.n_layers,
        config.n_heads,
        config.vocab_size
    );

    let tokenizer: Box<dyn Tokenizer + Send + Sync> = Box::new(
        GgufTokenizer::from_gguf(&gguf).with_context(|| "Failed to build tokenizer from GGUF")?,
    );

    let arch = gguf_config.architecture.as_str();
    let loaded_tensors = match arch {
        "gemma3" => gguf
            .load_and_remap_gemma3(path, config.n_layers)
            .with_context(|| "Failed to load/remap gemma3 tensors")?,
        "nomic-bert" => gguf
            .load_and_remap_nomic_bert(path, config.n_layers)
            .with_context(|| "Failed to load/remap nomic-bert tensors")?,
        _ => gguf
            .load_and_remap(path, config.n_layers)
            .with_context(|| "Failed to load/remap tensors")?,
    };
    let tensors = convert_tensors(loaded_tensors);

    let mut config = config;
    config.architecture = gguf_config.architecture.clone();
    let registry = create_registry();
    let mut model = registry.build_model(&config, tensors)
        .with_context(|| format!("Failed to build {} model", config.architecture))?;
    model.set_optimization_profile(profile);

    use rustml_quantizer::Fuser;
    let qkv_fuser = rustml_quantizer::QkvFuser;
    let fused_qkv = qkv_fuser.fuse(&mut model);
    if fused_qkv > 0 {
        log::info!("  {}: {} triples", qkv_fuser.describe(), fused_qkv);
    }

    warmup(&mut model);

    let (total_params, _) = model.parameter_count();
    log::info!("  Model ready: {:.1}M params", total_params as f64 / 1e6);

    let model_id = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "gguf-model".into());

    Ok(ModelBundle {
        model,
        tokenizer,
        model_id,
        chat_template: config.chat_template.clone(),
        eos_token_id: config.eos_token_id,
        bos_token_id: config.bos_token_id,
        profile,
    })
}

/// Load a model from HuggingFace SafeTensors.
pub fn load_safetensors(model_id: &str, profile: OptProfile) -> Result<ModelBundle> {
    let hub = HubApi::new();
    let bundle = match hub.get_cached(model_id) {
        Some(b) => {
            log::info!("Using cached model: {}", model_id);
            b
        }
        None => {
            log::info!("Downloading model: {}", model_id);
            hub.download_model_sync(model_id)
                .with_context(|| format!("Failed to download model: {}", model_id))?
        }
    };

    let json_config = bundle
        .load_config_sync()
        .with_context(|| "Failed to load config.json")?;
    let model_type = json_config["model_type"].as_str().unwrap_or("").to_string();
    let config = ModelConfig::from_json_value(&json_config)
        .with_context(|| "Failed to parse model config")?;

    log::info!(
        "  Config: arch={}, dim={}, layers={}, heads={}, vocab={}",
        if model_type.is_empty() { "gpt2" } else { &model_type },
        config.dim,
        config.n_layers,
        config.n_heads,
        config.vocab_size
    );

    let weights = bundle
        .load_tensors()
        .with_context(|| "Failed to load SafeTensors weights")?;
    log::info!("  {} tensors loaded", weights.len());

    let mut config = config;
    config.architecture = model_type.clone();
    let registry = create_registry();
    let mut model = registry.build_model(&config, weights)
        .with_context(|| format!("Failed to build {} model", if model_type.is_empty() { "gpt2" } else { &model_type }))?;
    model.set_optimization_profile(profile);

    // Post-construction optimization via pluggable providers
    if !model.output.is_quantized() {
        let quantizer = rustml_quantizer::ConfigQuantizer::from_toml(std::path::Path::new("quantization.toml"));
        match quantizer.quantize(&mut model) {
            Ok(n) if n > 0 => log::info!("  Quantized {} linear layers ({})", n, quantizer.describe()),
            Ok(_) => {}
            Err(e) => log::warn!("  Weight quantization failed: {}", e),
        }
    }

    use rustml_quantizer::Fuser;
    let gate_up_fuser = rustml_quantizer::GateUpFuser;
    let fused = gate_up_fuser.fuse(&mut model);
    if fused > 0 {
        log::info!("  {}: {} pairs", gate_up_fuser.describe(), fused);
    }
    let qkv_fuser = rustml_quantizer::QkvFuser;
    let fused_qkv = qkv_fuser.fuse(&mut model);
    if fused_qkv > 0 {
        log::info!("  {}: {} triples", qkv_fuser.describe(), fused_qkv);
    }

    warmup(&mut model);

    let (total_params, _) = model.parameter_count();
    log::info!("  Model ready: {:.1}M params", total_params as f64 / 1e6);

    let tokenizer_json = bundle.tokenizer_json_path();
    let tokenizer: Box<dyn Tokenizer + Send + Sync> = if tokenizer_json.exists() {
        let hf = HFTokenizer::from_file(&tokenizer_json)
            .with_context(|| "Failed to load HFTokenizer from tokenizer.json")?;
        log::info!("  Tokenizer: {} tokens (tokenizer.json)", hf.vocab_size());
        Box::new(hf)
    } else {
        let bpe = BpeTokenizer::from_files(bundle.vocab_path(), bundle.merges_path())
            .with_context(|| "Failed to load BPE tokenizer")?;
        log::info!("  Tokenizer: {} tokens (BPE)", bpe.vocab_size());
        Box::new(bpe)
    };

    let eos = config.eos_token_id.or_else(|| {
        if model_type.is_empty() || model_type == "gpt2" {
            Some(BpeTokenizer::GPT2_EOS_TOKEN_ID)
        } else {
            None
        }
    });

    Ok(ModelBundle {
        model,
        tokenizer,
        model_id: model_id.to_string(),
        chat_template: config.chat_template.clone(),
        eos_token_id: eos,
        bos_token_id: config.bos_token_id,
        profile,
    })
}

fn warmup(model: &mut LlmModel) {
    let start = Instant::now();
    if let Err(e) = model.warmup_decode() {
        log::warn!("  Decode warmup failed: {}", e);
    }
    log::info!("  Warmup: {:.0}ms", start.elapsed().as_secs_f64() * 1000.0);
}
