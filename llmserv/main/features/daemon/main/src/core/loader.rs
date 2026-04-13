use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};

use rustml_gguf::GGUFFile;
use rustml_hub::HubApi;
use rustml_model::{
    LlmModel, ModelConfig, ModelBuilderRegistry, OptProfile, convert_tensors,
    gguf_config_to_model_config,
};
use rustml_tokenizer::{BpeTokenizer, GgufTokenizer, HFTokenizer, Tokenizer};
use rustml_quantizer::Quantizer;

use llmbackend::{Model, ModelBackendLoader, ModelSource, ModelSpec};

use super::state::DefaultModel;

/// Native Rust backend loader. Dispatches on `[model].source` to the
/// SafeTensors or GGUF path, both of which produce a `DefaultModel`.
pub struct NativeRustBackendLoader;

impl ModelBackendLoader for NativeRustBackendLoader {
    fn name(&self) -> &'static str {
        "native_rust"
    }

    fn load(
        &self,
        spec: &ModelSpec,
        profile: OptProfile,
        merged_toml: &str,
    ) -> Result<Box<dyn Model>> {
        match spec.source {
            ModelSource::Safetensors => {
                let id = spec.id.as_deref().ok_or_else(|| {
                    anyhow!("[model].source = \"safetensors\" requires [model].id")
                })?;
                Ok(Box::new(load_safetensors(id, profile, merged_toml)?))
            }
            ModelSource::Gguf => {
                let path_str = spec.path.as_deref().ok_or_else(|| {
                    anyhow!("[model].source = \"gguf\" requires [model].path")
                })?;
                Ok(Box::new(load_gguf(&PathBuf::from(path_str), profile)?))
            }
        }
    }
}

/// Create the model registry with all supported architectures.
fn create_registry() -> ModelBuilderRegistry {
    let mut reg = ModelBuilderRegistry::new();
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
pub fn load_gguf(path: &Path, profile: OptProfile) -> Result<DefaultModel> {
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

    Ok(DefaultModel {
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
///
/// `quantization_toml` is the merged TOML text whose `[quantization]` section
/// configures the runtime weight quantizer. Callers produce it from the
/// application config loader.
pub fn load_safetensors(model_id: &str, profile: OptProfile, quantization_toml: &str) -> Result<DefaultModel> {
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

    let json_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(bundle.config_path())
            .with_context(|| "Failed to read config.json")?,
    )
    .with_context(|| "Failed to parse config.json")?;
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

    let weights = rustml_hub::load_safetensors(&bundle.weights_path())
        .with_context(|| "Failed to load SafeTensors weights")?;
    log::info!("  {} tensors loaded", weights.len());

    let mut config = config;
    config.architecture = model_type.clone();

    // HuggingFace puts chat_template in tokenizer_config.json, not
    // config.json. Pull it in if present so the generator can apply
    // the right template at decode time. Without this, instruct-
    // tuned models receive un-templated prompts and emit garbage
    // (looks like base-model continuation) — see investigation in
    // chat completion deploy of gemma-3-1b-it on 2026-04-13.
    if config.chat_template.is_none() {
        let tc_path = bundle.tokenizer_config_path();
        if tc_path.exists() {
            match std::fs::read_to_string(&tc_path) {
                Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(tc) => {
                        if let Some(tmpl) = tc["chat_template"].as_str() {
                            log::info!(
                                "  Chat template: {} chars from tokenizer_config.json",
                                tmpl.len()
                            );
                            config.chat_template = Some(tmpl.to_string());
                        }
                    }
                    Err(e) => log::warn!("  tokenizer_config.json parse failed: {}", e),
                },
                Err(e) => log::warn!("  tokenizer_config.json read failed: {}", e),
            }
        }
    }

    // Last-resort fallback: derive a marker from the architecture
    // when no template was discoverable. The generator's
    // `build_multi_turn_segments` dispatches on substring matches
    // (e.g. `template.contains("<start_of_turn>")`), so passing the
    // marker alone is enough to route to the right template branch.
    // Older HF caches downloaded before tokenizer_config.json was
    // requested don't include it, so this lets existing cached
    // models still serve correctly without re-downloading.
    if config.chat_template.is_none() {
        let marker = match config.architecture.as_str() {
            "gemma3" | "gemma3_text" => Some("<start_of_turn>"),
            "gemma4" | "gemma4_text" => Some("<|turn>"),
            "qwen2" => Some("<|im_start|>"),
            "llama" | "mistral" => Some("[INST]"),
            _ => None,
        };
        if let Some(m) = marker {
            log::info!(
                "  Chat template: derived marker '{}' for architecture '{}' (no tokenizer_config.json in cache)",
                m, config.architecture
            );
            config.chat_template = Some(m.to_string());
        }
    }

    let registry = create_registry();
    let mut model = registry.build_model(&config, weights)
        .with_context(|| format!("Failed to build {} model", if model_type.is_empty() { "gpt2" } else { &model_type }))?;
    model.set_optimization_profile(profile);

    // Post-construction optimization via pluggable providers
    if !model.output.is_quantized() {
        let quantizer = rustml_quantizer::ConfigQuantizer::from_toml_str(quantization_toml);
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

    Ok(DefaultModel {
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
