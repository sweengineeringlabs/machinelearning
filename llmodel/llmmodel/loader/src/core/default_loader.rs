use std::path::Path;

use swe_llmmodel_download::{Download, HuggingFaceDownload};
use swe_llmmodel_gguf::GGUFFile;
use swe_llmmodel_io::{LoadTensors, SafeTensorsStore};
use swe_llmmodel_model::{
    ModelBuilderRegistry, ModelConfig, OptProfile, convert_tensors, gguf_config_to_model_config,
};
use swe_llmmodel_quantizer::{ConfigQuantizer, Fuser, GateUpFuser, QkvFuser, Quantizer};
use swe_llmmodel_tokenizer::{BpeTokenizer, GgufTokenizer, HFTokenizer, Tokenizer};

use crate::api::error::{LoaderError, LoaderResult};
use crate::api::traits::LoadModel;
use crate::api::types::LoadedModel;
use crate::core::registry::create_registry;
use crate::core::warmup::warmup;

/// Default end-to-end model loader.
///
/// Builds its architecture registry lazily on construction via
/// [`create_registry`]; callers that need a custom registry can
/// write their own [`LoadModel`] impl.
pub struct DefaultLoader {
    registry: ModelBuilderRegistry,
}

impl Default for DefaultLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultLoader {
    pub fn new() -> Self {
        Self { registry: create_registry() }
    }

    /// Override the registry (e.g. a test registry with a single builder).
    pub fn with_registry(registry: ModelBuilderRegistry) -> Self {
        Self { registry }
    }
}

impl LoadModel for DefaultLoader {
    fn load_safetensors(
        &self,
        model_id: &str,
        profile: OptProfile,
        quantization_toml: &str,
    ) -> LoaderResult<LoadedModel> {
        let downloader = HuggingFaceDownload::new();
        let bundle = match downloader.get_cached(model_id) {
            Some(b) => {
                log::info!("Using cached model: {}", model_id);
                b
            }
            None => {
                log::info!("Downloading model: {}", model_id);
                downloader.download_model(model_id)?
            }
        };

        let json_config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(bundle.config_path())?)?;
        let model_type = json_config["model_type"].as_str().unwrap_or("").to_string();
        let mut config = ModelConfig::from_json_value(&json_config)
            .map_err(|e| LoaderError::Config(e.to_string()))?;
        config.architecture = model_type.clone();

        log::info!(
            "  Config: arch={}, dim={}, layers={}, heads={}, vocab={}",
            if model_type.is_empty() { "gpt2" } else { &model_type },
            config.dim,
            config.n_layers,
            config.n_heads,
            config.vocab_size
        );

        let weights = SafeTensorsStore.load(&bundle.weights_path())?;
        log::info!("  {} tensors loaded", weights.len());

        // HuggingFace puts chat_template in tokenizer_config.json, not
        // config.json. Pull it in if present so the generator can apply
        // the right template at decode time. Without this, instruct-
        // tuned models receive un-templated prompts and emit garbage
        // — see investigation on gemma-3-1b-it 2026-04-13.
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
        // (e.g. `template.contains("<start_of_turn>")`), so the marker
        // alone routes to the right template branch. Older HF caches
        // downloaded before tokenizer_config.json was requested don't
        // include it; this keeps existing cached models servable.
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

        let mut model = self.registry
            .build_model(&config, weights)
            .map_err(|e| LoaderError::Model(format!("Failed to build {} model: {}",
                if model_type.is_empty() { "gpt2" } else { &model_type }, e)))?;
        model.set_optimization_profile(profile);

        // Post-construction optimization via pluggable providers.
        if !model.output.is_quantized() {
            let quantizer = ConfigQuantizer::from_toml_str(quantization_toml);
            match quantizer.quantize(&mut model) {
                Ok(n) if n > 0 => log::info!(
                    "  Quantized {} linear layers ({})",
                    n,
                    quantizer.describe()
                ),
                Ok(_) => {}
                Err(e) => log::warn!("  Weight quantization failed: {}", e),
            }
        }

        let gate_up_fuser = GateUpFuser;
        let fused = gate_up_fuser.fuse(&mut model);
        if fused > 0 {
            log::info!("  {}: {} pairs", gate_up_fuser.describe(), fused);
        }
        let qkv_fuser = QkvFuser;
        let fused_qkv = qkv_fuser.fuse(&mut model);
        if fused_qkv > 0 {
            log::info!("  {}: {} triples", qkv_fuser.describe(), fused_qkv);
        }

        warmup(&mut model);

        let (total_params, _) = model.parameter_count();
        log::info!("  Model ready: {:.1}M params", total_params as f64 / 1e6);

        // Diagnostic: log dtype of representative weights so we can
        // verify the [quantization] config actually took effect.
        log::info!(
            "  Weight dtypes after quantize: output.weight={:?}, layer0.attn.q_proj.weight={:?}",
            model.output.weight.dtype(),
            model.layers.first().map(|l| l.attention.q_proj.weight.dtype()),
        );

        let tokenizer_json = bundle.tokenizer_json_path();
        let tokenizer: Box<dyn Tokenizer + Send + Sync> = if tokenizer_json.exists() {
            let hf = HFTokenizer::from_file(&tokenizer_json)
                .map_err(|e| LoaderError::Tokenizer(format!("HFTokenizer load failed: {}", e)))?;
            log::info!("  Tokenizer: {} tokens (tokenizer.json)", hf.vocab_size());
            Box::new(hf)
        } else {
            let bpe = BpeTokenizer::from_files(bundle.vocab_path(), bundle.merges_path())
                .map_err(|e| LoaderError::Tokenizer(format!("BPE load failed: {}", e)))?;
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

        Ok(LoadedModel {
            model,
            tokenizer,
            model_id: model_id.to_string(),
            chat_template: config.chat_template.clone(),
            eos_token_id: eos,
            bos_token_id: config.bos_token_id,
            profile,
        })
    }

    fn load_gguf(&self, path: &Path, profile: OptProfile) -> LoaderResult<LoadedModel> {
        log::info!("Loading GGUF: {}", path.display());
        let gguf = GGUFFile::parse_header(path)
            .map_err(|e| LoaderError::Gguf(format!("parse header: {}", e)))?;
        log::info!("  GGUF v{}, {} tensors", gguf.version, gguf.tensor_infos.len());

        let gguf_config = gguf
            .to_model_config()
            .map_err(|e| LoaderError::Gguf(format!("extract config: {}", e)))?;
        let mut config = gguf_config_to_model_config(&gguf_config)
            .map_err(|e| LoaderError::Config(format!("gguf→model config: {}", e)))?;
        config.architecture = gguf_config.architecture.clone();

        log::info!(
            "  arch={}, dim={}, layers={}, heads={}, vocab={}",
            config.architecture,
            config.dim,
            config.n_layers,
            config.n_heads,
            config.vocab_size
        );

        let tokenizer: Box<dyn Tokenizer + Send + Sync> = Box::new(
            GgufTokenizer::from_gguf(&gguf)
                .map_err(|e| LoaderError::Tokenizer(format!("gguf tokenizer: {}", e)))?,
        );

        let arch = gguf_config.architecture.as_str();
        let loaded_tensors = match arch {
            "gemma3" => gguf
                .load_and_remap_gemma3(path, config.n_layers)
                .map_err(|e| LoaderError::Gguf(format!("load/remap gemma3: {}", e)))?,
            "nomic-bert" => gguf
                .load_and_remap_nomic_bert(path, config.n_layers)
                .map_err(|e| LoaderError::Gguf(format!("load/remap nomic-bert: {}", e)))?,
            _ => gguf
                .load_and_remap(path, config.n_layers)
                .map_err(|e| LoaderError::Gguf(format!("load/remap: {}", e)))?,
        };
        let tensors = convert_tensors(loaded_tensors);

        let mut model = self.registry
            .build_model(&config, tensors)
            .map_err(|e| LoaderError::Model(format!("Failed to build {} model: {}",
                config.architecture, e)))?;
        model.set_optimization_profile(profile);

        let qkv_fuser = QkvFuser;
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

        Ok(LoadedModel {
            model,
            tokenizer,
            model_id,
            chat_template: config.chat_template.clone(),
            eos_token_id: config.eos_token_id,
            bos_token_id: config.bos_token_id,
            profile,
        })
    }
}
