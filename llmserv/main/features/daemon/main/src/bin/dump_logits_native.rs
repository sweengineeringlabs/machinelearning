//! Dump first-token logits from the native_rust forward pass for a
//! fixed token sequence. Used to diff against the llama.cpp side
//! during P9 (gemma3 quality) investigation.
//!
//! Output format (stdout, one logit per line):
//!   LOGIT <vocab_idx> <float>
//! Plus a header with metadata (model, n_tokens, vocab_size).
//!
//! Run:
//!   cargo run --release --bin dump-logits-native > /tmp/native_logits.txt
//!
//! Reads the same XDG config the main daemon does, so configure
//! the model + quantization via application.toml.

use std::path::PathBuf;

use anyhow::{Result, bail};
use swe_llmmodel_model::{LanguageModel, OptProfile};
use swe_ml_tensor::{DType, Tensor};
use swellmd::{ModelSource, load_config, load_gguf, load_safetensors};

const FIXED_TOKENS: &[u32] = &[2, 9259, 235269]; // <bos> Hello ,

fn main() -> Result<()> {
    env_logger::init();

    let loaded = load_config()?;
    let cfg = &loaded.app;

    eprintln!(
        "[dump-logits-native] model.backend={:?} model.source={:?} model.id={:?} model.path={:?}",
        cfg.model.backend, cfg.model.source, cfg.model.id, cfg.model.path
    );

    let profile = OptProfile::Optimized;
    let model = match cfg.model.source {
        ModelSource::Safetensors => {
            let id = cfg
                .model
                .id
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("source = safetensors requires id"))?;
            load_safetensors(id, profile, &loaded.merged_toml)?
        }
        ModelSource::Gguf => {
            let path_str = cfg
                .model
                .path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("source = gguf requires path"))?;
            load_gguf(&PathBuf::from(path_str), profile)?
        }
    };

    let n_tokens = FIXED_TOKENS.len();
    let input_floats: Vec<f32> = FIXED_TOKENS.iter().map(|&t| t as f32).collect();
    let input_bytes: Vec<u8> = input_floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let input_tensor = Tensor::new(input_bytes, vec![1, n_tokens], DType::F32);

    let logits = model.model.forward(&input_tensor)?;

    let dims = logits.shape().to_vec();
    let total: usize = dims.iter().product();
    let last_pos_offset = total - dims[dims.len() - 1];
    let vocab_size = dims[dims.len() - 1];

    eprintln!(
        "[dump-logits-native] forward output shape={:?}, total={}, last_pos_offset={}, vocab_size={}",
        dims, total, last_pos_offset, vocab_size
    );

    let all: Vec<f32> = logits.iter().collect();
    if all.len() != total {
        bail!(
            "tensor.iter() yielded {} floats but shape implies {}",
            all.len(),
            total
        );
    }
    let last_pos = &all[last_pos_offset..];

    println!(
        "# dump-logits-native model={} n_tokens={} vocab_size={}",
        model.model_id, n_tokens, vocab_size
    );
    println!("# token_ids={:?}", FIXED_TOKENS);
    for (i, v) in last_pos.iter().enumerate() {
        println!("LOGIT {} {}", i, v);
    }
    Ok(())
}
