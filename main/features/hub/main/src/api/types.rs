//! Data types for hub API operations

use std::path::PathBuf;

/// A bundle of downloaded model files (SafeTensors format)
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// Model identifier
    pub model_id: String,
    /// Path to the model directory
    pub model_dir: PathBuf,
}

impl ModelFiles {
    /// Get path to config.json
    pub fn config_path(&self) -> PathBuf {
        self.model_dir.join("config.json")
    }

    /// Get path to model weights (SafeTensors format)
    pub fn weights_path(&self) -> PathBuf {
        self.model_dir.join("model.safetensors")
    }

    /// Get path to vocab.json
    pub fn vocab_path(&self) -> PathBuf {
        self.model_dir.join("vocab.json")
    }

    /// Get path to merges.txt
    pub fn merges_path(&self) -> PathBuf {
        self.model_dir.join("merges.txt")
    }

    /// Get path to tokenizer.json (HuggingFace universal tokenizer)
    pub fn tokenizer_json_path(&self) -> PathBuf {
        self.model_dir.join("tokenizer.json")
    }

    /// Get path to tokenizer_config.json. Holds the chat-template
    /// Jinja2 string for instruct-tuned models (separate from
    /// config.json which has architectural params only).
    pub fn tokenizer_config_path(&self) -> PathBuf {
        self.model_dir.join("tokenizer_config.json")
    }

}

/// A bundle for GGUF model files
#[derive(Debug, Clone)]
pub struct GgufBundle {
    /// Path to the GGUF file
    pub gguf_path: PathBuf,
    /// Model identifier (if downloaded from hub)
    pub model_id: Option<String>,
}

impl GgufBundle {
    /// Create from a local GGUF file path
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            gguf_path: path.into(),
            model_id: None,
        }
    }
}
