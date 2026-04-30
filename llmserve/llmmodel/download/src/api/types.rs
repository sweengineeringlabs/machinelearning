use std::path::PathBuf;

/// Files downloaded for a SafeTensors-backed model.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub model_id: String,
    pub model_dir: PathBuf,
}

impl ModelFiles {
    pub fn config_path(&self) -> PathBuf {
        self.model_dir.join("config.json")
    }

    pub fn weights_path(&self) -> PathBuf {
        self.model_dir.join("model.safetensors")
    }

    pub fn vocab_path(&self) -> PathBuf {
        self.model_dir.join("vocab.json")
    }

    pub fn merges_path(&self) -> PathBuf {
        self.model_dir.join("merges.txt")
    }

    pub fn tokenizer_json_path(&self) -> PathBuf {
        self.model_dir.join("tokenizer.json")
    }

    /// Holds the chat-template Jinja2 string for instruct-tuned models
    /// (separate from config.json which has architectural params only).
    pub fn tokenizer_config_path(&self) -> PathBuf {
        self.model_dir.join("tokenizer_config.json")
    }
}

/// A downloaded GGUF model file.
#[derive(Debug, Clone)]
pub struct GgufBundle {
    pub gguf_path: PathBuf,
    pub model_id: Option<String>,
}

impl GgufBundle {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            gguf_path: path.into(),
            model_id: None,
        }
    }
}
