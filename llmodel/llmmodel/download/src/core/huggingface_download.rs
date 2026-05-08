use std::path::{Path, PathBuf};

use crate::api::error::{DownloadError, DownloadResult};
use crate::api::traits::Download;
use crate::api::types::{GgufBundle, ModelFiles};

/// HuggingFace-Hub–backed downloader.
#[derive(Debug, Clone)]
pub struct HuggingFaceDownload {
    base_url: String,
    cache_dir: PathBuf,
    token: Option<String>,
}

impl Default for HuggingFaceDownload {
    fn default() -> Self {
        Self::new()
    }
}

impl HuggingFaceDownload {
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("rustml")
            .join("hub");

        let token = std::env::var("HF_TOKEN").ok();

        Self {
            base_url: "https://huggingface.co".to_string(),
            cache_dir,
            token,
        }
    }

    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            ..Self::new()
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Async download of a full SafeTensors model (reqwest path).
    pub async fn download_model_async(&self, model_id: &str) -> DownloadResult<ModelFiles> {
        let model_dir = self.cache_dir.join(model_id.replace('/', "--"));
        tokio::fs::create_dir_all(&model_dir).await?;

        let files = [
            "config.json",
            "model.safetensors",
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
        ];

        for file in files {
            let file_path = model_dir.join(file);
            if !file_path.exists() {
                self.download_file_async(model_id, file, &file_path).await?;
            }
        }

        Ok(ModelFiles {
            model_id: model_id.to_string(),
            model_dir,
        })
    }

    async fn download_file_async(
        &self,
        model_id: &str,
        filename: &str,
        dest: &Path,
    ) -> DownloadResult<()> {
        let url = format!(
            "{}/{}/resolve/main/{}",
            self.base_url, model_id, filename
        );

        let client = reqwest::Client::new();
        let mut request = client.get(&url);

        if let Some(ref token) = self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(|e| {
            DownloadError::Network(format!("Failed to download {}: {}", filename, e))
        })?;

        if !response.status().is_success() {
            if filename == "model.safetensors" || filename == "tokenizer.json" {
                return Ok(());
            }
            return Err(DownloadError::Network(format!(
                "Failed to download {}: HTTP {}",
                filename,
                response.status()
            )));
        }

        let bytes = response.bytes().await.map_err(|e| {
            DownloadError::Network(format!("Failed to read response: {}", e))
        })?;

        tokio::fs::write(dest, &bytes).await?;
        Ok(())
    }

    fn hf_sync_api(&self) -> DownloadResult<hf_hub::api::sync::Api> {
        match self.token {
            Some(ref t) => hf_hub::api::sync::ApiBuilder::new()
                .with_token(Some(t.clone()))
                .build(),
            None => hf_hub::api::sync::Api::new(),
        }
        .map_err(|e| DownloadError::Network(format!("Failed to create hf-hub API: {}", e)))
    }

    fn link_to_cache(&self, model_id: &str, model_dir: &Path) {
        let cache_entry = self.cache_dir.join(model_id.replace('/', "--"));
        if cache_entry.exists() {
            return;
        }
        let _ = std::fs::create_dir_all(&self.cache_dir);
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(model_dir, &cache_entry);
        }
        #[cfg(windows)]
        {
            let _ = std::os::windows::fs::symlink_dir(model_dir, &cache_entry);
        }
    }
}

impl Download for HuggingFaceDownload {
    fn download_model(&self, model_id: &str) -> DownloadResult<ModelFiles> {
        let api = self.hf_sync_api()?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json").map_err(|e| {
            DownloadError::Network(format!("Failed to download config.json: {}", e))
        })?;

        let model_dir = config_path.parent().unwrap_or(&config_path).to_path_buf();

        let _weights = repo.get("model.safetensors").ok();
        let _vocab = repo.get("vocab.json").ok();
        let _merges = repo.get("merges.txt").ok();
        let _tokenizer = repo.get("tokenizer.json").ok();
        let _tokenizer_config = repo.get("tokenizer_config.json").ok();

        self.link_to_cache(model_id, &model_dir);

        Ok(ModelFiles {
            model_id: model_id.to_string(),
            model_dir,
        })
    }

    fn download_gguf(&self, model_id: &str, filename: &str) -> DownloadResult<GgufBundle> {
        let api = self.hf_sync_api()?;
        let repo = api.model(model_id.to_string());
        let gguf_path = repo.get(filename).map_err(|e| {
            DownloadError::Network(format!("Failed to download {}: {}", filename, e))
        })?;

        let gguf_dir = gguf_path.parent().unwrap_or(&gguf_path).to_path_buf();
        self.link_to_cache(model_id, &gguf_dir);

        Ok(GgufBundle {
            gguf_path,
            model_id: Some(model_id.to_string()),
        })
    }

    fn is_cached(&self, model_id: &str) -> bool {
        let model_dir = self.cache_dir.join(model_id.replace('/', "--"));
        model_dir.exists() && model_dir.join("config.json").exists()
    }

    fn get_cached(&self, model_id: &str) -> Option<ModelFiles> {
        if self.is_cached(model_id) {
            Some(ModelFiles {
                model_id: model_id.to_string(),
                model_dir: self.cache_dir.join(model_id.replace('/', "--")),
            })
        } else {
            None
        }
    }

    fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_has_non_empty_cache_dir() {
        let dl = HuggingFaceDownload::new();
        assert!(!dl.cache_dir().as_os_str().is_empty());
    }

    #[test]
    fn test_with_cache_dir_override() {
        let dl = HuggingFaceDownload::with_cache_dir("/tmp/test-cache");
        assert_eq!(dl.cache_dir(), Path::new("/tmp/test-cache"));
    }

    #[test]
    fn test_is_cached_false_for_unknown_model() {
        let dl = HuggingFaceDownload::with_cache_dir("/tmp/empty-cache-xyz-does-not-exist");
        assert!(!dl.is_cached("nonexistent/model"));
    }

    #[test]
    fn test_get_cached_none_for_unknown_model() {
        let dl = HuggingFaceDownload::with_cache_dir("/tmp/empty-cache-xyz-does-not-exist");
        assert!(dl.get_cached("nonexistent/model").is_none());
    }
}
