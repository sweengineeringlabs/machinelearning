use crate::api::error::{GenerationError, GenerationResult};

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub do_sample: bool,
    pub repetition_penalty: f32,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            do_sample: true,
            repetition_penalty: 1.0,
            eos_token_id: Some(50256),
            pad_token_id: Some(50256),
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> GenerationResult<()> {
        if self.temperature < 0.0 {
            return Err(GenerationError::Generation(
                format!("temperature must be >= 0.0, got {}", self.temperature),
            ));
        }
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(GenerationError::Generation("top_k must be > 0".into()));
            }
        }
        if let Some(p) = self.top_p {
            if p <= 0.0 || p > 1.0 {
                return Err(GenerationError::Generation(format!("top_p must be in (0.0, 1.0], got {}", p)));
            }
        }
        if self.repetition_penalty <= 0.0 {
            return Err(GenerationError::Generation(format!("repetition_penalty must be > 0.0, got {}", self.repetition_penalty)));
        }
        Ok(())
    }

    pub fn greedy(max_new_tokens: usize) -> Self {
        Self { max_new_tokens, do_sample: false, ..Default::default() }
    }

    pub fn with_temperature(max_new_tokens: usize, temperature: f32) -> Self {
        Self { max_new_tokens, temperature, do_sample: true, ..Default::default() }
    }

    pub fn with_top_k(max_new_tokens: usize, top_k: usize, temperature: f32) -> Self {
        Self { max_new_tokens, temperature, top_k: Some(top_k), do_sample: true, ..Default::default() }
    }

    pub fn with_top_p(max_new_tokens: usize, top_p: f32, temperature: f32) -> Self {
        Self { max_new_tokens, temperature, top_p: Some(top_p), do_sample: true, ..Default::default() }
    }
}
