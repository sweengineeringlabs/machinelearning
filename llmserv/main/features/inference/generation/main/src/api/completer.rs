//! Text-completion contract — the backend-agnostic surface the daemon
//! calls to produce text from a model.
//!
//! This trait replaces the concrete `Generator` struct at the daemon-model
//! boundary. `Generator` remains available for direct consumers (CLI,
//! examples) that build it explicitly; the trait exists so that alternative
//! backends (llama.cpp, remote HTTP proxies, mocks) can satisfy the same
//! daemon contract without leaking our concrete `Tensor` or `KVCache`
//! types across the boundary.

use crate::api::error::GenerationResult;
use std::time::Instant;

/// Per-request sampling/decode parameters.
///
/// Model-level defaults (EOS/BOS token IDs, chat template, context length,
/// optimization profile) are owned by the completer implementation and are
/// NOT part of this struct — they don't vary per request.
#[derive(Debug, Clone)]
pub struct CompletionParams {
    /// Sampling temperature. `0.0` selects greedy decoding.
    pub temperature: f32,
    /// Maximum number of tokens to generate (excluding the prompt).
    pub max_tokens: usize,
    /// Optional top-k sampling cutoff.
    pub top_k: Option<usize>,
    /// Optional top-p (nucleus) sampling cutoff.
    pub top_p: Option<f32>,
    /// Optional repetition penalty factor.
    pub repetition_penalty: Option<f32>,
    /// Optional wall-clock deadline. Decode halts when reached.
    pub deadline: Option<Instant>,
}

impl CompletionParams {
    /// Construct with the minimum required fields and no sampling cutoffs.
    pub fn new(temperature: f32, max_tokens: usize) -> Self {
        Self {
            temperature,
            max_tokens,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            deadline: None,
        }
    }
}

/// Produces text from a language model.
///
/// Implementors own their own decode state (KV cache, sampler, tokenizer
/// bindings) internally — none of those types cross the trait surface.
/// Methods take `&mut self` because decoding mutates internal state;
/// callbacks are `&mut dyn FnMut` rather than a generic `F: FnMut` to keep
/// the trait `dyn`-compatible.
pub trait TextCompleter: Send {
    /// Generate a completion for a raw prompt. Returns the full generated
    /// text (without the prompt).
    fn complete(
        &mut self,
        prompt: &str,
        params: &CompletionParams,
    ) -> GenerationResult<String>;

    /// Generate a completion with a per-token streaming callback. The
    /// callback receives each new token ID and returns `false` to stop
    /// early. Returns the full generated text.
    fn complete_stream(
        &mut self,
        prompt: &str,
        params: &CompletionParams,
        callback: &mut dyn FnMut(u32) -> bool,
    ) -> GenerationResult<String>;

    /// Generate a chat-turn completion. The implementation applies the
    /// model's chat template internally and returns only the assistant's
    /// new response (the prompt is stripped). Callback receives each new
    /// token ID and returns `false` to stop early.
    fn complete_turn_stream(
        &mut self,
        messages: &[(&str, &str)],
        params: &CompletionParams,
        callback: &mut dyn FnMut(u32) -> bool,
    ) -> GenerationResult<String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_params_new_sets_required_fields_only() {
        let p = CompletionParams::new(0.7, 128);
        assert_eq!(p.temperature, 0.7);
        assert_eq!(p.max_tokens, 128);
        assert!(p.top_k.is_none());
        assert!(p.top_p.is_none());
        assert!(p.repetition_penalty.is_none());
        assert!(p.deadline.is_none());
    }

    #[test]
    fn test_completion_params_clone_preserves_all_fields() {
        let p = CompletionParams {
            temperature: 0.5,
            max_tokens: 64,
            top_k: Some(40),
            top_p: Some(0.9),
            repetition_penalty: Some(1.1),
            deadline: None,
        };
        let q = p.clone();
        assert_eq!(q.temperature, 0.5);
        assert_eq!(q.max_tokens, 64);
        assert_eq!(q.top_k, Some(40));
        assert_eq!(q.top_p, Some(0.9));
        assert_eq!(q.repetition_penalty, Some(1.1));
    }
}
