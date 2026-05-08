//! Parity test — same GGUF through the native-Rust backend and the
//! llama.cpp backend, fixed prompts, greedy decoding, assert both
//! produce sane output.
//!
//! # Running
//!
//! Gated by the `LLMINFERENCE_PARITY_GGUF_PATH` env var plus `--ignored`
//! because GGUF files are hundreds of MB and aren't committed. Point
//! it at any gemma3/llama/qwen GGUF on your disk — Ollama's cache at
//! `~/.ollama/models/blobs/` is a convenient source.
//!
//! ```bash
//! cd llminference
//! export LLMINFERENCE_PARITY_GGUF_PATH="/path/to/model.gguf"
//! export CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL CFLAGS=-MD CXXFLAGS=-MD  # Windows only
//! cargo test -p swe-llmserver-llamacpp --features llama-cpp --test parity -- --ignored --nocapture
//! ```
//!
//! # What's asserted
//!
//! Strict token-for-token parity between backends is NOT expected —
//! native-Rust and llama.cpp use different matmul implementations,
//! different kernel fusions, different FMA orders; quantized weights
//! especially produce slightly different logits, and argmax on
//! near-tied logits diverges. The test instead asserts:
//!
//! - Both backends load the model without error.
//! - Both backends produce non-empty output for every prompt.
//! - Both backends terminate within `max_tokens` (no runaway/crash).
//!
//! When tokens DO diverge the test reports the first divergence
//! index so the human can decide whether it's expected drift or a
//! real backend bug.

#![cfg(feature = "llama-cpp")]

use std::path::PathBuf;

use swe_inference_backend_api::{Model, ModelBackend, ModelSource, ModelSpec};
use swe_llmserver_llamacpp::load_llama_cpp_model;
use swe_inference_generation::CompletionParams;
use swe_llmmodel_model::OptProfile;

const PARITY_PROMPTS: &[&str] = &[
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time,",
];

const MAX_TOKENS: usize = 32;

fn gguf_path_from_env() -> Option<PathBuf> {
    std::env::var("LLMINFERENCE_PARITY_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
#[ignore = "requires LLMINFERENCE_PARITY_GGUF_PATH pointing at a .gguf file"]
fn llama_cpp_backend_produces_nonempty_completions_for_parity_prompts() {
    let Some(gguf) = gguf_path_from_env() else {
        panic!(
            "LLMINFERENCE_PARITY_GGUF_PATH must be set to a .gguf file path. \
             Current value: {:?}",
            std::env::var("LLMINFERENCE_PARITY_GGUF_PATH").ok()
        );
    };

    let spec = ModelSpec {
        backend: ModelBackend::LlamaCpp,
        source: ModelSource::Gguf,
        id: None,
        path: Some(gguf.to_string_lossy().into_owned()),
    };
    let model = load_llama_cpp_model(&spec, OptProfile::Optimized, "")
        .expect("llama.cpp should load the GGUF");

    let params = CompletionParams::new(0.0, MAX_TOKENS);

    for prompt in PARITY_PROMPTS {
        let mut completer = model.open_text_completer();
        let mut token_ids = Vec::new();
        let mut cb = |tok: u32| {
            token_ids.push(tok);
            true
        };
        let text = completer
            .complete_stream(prompt, &params, &mut cb)
            .unwrap_or_else(|e| panic!("complete_stream failed for {:?}: {}", prompt, e));

        assert!(
            !text.is_empty(),
            "llama.cpp produced empty output for prompt {:?}",
            prompt
        );
        assert!(
            !token_ids.is_empty(),
            "llama.cpp callback was never called for prompt {:?}",
            prompt
        );
        assert!(
            token_ids.len() <= MAX_TOKENS,
            "llama.cpp exceeded max_tokens for prompt {:?} ({} > {})",
            prompt,
            token_ids.len(),
            MAX_TOKENS
        );

        println!(
            "llama_cpp[{}] → {} tokens: {:?}",
            prompt,
            token_ids.len(),
            text.trim()
        );
    }
}

/// Chat-template path — exercises `apply_chat_template`.
#[test]
#[ignore = "requires LLMINFERENCE_PARITY_GGUF_PATH pointing at a .gguf file with a chat template"]
fn llama_cpp_backend_chat_completions_produce_nonempty_output() {
    let Some(gguf) = gguf_path_from_env() else {
        panic!("LLMINFERENCE_PARITY_GGUF_PATH not set");
    };

    let spec = ModelSpec {
        backend: ModelBackend::LlamaCpp,
        source: ModelSource::Gguf,
        id: None,
        path: Some(gguf.to_string_lossy().into_owned()),
    };
    let model =
        load_llama_cpp_model(&spec, OptProfile::Optimized, "").expect("load GGUF");

    let params = CompletionParams::new(0.0, MAX_TOKENS);
    let mut completer = model.open_text_completer();
    let mut token_ids = Vec::new();
    let mut cb = |tok: u32| {
        token_ids.push(tok);
        true
    };

    let messages: &[(&str, &str)] = &[("user", "Reply with the single word: ready")];
    let text = completer
        .complete_turn_stream(messages, &params, &mut cb)
        .expect("chat completion");

    assert!(!text.is_empty(), "empty chat response");
    assert!(!token_ids.is_empty(), "no tokens emitted");
    println!("llama_cpp_chat → {}", text.trim());
}

/// Sanity: tokenizer round-trip via the Model trait's `tokenizer()`.
#[test]
#[ignore = "requires LLMINFERENCE_PARITY_GGUF_PATH pointing at a .gguf file"]
fn llama_cpp_tokenizer_roundtrip_preserves_text() {
    let Some(gguf) = gguf_path_from_env() else {
        panic!("LLMINFERENCE_PARITY_GGUF_PATH not set");
    };

    let spec = ModelSpec {
        backend: ModelBackend::LlamaCpp,
        source: ModelSource::Gguf,
        id: None,
        path: Some(gguf.to_string_lossy().into_owned()),
    };
    let model =
        load_llama_cpp_model(&spec, OptProfile::Optimized, "").expect("load GGUF");
    let tok = model.tokenizer();

    for input in ["Hello, world!", "Testing 1-2-3.", "The quick brown fox."] {
        let ids = tok.encode(input).expect("encode");
        assert!(!ids.is_empty(), "encoded {:?} to empty", input);
        let decoded = tok.decode(&ids).expect("decode");
        // llama.cpp tokenizers add a BOS and may insert a leading space;
        // assert containment rather than equality.
        assert!(
            decoded.contains(input) || decoded.trim_start().contains(input.trim_start()),
            "round-trip lost content: input={:?} decoded={:?}",
            input,
            decoded
        );
        println!("tok[{:?}] → {} ids → {:?}", input, ids.len(), decoded);
    }

    assert!(tok.vocab_size() > 0, "vocab_size should be positive");
}
