//! Content-correctness regression tests.
//!
//! These are the missing tests that would have caught the gemma3
//! garbage-output bug (P9) on day one instead of letting it sit
//! latent until a deploy session surfaced it. See the
//! `feedback_content_correctness_required` memory and the 2026-04-13
//! session log under `docs/0-ideation/discussion/`.
//!
//! # What these test
//!
//! For each backend (native_rust, llama_cpp), three fixed prompts are
//! sent. The response text MUST contain at least one of the expected
//! substrings. This catches:
//!
//! - Forward-pass numerical bugs (the kind P9 was) — output becomes
//!   garbage that doesn't contain the expected fact.
//! - Sampling bugs — argmax returns wrong indices, output drifts.
//! - Tokenizer/template bugs — model sees malformed input, produces
//!   off-prompt output.
//! - Catastrophic regressions in any layer of the stack between
//!   "request comes in" and "string goes out".
//!
//! What these do NOT test:
//!
//! - Latency or performance — separate concern, see `docs/5-testing/perf/`.
//! - Tail behavior at concurrency — content-correct ≠ scheduling-correct.
//! - Numerical exact-match against a reference — substring-match
//!   tolerates the FMA-ordering noise that's expected between
//!   different forward-pass implementations.
//!
//! # How to run
//!
//! Tests are gated by env vars + `--ignored` because they need a
//! real model loaded (no point running with an empty config). Local:
//!
//! ```bash
//! # native_rust path against an HF model id (downloads to cache if needed)
//! export LLMSERV_CC_HF_ID="google/gemma-3-1b-it"
//! cargo test -p swellmd --test content_correctness native_rust -- --ignored --nocapture
//!
//! # llama_cpp path against a local GGUF (e.g. Ollama's cache)
//! export LLMSERV_CC_GGUF="C:/Users/elvis/.ollama/models/blobs/sha256-..."
//! cargo test -p swellmd --features backend-llama-cpp \
//!     --test content_correctness llama_cpp -- --ignored --nocapture
//! ```
//!
//! # CI integration
//!
//! Adding the env vars to the CI job's environment turns these from
//! ignored→active. The CI runner needs disk space + network to
//! download (HF case) or a pre-staged GGUF (llama_cpp case). Worth
//! the cost — the alternative is shipping silent bugs.

use rustml_generation::CompletionParams;
use swe_llmmodel_loader::{DefaultLoader, LoadModel};
use swe_llmmodel_model::OptProfile;
use swellmd::{DefaultModel, Model};

/// Prompts and the substrings we accept as proof of a correct
/// answer. Each prompt has multiple acceptable substrings to
/// tolerate small variations in phrasing, capitalization, or
/// units (e.g. "221" vs "two hundred twenty-one").
///
/// Substring matching is case-insensitive (we lowercase both the
/// response and the expected before comparison).
const CASES: &[(&str, &[&str])] = &[
    ("What is the capital of France?", &["paris"]),
    ("What is 13 times 17?", &["221", "two hundred"]),
    ("Name the largest country in South America.", &["brazil"]),
];

const MAX_TOKENS: usize = 50;

fn assert_response_contains_expected(prompt: &str, response: &str, expected: &[&str]) {
    let response_lower = response.to_lowercase();
    let any_match = expected
        .iter()
        .any(|s| response_lower.contains(&s.to_lowercase()));
    assert!(
        any_match,
        "\nContent-correctness FAIL\n\
         prompt    : {:?}\n\
         response  : {:?}\n\
         expected  : one of {:?}\n\
         (substring matching is case-insensitive)\n",
        prompt, response, expected
    );
}

#[test]
#[ignore = "needs LLMSERV_CC_HF_ID env var (an HF model id like google/gemma-3-1b-it)"]
fn native_rust_chat_completes_correctly_on_known_prompts() {
    let model_id = std::env::var("LLMSERV_CC_HF_ID")
        .expect("LLMSERV_CC_HF_ID must be set to an HF model id");

    let loaded = DefaultLoader::new()
        .load_safetensors(&model_id, OptProfile::Optimized, "")
        .expect("load model from hub");
    let model: DefaultModel = loaded.into();

    let params = CompletionParams::new(0.0, MAX_TOKENS);

    for (prompt, expected) in CASES {
        let mut completer = model.open_text_completer();
        let messages: &[(&str, &str)] = &[("user", prompt)];
        let mut cb = |_t: u32| true;
        let response = completer
            .complete_turn_stream(messages, &params, &mut cb)
            .unwrap_or_else(|e| panic!("complete_turn_stream failed for {:?}: {}", prompt, e));

        eprintln!("[native_rust] Q: {}", prompt);
        eprintln!("[native_rust] A: {}", response.trim());
        assert_response_contains_expected(prompt, &response, expected);
    }
}

#[cfg(feature = "backend-llama-cpp")]
#[test]
#[ignore = "needs LLMSERV_CC_GGUF env var (path to a .gguf file)"]
fn llama_cpp_chat_completes_correctly_on_known_prompts() {
    use std::path::PathBuf;

    let gguf_path = std::env::var("LLMSERV_CC_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .expect("LLMSERV_CC_GGUF must be set to an existing .gguf file path");

    let spec = swellmd::ModelSpec {
        backend: swellmd::ModelBackend::LlamaCpp,
        source: swellmd::ModelSource::Gguf,
        id: None,
        path: Some(gguf_path.to_string_lossy().into_owned()),
    };
    let model =
        rustml_backend_llama_cpp::load_llama_cpp_model(&spec, OptProfile::Optimized, "")
            .expect("load gguf via llama_cpp");

    let params = CompletionParams::new(0.0, MAX_TOKENS);

    for (prompt, expected) in CASES {
        let mut completer = model.open_text_completer();
        let messages: &[(&str, &str)] = &[("user", prompt)];
        let mut cb = |_t: u32| true;
        let response = completer
            .complete_turn_stream(messages, &params, &mut cb)
            .unwrap_or_else(|e| panic!("complete_turn_stream failed for {:?}: {}", prompt, e));

        eprintln!("[llama_cpp] Q: {}", prompt);
        eprintln!("[llama_cpp] A: {}", response.trim());
        assert_response_contains_expected(prompt, &response, expected);
    }
}
