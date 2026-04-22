//! Dump first-token logits from the llama.cpp forward pass for a
//! fixed token sequence. Pair with `dump-logits-native` (in the
//! daemon crate) for the P9 logits-diff investigation.
//!
//! The fixed token sequence MUST match the native dumper's, so we
//! compare logits on the exact same input rather than on the same
//! prompt re-tokenized by two different tokenizers.
//!
//! Run:
//!   export LLMSERV_DUMP_GGUF="path/to/model.gguf"
//!   cargo test -p swe-llmserver-llamacpp --features llama-cpp \
//!     --test dump_logits dump_logits -- --ignored --nocapture \
//!     > /tmp/llamacpp_logits.txt

#![cfg(feature = "llama-cpp")]

use std::path::PathBuf;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

const FIXED_TOKENS: &[i32] = &[2, 9259, 235269]; // <bos> Hello ,

#[test]
#[ignore = "requires LLMSERV_DUMP_GGUF env var pointing at a .gguf file"]
fn dump_logits() {
    let gguf = std::env::var("LLMSERV_DUMP_GGUF")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .expect("LLMSERV_DUMP_GGUF must be set to an existing .gguf path");

    let backend = LlamaBackend::init().expect("LlamaBackend::init");
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &gguf, &params).expect("load gguf");

    let n_vocab = model.n_vocab();
    let n_tokens = FIXED_TOKENS.len();

    let ctx_params =
        LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(2048));
    let mut ctx = model.new_context(&backend, ctx_params).expect("new_context");

    let mut batch = LlamaBatch::new(n_tokens.max(8), 1);
    let last_idx = n_tokens - 1;
    for (i, &t) in FIXED_TOKENS.iter().enumerate() {
        batch
            .add(LlamaToken(t), i as i32, &[0], i == last_idx)
            .expect("batch.add");
    }
    ctx.decode(&mut batch).expect("ctx.decode");

    let logits = ctx.get_logits_ith(batch.n_tokens() - 1);
    eprintln!(
        "[dump-logits-llamacpp] gguf={} n_tokens={} n_vocab={} logits.len={}",
        gguf.display(),
        n_tokens,
        n_vocab,
        logits.len()
    );

    println!(
        "# dump-logits-llamacpp gguf={} n_tokens={} vocab_size={}",
        gguf.display(),
        n_tokens,
        logits.len()
    );
    println!("# token_ids={:?}", FIXED_TOKENS);
    for (i, v) in logits.iter().enumerate() {
        println!("LOGIT {} {}", i, v);
    }
}
