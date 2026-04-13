use rustml_generation::TextCompleter;
use rustml_inference_layers::PoolingStrategy;
use rustml_model::ModelResult;
use rustml_tokenizer::Tokenizer;

/// Interface for a loaded model instance ready for inference.
///
/// This is the daemon's DI seam for swapping backends. A native-Rust
/// backend, a `llama.cpp`-backed backend, a remote-HTTP proxy, or a test
/// mock can all satisfy this trait — the router code is coupled to the
/// trait, not to any concrete model type. No `Tensor` or `KVCache` types
/// cross this trait surface, so backends that own their own internal
/// representations (like `llama.cpp`) aren't forced to fake our
/// concrete types.
pub trait Model: Send + Sync {
    /// Model identifier (e.g. HuggingFace model ID or GGUF filename).
    fn model_id(&self) -> &str;

    /// Open a per-request text completer. Cheap — borrows the model,
    /// allocates a small config struct. One call per incoming HTTP
    /// request; per-request sampling params are passed to the
    /// completer's `complete*` methods.
    fn open_text_completer(&self) -> Box<dyn TextCompleter + '_>;

    /// Access the tokenizer.
    fn tokenizer(&self) -> &dyn Tokenizer;

    /// Compute a pooled embedding for a single tokenized input.
    ///
    /// Returns a flat `Vec<f32>` whose length equals the model's
    /// embedding dimension — the pooling strategy collapses the sequence
    /// dimension. Backends that don't serve embeddings should return
    /// a `ModelError::Model("embeddings not supported by ...")`.
    ///
    /// Callers supply token IDs directly rather than a `Tensor` so the
    /// trait stays free of native-Rust internal types. Backends
    /// construct whatever internal representation they need themselves.
    fn embed(&self, token_ids: &[u32], strategy: PoolingStrategy) -> ModelResult<Vec<f32>>;
}
