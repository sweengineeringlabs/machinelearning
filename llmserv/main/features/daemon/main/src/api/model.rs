use rustml_generation::TextCompleter;
use rustml_inference_layers::PoolingStrategy;
use rustml_model::ModelResult;
use rustml_tokenizer::Tokenizer;
use swe_ml_tensor::Tensor;

/// Interface for a loaded model instance ready for inference.
///
/// This is the daemon's DI seam for swapping backends. A native-Rust
/// backend, a `llama.cpp`-backed backend, a remote-HTTP proxy, or a test
/// mock can all satisfy this trait — the router code is coupled to the
/// trait, not to any concrete model type. No `Tensor` or `KVCache` types
/// appear on the `TextCompleter` returned by `open_text_completer`, so
/// backends that own their own decode state (like `llama.cpp`) aren't
/// forced to fake our concrete types.
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

    /// Compute embeddings for the given input with the specified pooling strategy.
    fn embed(&self, input_ids: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor>;
}
