pub use crate::api::loader::LlamaCppBackendLoader;
// `LlamaCppModel` and `LlamaCppTextCompleter` are deliberately NOT
// re-exported — they're `pub(crate)` so the self-referential pool's
// safety invariants can't be broken by external code. See
// `core/model.rs` module docs.
pub use crate::core::model::load_llama_cpp_model;
