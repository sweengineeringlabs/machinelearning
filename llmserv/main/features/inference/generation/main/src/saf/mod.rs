//! Facade re-exports for swe-inference-generation

pub use crate::api::completer::{CompletionParams, TextCompleter};
pub use crate::api::error::*;
pub use crate::api::types::*;
pub use crate::core::generator::Generator;
pub use crate::core::sampling::{
    apply_repetition_penalty, apply_top_k, apply_top_p, apply_top_p_buffered,
    argmax, compute_log_probs, sample_categorical, top_n_indices, SamplingBuffer,
};
