// Public facade — re-exports the full API surface.

pub use crate::api::error::{EmbeddingError, EmbeddingResult};
pub use crate::api::traits::Embed;
pub use crate::core::embedding::DefaultEmbedding;

// API types (OpenAI-compatible)
pub use crate::api::types::{
    EmbeddingData, EmbeddingInput, EmbeddingUsage, EmbeddingsRequest, EmbeddingsResponse,
};
