pub use crate::api::error::{EmbeddingError, EmbeddingResult};
pub use crate::api::traits::Embed;
pub use crate::core::embedding::DefaultEmbedding;
pub use crate::core::normalize::l2_normalize;

// API types (OpenAI-compatible)
pub use crate::api::types::{
    EmbeddingData, EmbeddingInput, EmbeddingUsage, EmbeddingsRequest, EmbeddingsResponse,
};
