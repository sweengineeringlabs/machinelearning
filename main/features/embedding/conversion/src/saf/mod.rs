pub use crate::api::error::{EmbeddingError, EmbeddingResult};
pub use crate::api::traits::{Embed, Normalize};
pub use crate::core::embedding::DefaultEmbedding;
pub use crate::core::normalize::L2Normalize;

// API types (OpenAI-compatible)
pub use crate::api::types::{
    EmbeddingData, EmbeddingInput, EmbeddingUsage, EmbeddingsRequest, EmbeddingsResponse,
};
