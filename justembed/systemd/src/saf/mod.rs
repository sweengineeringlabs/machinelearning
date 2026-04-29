pub use crate::api::config::{
    AppConfig, EmbeddingGrpcConfig, EmbeddingGrpcTlsConfig,
};
pub use crate::api::proto::{EmbedRequest, EmbedResponse, FloatVector, EMBED_METHOD_PATH};
pub use crate::core::config::load as load_config;
pub use crate::core::embed::{EmbedError, EmbedOutcome, embed_inputs};
pub use crate::core::grpc_handler::EmbedHandler;
pub use crate::core::grpc_inbound::RegistryGrpcInbound;
pub use crate::core::grpc_server::{EmbedGrpcServer, start_grpc_server};
pub use crate::core::loader::load_gguf;
pub use crate::core::state::EmbeddingState;

// Shared systemd helpers — re-exported so the binary doesn't need to
// depend on swe-systemd directly.
pub use swe_systemd::{LoadedConfig, apply_logging_filter};
