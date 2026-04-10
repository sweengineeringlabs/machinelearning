//! Neural network traits

use crate::api::error::NnResult;
use rustml_core::Tensor;

/// Base attention trait
pub trait Attention {
    /// Compute attention output
    fn forward(&self, x: &Tensor) -> NnResult<Tensor>;
}

/// Trait for layers that support parameter freezing (for fine-tuning).
pub trait Freezable {
    fn is_frozen(&self) -> bool;
    fn freeze(&mut self);
    fn unfreeze(&mut self);
}

/// Per-layer input injection (e.g. Gemma 4 PLE).
///
/// Given token indices and the current hidden state, produce an updated
/// hidden state that incorporates per-layer embedding information.
pub trait PerLayerInput {
    /// Inject per-layer information for the given layer index.
    ///
    /// * `indices` - token IDs `[B, S]`
    /// * `hidden`  - current hidden state `[B, S, D]`
    /// * `layer`   - which transformer layer is requesting the injection
    ///
    /// Returns the updated hidden state `[B, S, D]`.
    fn inject(&self, indices: &Tensor, hidden: &Tensor, layer: usize) -> NnResult<Tensor>;

    /// Number of layers this input covers.
    fn num_layers(&self) -> usize;
}
