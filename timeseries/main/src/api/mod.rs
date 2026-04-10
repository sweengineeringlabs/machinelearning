// Re-export training-engine API for backwards compatibility.
// Since training_engine::api is private, we re-create the module structure
// using the public surface.

pub mod error {
    pub use training_engine::{SwetsError, SwetsResult};
}

pub mod layer {
    pub use training_engine::Layer;
}

pub mod loss {
    pub use training_engine::Loss;
}

pub mod optim {
    pub use training_engine::{LRScheduler, Optimizer};
}

pub use training_engine::pool;
pub use training_engine::tape;

pub mod tensor {
    pub use training_engine::{Tensor, TensorId};
}
