// Re-export training-engine API for backwards compatibility.
// Since swe_ml_training::api is private, we re-create the module structure
// using the public surface.

pub mod error {
    pub use swe_ml_training::{SwetsError, SwetsResult};
}

pub mod layer {
    pub use swe_ml_training::Layer;
}

pub mod loss {
    pub use swe_ml_training::Loss;
}

pub mod optim {
    pub use swe_ml_training::{LRScheduler, Optimizer};
}

pub use swe_ml_training::pool;
pub use swe_ml_training::tape;

pub mod tensor {
    pub use swe_ml_training::{Tensor, TensorId};
}
