//! Public trait definitions.
//!
//! Define service traits here that core/ will implement.

use super::config::Config;
use super::error::Error;

/// Primary service trait for crates_quant_packer.
///
/// Implement this in core/ to define the crate's main behavior.
pub trait Service: Send + Sync {
    /// Execute the primary operation with the given configuration.
    fn execute(&self, config: &Config) -> Result<(), Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_trait_is_object_safe() {
        fn _accept(_s: &dyn Service) {}
    }
}
