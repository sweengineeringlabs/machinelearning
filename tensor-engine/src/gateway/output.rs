// Gateway output types — data leaving the tensor engine boundary

/// Marker type for gateway output validation.
pub(crate) struct GatewayOutput;

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: GatewayOutput
    #[test]
    fn test_gateway_output_exists() {
        let _output = GatewayOutput;
    }
}
