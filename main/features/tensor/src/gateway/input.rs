// Gateway input types — raw data entering the tensor engine boundary

/// Marker type for gateway input validation.
pub(crate) struct GatewayInput;

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: GatewayInput
    #[test]
    fn test_gateway_input_exists() {
        let _input = GatewayInput;
    }
}
