/// The type of normalization to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalerType {
    /// Min-Max normalization to [0, 1]:  `(x - min) / (max - min)`
    MinMax,
    /// Standardization (z-score):  `(x - mean) / std`
    Standard,
    /// Robust scaling using median and IQR:  `(x - median) / (Q75 - Q25)`
    Robust,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaler_type_equality() {
        assert_eq!(ScalerType::MinMax, ScalerType::MinMax);
        assert_ne!(ScalerType::MinMax, ScalerType::Standard);
        assert_ne!(ScalerType::Standard, ScalerType::Robust);
    }

    #[test]
    fn test_scaler_type_clone() {
        let st = ScalerType::Robust;
        let cloned = st;
        assert_eq!(cloned, ScalerType::Robust);
    }
}
