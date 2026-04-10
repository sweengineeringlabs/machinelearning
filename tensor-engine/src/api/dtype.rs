/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    #[default]
    F32,
    F16,
    BF16,
    I8,
    U8,
    /// Block-quantized 8-bit: 32 elements/block, 34 bytes/block
    Q8_0,
    /// Block-quantized 4-bit: 32 elements/block, 18 bytes/block
    Q4_0,
    /// Block-quantized 4-bit with min: 32 elements/block, 20 bytes/block
    Q4_1,
}

impl DType {
    /// Per-element byte size. Returns 0 for block-quantized types.
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
            DType::Q8_0 => 0,
            DType::Q4_0 => 0,
            DType::Q4_1 => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: size
    #[test]
    fn test_dtype_size_f32_is_four_bytes() {
        assert_eq!(DType::F32.size(), 4);
    }

    /// @covers: size
    #[test]
    fn test_dtype_size_f16_is_two_bytes() {
        assert_eq!(DType::F16.size(), 2);
    }

    /// @covers: size
    #[test]
    fn test_dtype_size_bf16_is_two_bytes() {
        assert_eq!(DType::BF16.size(), 2);
    }

    /// @covers: size
    #[test]
    fn test_dtype_size_i8_is_one_byte() {
        assert_eq!(DType::I8.size(), 1);
    }

    /// @covers: size
    #[test]
    fn test_dtype_size_u8_is_one_byte() {
        assert_eq!(DType::U8.size(), 1);
    }

    /// @covers: size
    #[test]
    fn test_dtype_size_quantized_types_return_zero() {
        assert_eq!(DType::Q8_0.size(), 0);
        assert_eq!(DType::Q4_0.size(), 0);
        assert_eq!(DType::Q4_1.size(), 0);
    }

    /// @covers: size
    #[test]
    fn test_dtype_default_is_f32() {
        assert_eq!(DType::default(), DType::F32);
    }

    /// @covers: size
    #[test]
    fn test_dtype_clone_preserves_equality() {
        let d = DType::BF16;
        assert_eq!(d, d.clone());
    }
}
