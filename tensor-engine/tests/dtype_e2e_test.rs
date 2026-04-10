//! E2E tests for the DType enum.

use tensor_engine::DType;

/// @covers: DType::size
#[test]
fn test_dtype_sizes_match_expected() {
    assert_eq!(DType::F32.size(), 4);
    assert_eq!(DType::F16.size(), 2);
    assert_eq!(DType::BF16.size(), 2);
    assert_eq!(DType::I8.size(), 1);
    assert_eq!(DType::U8.size(), 1);
    assert_eq!(DType::Q8_0.size(), 0);
    assert_eq!(DType::Q4_0.size(), 0);
    assert_eq!(DType::Q4_1.size(), 0);
}

/// @covers: DType (Default)
#[test]
fn test_dtype_default_is_f32() {
    assert_eq!(DType::default(), DType::F32);
}
